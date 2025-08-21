#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo.py — Batch T2V over HF models with resume-safe GCS uploads.

- You specify FULL HF model IDs via --models (comma-separated or single).
- Slice prompts by --batch-id & --batch-size (e.g., id=1,size=1000 => [1000,2000)).
- Skip if target GCS object already exists (resume).

Families supported (strict per docs):
  * Wan 2.1 / 2.2 T2V (1.3B / 14B / A14B) -> WanPipeline + AutoencoderKLWan(subfolder='vae')
  * SkyReels-V2 (T2V / DF)                -> SkyReelsV2Pipeline / SkyReelsV2DiffusionForcingPipeline
                                             + AutoencoderKLWan + UniPCMultistepScheduler(flow_shift=8.0)
  * LTX-Video (stable)                     -> LTXPipeline (direct T2V)
  * LTX-Video 0.9.7-dev / -distilled       -> LTXConditionPipeline + LTXLatentUpsamplePipeline (2-stage)
  * CogVideoX (2B / 5B)                    -> CogVideoXPipeline
  * ModelScope / ZeroScope T2V             -> DiffusionPipeline(TextToVideo)
  * Mochi-1 preview                        -> MochiPipeline (bf16 variant + autocast)
  * HunyuanVideo                           -> HunyuanVideoPipeline
  * Latte                                  -> LattePipeline
  * AnimateLCM                             -> MotionAdapter + AnimateDiffPipeline + LCMScheduler
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import List, Optional

import torch
from diffusers.utils import export_to_video
from google.cloud import storage


# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts-file", required=True, help="TXT file, one prompt per line")
    p.add_argument("--batch-id", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--models", required=True, help="Full HF repo id(s), comma-separated allowed")
    p.add_argument("--gcs-bucket", required=True)
    p.add_argument("--gcs-prefix", required=True)

    # optional overrides
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    # AnimateLCM needs a base SD checkpoint
    p.add_argument("--animatelcm-base", type=str, default="emilianJR/epiCRealism",
                   help="Base checkpoint repo for AnimateLCM (e.g., emilianJR/epiCRealism)")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--local-dir", default="./_t2v_tmp")
    p.add_argument("--gcs-key", default=None, help="Optional path to service-account JSON")
    return p.parse_args()


# ------------------------- IO helpers -------------------------

def read_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    return [x for x in lines if x.strip()]


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_").replace(" ", "_")


def gcs_bucket(bucket_name: str, gcs_key: Optional[str]) -> storage.Bucket:
    client = storage.Client.from_service_account_json(gcs_key) if gcs_key else storage.Client()
    return client.bucket(bucket_name)


def gcs_exists(bucket: storage.Bucket, path: str) -> bool:
    return bucket.blob(path).exists()


def gcs_upload(bucket: storage.Bucket, src: str, dst_path: str):
    blob = bucket.blob(dst_path)
    blob.chunk_size = 16 * 1024 * 1024
    blob.upload_from_filename(src)


# ------------------------- Defaults per family -------------------------

class Cfg:
    def __init__(self, h=None, w=None, f=None, steps=None, gs=None, fps=None):
        self.h, self.w, self.f, self.steps, self.gs, self.fps = h, w, f, steps, gs, fps


def defaults_for(model_id: str) -> Cfg:
    mid = model_id.lower()
    if "cogvideox" in mid:
        return Cfg(480, 768, 81, 50, 6.0, 16)
    if "hunyuanvideo" in mid:
        return Cfg(720, 1280, 61, 30, 6.0, 15)
    if "ltx-video-0.9.7" in mid:
        return Cfg(768, 1152, 161, None, 5.0, 24)
    if "ltx-video" in mid:
        return Cfg(512, 768, 161, 50, 5.0, 24)
    if "wan" in mid:
        return Cfg(480, 848, 81, 30, 5.0, 16)
    if "skyreels-v2" in mid:
        return Cfg(544, 960, 97, 30, 6.0, 24)
    if "zeroscope" in mid or "text-to-video-ms" in mid or "text_to_video" in mid:
        return Cfg(None, None, 16, 25, 7.5, 8)
    if "mochi" in mid:
        return Cfg(480, 848, 85, None, None, 30)
    if "latte" in mid:
        # Latte 常用 480p/720p，帧数依模型/显存而定
        return Cfg(480, 848, 97, 30, 6.0, 24)
    if "animatelcm" in mid:
        # AnimateLCM 走 SD 基座 + MotionAdapter，24~30fps 常见
        return Cfg(512, 512, 64, 6, None, 24)
    return Cfg(None, None, 32, 30, 6.0, 12)


def merge_cfg(base: Cfg, o: Cfg) -> Cfg:
    return Cfg(
        o.h if o.h is not None else base.h,
        o.w if o.w is not None else base.w,
        o.f if o.f is not None else base.f,
        o.steps if o.steps is not None else base.steps,
        o.gs if o.gs is not None else base.gs,
        o.fps if o.fps is not None else base.fps,
    )


# ------------------------- Utils -------------------------

def _bf16():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _fp16():
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _ltx_round_to_vae_ratio(pipeline, height, width):
    r = getattr(pipeline, "vae_temporal_compression_ratio", None)
    if not r:
        return height, width
    return height - (height % r), width - (width % r)


# ------------------------- Runners (families) -------------------------

def run_cogvideox(model_id, device, prompt, cfg, gen=None):
    from diffusers import CogVideoXPipeline
    pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=_bf16())
    pipe.to(device)
    out = pipe(prompt=prompt,
               num_frames=cfg.f or 81,
               num_inference_steps=cfg.steps or 50,
               guidance_scale=cfg.gs or 6.0,
               height=cfg.h or 480, width=cfg.w or 768,
               generator=gen)
    return out.frames[0], (cfg.fps or 16)


def run_wan_t2v(model_id, device, prompt, cfg, gen=None):
    mid = model_id.lower()
    if any(x in mid for x in ["-i2v-", "-ti2v-", "-vace-", "-flf2v-"]):
        raise RuntimeError("Wan repo is not T2V; please use a *-T2V-* model id.")
    from diffusers import WanPipeline, AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=_bf16())
    pipe.to(device)
    out = pipe(prompt=prompt,
               height=cfg.h or 480, width=cfg.w or 848,
               num_frames=cfg.f or 81,
               num_inference_steps=cfg.steps or 30,
               guidance_scale=cfg.gs or 5.0,
               generator=gen)
    return out.frames[0], (cfg.fps or 16)


def run_skyreels(model_id, device, prompt, cfg, gen=None):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    if "-df-" in model_id.lower():
        from diffusers import SkyReelsV2DiffusionForcingPipeline as SRPipe
    else:
        from diffusers import SkyReelsV2Pipeline as SRPipe
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = SRPipe.from_pretrained(model_id, vae=vae, torch_dtype=_bf16())
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
    pipe = pipe.to(device)
    kwargs = dict(
        prompt=prompt,
        height=cfg.h or 544, width=cfg.w or 960,
        num_frames=cfg.f or 97,
        num_inference_steps=cfg.steps or (30 if "-df-" in model_id.lower() else 50),
        guidance_scale=cfg.gs or 6.0,
        generator=gen,
    )
    if "-df-" in model_id.lower():
        kwargs.update(dict(ar_step=5, causal_block_size=5, overlap_history=None, addnoise_condition=20))
    out = pipe(**kwargs)
    return out.frames[0], (cfg.fps or 24)


def run_ltx_base(model_id, device, prompt, cfg, gen=None):
    from diffusers import LTXPipeline
    pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=_bf16())
    pipe.to(device)
    out = pipe(
        prompt=prompt,
        negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
        width=cfg.w or 768, height=cfg.h or 512,
        num_frames=cfg.f or 161,
        num_inference_steps=cfg.steps or 50,
        decode_timestep=0.03, decode_noise_scale=0.025,
        generator=gen,
    )
    return out.frames[0], (cfg.fps or 24)


def run_ltx_097_chain(model_id, device, prompt, cfg, gen=None, distilled=False):
    from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
    pipe = LTXConditionPipeline.from_pretrained(model_id, torch_dtype=_bf16())
    upsampler_id = "Lightricks/ltxv-spatial-upscaler-0.9.7"
    pipe_up = LTXLatentUpsamplePipeline.from_pretrained(upsampler_id, vae=pipe.vae, torch_dtype=_bf16())
    pipe.to(device); pipe_up.to(device); pipe.vae.enable_tiling()

    expected_h, expected_w = (cfg.h or 768), (cfg.w or 1152)
    downscale = 2 / 3
    h0, w0 = int(expected_h * downscale), int(expected_w * downscale)
    h0, w0 = _ltx_round_to_vae_ratio(pipe, h0, w0)
    f = cfg.f or 161
    neg = "worst quality, inconsistent motion, blurry, jittery, distorted"

    if distilled:
        base = dict(prompt=prompt, negative_prompt=neg, width=w0, height=h0, num_frames=f,
                    timesteps=[1000, 993, 987, 981, 975, 909, 725, 0.03],
                    decode_timestep=0.05, decode_noise_scale=0.025, image_cond_noise_scale=0.0,
                    guidance_scale=1.0, guidance_rescale=0.7, generator=gen, output_type="latent")
        latents = pipe(**base).frames
        up = pipe_up(latents=latents, adain_factor=1.0, output_type="latent").frames
        h1, w1 = h0 * 2, w0 * 2
        refine = dict(prompt=prompt, negative_prompt=neg, width=w1, height=h1, num_frames=f,
                      denoise_strength=0.999, timesteps=[1000, 909, 725, 421, 0],
                      latents=up, decode_timestep=0.05, decode_noise_scale=0.025,
                      image_cond_noise_scale=0.0, guidance_scale=1.0, guidance_rescale=0.7,
                      generator=gen, output_type="pil")
        frames = pipe(**refine).frames[0]
    else:
        base = dict(prompt=prompt, negative_prompt=neg, width=w0, height=h0, num_frames=f,
                    num_inference_steps=30, decode_timestep=0.05, decode_noise_scale=0.025,
                    image_cond_noise_scale=0.0, guidance_scale=cfg.gs or 5.0, guidance_rescale=0.7,
                    generator=gen, output_type="latent")
        latents = pipe(**base).frames
        up = pipe_up(latents=latents, output_type="latent").frames
        h1, w1 = h0 * 2, w0 * 2
        refine = dict(prompt=prompt, negative_prompt=neg, width=w1, height=h1, num_frames=f,
                      denoise_strength=0.4, num_inference_steps=10,
                      latents=up, decode_timestep=0.05, decode_noise_scale=0.025,
                      image_cond_noise_scale=0.0, guidance_scale=cfg.gs or 5.0, guidance_rescale=0.7,
                      generator=gen, output_type="pil")
        frames = pipe(**refine).frames[0]

    frames = [frm.resize((expected_w, expected_h)) for frm in frames]
    return frames, (cfg.fps or 24)


def run_mochi(model_id, device, prompt, cfg, gen=None):
    from diffusers import MochiPipeline
    pipe = MochiPipeline.from_pretrained(model_id, variant="bf16", torch_dtype=_bf16())
    pipe.enable_model_cpu_offload(); pipe.enable_vae_tiling()
    with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
        out = pipe(prompt=prompt,
                   num_frames=cfg.f or 85,
                   height=cfg.h or 480, width=cfg.w or 848,
                   generator=gen)
    return out.frames[0], (cfg.fps or 30)


def run_modelscope_or_zeroscope(model_id, device, prompt, cfg, gen=None):
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=_fp16(),
        variant="fp16" if "text-to-video-ms" in model_id.lower() else None
    )
    pipe.enable_model_cpu_offload()
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    out = pipe(prompt,
               num_frames=cfg.f or 16,
               num_inference_steps=cfg.steps or 25,
               height=cfg.h if cfg.h else None,
               width=cfg.w if cfg.w else None,
               guidance_scale=cfg.gs or 7.5,
               generator=gen)
    return out.frames[0], (cfg.fps or 8)


def run_hunyuanvideo(model_id, device, prompt, cfg, gen=None):
    # diffusers.HunyuanVideoPipeline official usage
    from diffusers import HunyuanVideoPipeline
    pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=_bf16())
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    out = pipe(prompt=prompt,
               height=cfg.h or 720, width=cfg.w or 1280,
               num_frames=cfg.f or 61,              # 4*k+1
               num_inference_steps=cfg.steps or 30,
               guidance_scale=cfg.gs or 6.0,
               generator=gen)
    return out.frames[0], (cfg.fps or 15)


def run_latte(model_id, device, prompt, cfg, gen=None):
    # diffusers.LattePipeline official usage
    from diffusers import LattePipeline
    pipe = LattePipeline.from_pretrained(model_id, torch_dtype=_bf16())
    pipe = pipe.to(device)
    out = pipe(prompt=prompt,
               height=cfg.h or 480, width=cfg.w or 848,
               num_frames=cfg.f or 97,
               num_inference_steps=cfg.steps or 30,
               guidance_scale=cfg.gs or 6.0,
               generator=gen)
    return out.frames[0], (cfg.fps or 24)


def run_animatelcm(adapter_repo, base_repo, device, prompt, cfg, gen=None):
    """
    AnimateLCM = MotionAdapter + AnimateDiffPipeline + LCMScheduler
    adapter_repo: e.g., "wangfuyun/AnimateLCM"
    base_repo:    e.g., "emilianJR/epiCRealism" (pass via --animatelcm-base)
    """
    from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
    adapter = MotionAdapter.from_pretrained(adapter_repo, torch_dtype=_fp16())
    pipe = AnimateDiffPipeline.from_pretrained(base_repo, motion_adapter=adapter, torch_dtype=_fp16())
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    pipe = pipe.to(device)
    out = pipe(prompt=prompt,
               height=cfg.h or 512, width=cfg.w or 512,
               num_frames=cfg.f or 64,
               guidance_scale=cfg.gs if cfg.gs is not None else 1.0,  # LCM通常低/无CFG
               num_inference_steps=cfg.steps or 6,
               generator=gen)
    # AnimateDiff 常见输出为帧序列；导出 MP4
    return out.frames[0], (cfg.fps or 24)


# ------------------------- Main -------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    prompts = read_prompts(args.prompts_file)
    total = len(prompts)
    start = args.batch_id * args.batch_size
    end = min(start + args.batch_size, total)
    if start >= total:
        print(f"Nothing to do: start={start} >= total={total}")
        sys.exit(0)

    bucket = gcs_bucket(args.gcs_bucket, args.gcs_key)
    os.makedirs(args.local_dir, exist_ok=True)

    cli = Cfg(args.height, args.width, args.num_frames, args.num_steps, args.guidance_scale, args.fps)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"Prompts slice: [{start}:{end}) / {total}")
    print("Models:", models)

    for model_id in models:
        key = sanitize_model_id(model_id)
        base = defaults_for(model_id)
        cfg = merge_cfg(base, cli)
        mid = model_id.lower()

        print(f"\n==== Model: {model_id} ====")

        for idx in range(start, end):
            prompt = prompts[idx]
            prompt_id = str(idx)
            gcs_path = f"{args.gcs_prefix}/{key}/{prompt_id}.mp4"

            if args.skip_existing and gcs_exists(bucket, gcs_path):
                print(f"[SKIP] exists gs://{args.gcs_bucket}/{gcs_path}")
                continue

            local_dir = os.path.join(args.local_dir, key)
            os.makedirs(local_dir, exist_ok=True)
            local_mp4 = os.path.join(local_dir, f"{prompt_id}.mp4")
            if os.path.exists(local_mp4):
                try:
                    os.remove(local_mp4)
                except Exception:
                    pass

            gen = None
            if args.seed is not None:
                gen = torch.Generator(device=args.device).manual_seed((args.seed + idx) & 0xFFFFFFFF)

            try:
                # Strict dispatch per family
                if "wan" in mid:
                    frames, fps = run_wan_t2v(model_id, args.device, prompt, cfg, gen)
                elif "skyreels-v2" in mid:
                    frames, fps = run_skyreels(model_id, args.device, prompt, cfg, gen)
                elif mid.startswith("lightricks/ltx-video-0.9.7-distilled"):
                    frames, fps = run_ltx_097_chain(model_id, args.device, prompt, cfg, gen, distilled=True)
                elif mid.startswith("lightricks/ltx-video-0.9.7-dev"):
                    frames, fps = run_ltx_097_chain(model_id, args.device, prompt, cfg, gen, distilled=False)
                elif mid.startswith("lightricks/ltx-video"):
                    frames, fps = run_ltx_base(model_id, args.device, prompt, cfg, gen)
                elif "cogvideox" in mid:
                    frames, fps = run_cogvideox(model_id, args.device, prompt, cfg, gen)
                elif "mochi" in mid:
                    frames, fps = run_mochi(model_id, args.device, prompt, cfg, gen)
                elif any(x in mid for x in ["zeroscope", "text-to-video-ms", "text_to_video"]):
                    frames, fps = run_modelscope_or_zeroscope(model_id, args.device, prompt, cfg, gen)
                elif "hunyuanvideo" in mid:
                    frames, fps = run_hunyuanvideo(model_id, args.device, prompt, cfg, gen)
                elif "latte" in mid:
                    frames, fps = run_latte(model_id, args.device, prompt, cfg, gen)
                elif "animatelcm" in mid:
                    base_ckpt = args.animatelcm_base
                    if not base_ckpt:
                        raise RuntimeError("AnimateLCM requires --animatelcm-base=<base SD repo>, e.g., emilianJR/epiCRealism")
                    frames, fps = run_animatelcm(model_id, base_ckpt, args.device, prompt, cfg, gen)
                else:
                    from diffusers import DiffusionPipeline
                    pipe = DiffusionPipeline.from_pretrained(model_id)
                    if hasattr(pipe, "to"):
                        pipe.to(args.device)
                    out = pipe(prompt=prompt, num_frames=cfg.f or 32, generator=gen)
                    frames, fps = (out.frames[0] if hasattr(out, "frames") else out[0]), (cfg.fps or 12)

                export_to_video(frames, local_mp4, fps=fps)
                gcs_upload(bucket, local_mp4, gcs_path)
                print(f"[OK] {key}:{prompt_id} -> gs://{args.gcs_bucket}/{gcs_path}")

            except Exception as e:
                print(f"[ERR] {key}:{prompt_id}: {e}", file=sys.stderr)
            finally:
                if os.path.exists(local_mp4):
                    try:
                        os.remove(local_mp4)
                    except Exception:
                        pass

    print("Done.")
