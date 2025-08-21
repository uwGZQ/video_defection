#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple, resume-safe batch Text-to-Video runner for Hugging Face Diffusers models.
- YOU specify exact model ids via --models (comma-separated)
- Slices prompts by --batch-id & --batch-size (e.g., id=1,size=1000 => [1000,2000))
- Skips if target GCS object already exists (resume-friendly)
- Minimal per-family adapters so many variants "just work" (Wan 2.1/2.2 T2V sizes, SkyReels-V2 T2V/DF,
  CogVideoX 2B/5B, HunyuanVideo, LTX-Video, ModelScope/ZeroScope, Mochi-1 preview, etc.)

Example:
python demo.py \
  --prompts-file prompts.txt \
  --batch-id 1 --batch-size 1000 \
  --models Wan-AI/Wan2.1-T2V-1.3B-Diffusers,THUDM/CogVideoX-5b \
  --gcs-bucket oe-training-jieyu \
  --gcs-prefix ziqigao/video_defection \
  --height 480 --width 768 --num-frames 81 --fps 16 --seed 42

Dependencies:
  pip install -U diffusers transformers accelerate torch torchvision ftfy
  pip install -U google-cloud-storage imageio[ffmpeg]
  # auth:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import List, Optional

import torch
from google.cloud import storage

# diffusers imports (families)
from diffusers import (
    DiffusionPipeline,
    HunyuanVideoPipeline,
    LTXPipeline,
    CogVideoXPipeline,
)
# optional families (import guarded inside builders if not installed in this version)
from diffusers.utils import export_to_video

# Some families live behind optional classes; we import inside try/except in builders

# --------------------- CLI ---------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts-file", required=True, help="TXT file, one prompt per line")
    p.add_argument("--batch-id", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--models", required=True, help="Comma-separated HF model ids (e.g. Wan-AI/Wan2.1-T2V-1.3B-Diffusers)")
    p.add_argument("--gcs-bucket", required=True)
    p.add_argument("--gcs-prefix", required=True)

    # optional unified overrides
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--local-dir", default="./_t2v_tmp")
    p.add_argument("--gcs-key", default=None, help="Optional path to service-account JSON; else env var is used")
    return p.parse_args()


# ------------------ helpers -------------------

def read_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    return [x for x in lines if x.strip()]


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_").replace(" ", "_")


def gcs_client(bucket: str, gcs_key: Optional[str]):
    if gcs_key:
        client = storage.Client.from_service_account_json(gcs_key)
    else:
        client = storage.Client()
    return client.bucket(bucket)


def blob_exists(bucket: storage.Bucket, path: str) -> bool:
    return bucket.blob(path).exists()


def upload_file(bucket: storage.Bucket, src: str, dst_path: str):
    blob = bucket.blob(dst_path)
    blob.chunk_size = 16 * 1024 * 1024
    blob.upload_from_filename(src)


# --------------- family defaults ---------------
class Cfg:
    def __init__(self, h=None, w=None, f=None, steps=None, gs=None, fps=None):
        self.h, self.w, self.f, self.steps, self.gs, self.fps = h, w, f, steps, gs, fps

def defaults_for(model_id: str) -> Cfg:
    mid = model_id.lower()
    if "cogvideox" in mid:
        return Cfg(480, 768, 81, 50, 6.0, 16)
    if "hunyuanvideo" in mid:
        return Cfg(720, 1280, 61, 30, 6.0, 15)
    if "ltx" in mid:
        return Cfg(512, 768, 161, 50, 5.0, 24)
    if "wan" in mid:
        # 1.3B默认480p; 14B可配720p
        return Cfg(480, 848, 81, 30, 5.0, 16)
    if "skyreels" in mid:
        return Cfg(544, 960, 97, 30, 6.0, 24)
    if "zeroscope" in mid or "text-to-video-ms" in mid or "text_to_video" in mid:
        return Cfg(None, None, 16, 25, 7.5, 8)
    if "mochi" in mid:
        return Cfg(480, 848, 84, 64, None, 30)
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


# ----------------- builders -------------------

def build_pipeline(model_id: str, device: str):
    mid = model_id.lower()
    bf16 = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    fp16 = torch.float16 if torch.cuda.is_available() else torch.float32

    # Wan families (T2V variants only). Skip I2V/TI2V/VACE
    if "wan" in mid:
        if any(x in mid for x in ["-i2v-", "-ti2v-", "-vace-", "-flf2v-"]):
            raise RuntimeError("Wan model is not T2V (I2V/TI2V/VACE/FLF2V detected). Use a *-T2V-* repo.")
        from diffusers import WanPipeline
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=bf16)
        pipe.to(device)
        return pipe

    # SkyReels-V2 (T2V and DF are both text-to-video capable)
    if "skyreels" in mid:
        from diffusers import AutoencoderKLWan
        if "-df-" in mid:
            from diffusers import SkyReelsV2DiffusionForcingPipeline as _P
        else:
            from diffusers import SkyReelsV2Pipeline as _P
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe = _P.from_pretrained(model_id, vae=vae, torch_dtype=bf16)
        pipe.to(device)
        return pipe

    # CogVideoX
    if "cogvideox" in mid:
        pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=bf16)
        pipe.to(device)
        return pipe

    # HunyuanVideo
    if "hunyuanvideo" in mid:
        pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=bf16)
        pipe.enable_model_cpu_offload()
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        return pipe

    # LTX-Video
    if "ltx" in mid:
        pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=bf16)
        pipe.to(device)
        return pipe

    # ModelScope / ZeroScope and other generic T2V repos
    if any(x in mid for x in ["zeroscope", "text-to-video-ms", "text_to_video"]):
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=fp16, variant="fp16" if "text-to-video-ms" in mid else None)
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        return pipe

    # Mochi preview (if installed)
    if "mochi" in mid:
        try:
            from diffusers import MochiPipeline
            pipe = MochiPipeline.from_pretrained(model_id, variant="bf16", torch_dtype=bf16)
        except Exception:
            from diffusers import MochiPipeline
            pipe = MochiPipeline.from_pretrained(model_id)
        pipe.enable_model_cpu_offload()
        return pipe

    # Fallback
    pipe = DiffusionPipeline.from_pretrained(model_id)
    if hasattr(pipe, "to"):
        pipe.to(device)
    return pipe


# -------------- main run loop ---------------
if __name__ == "__main__":
    args = parse_args()

    # seed base
    if args.seed is not None:
        torch.manual_seed(args.seed)

    prompts = read_prompts(args.prompts_file)
    total = len(prompts)
    start = args.batch_id * args.batch_size
    end = min(start + args.batch_size, total)
    if start >= total:
        print(f"Nothing to do: start={start} >= total={total}")
        sys.exit(0)

    bucket = gcs_client(args.gcs_bucket, args.gcs_key)

    # CLI overrides cfg
    cli = Cfg(args.height, args.width, args.num_frames, args.num_steps, args.guidance_scale, args.fps)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"Prompts slice: [{start}:{end}) of {total}")
    print(f"Models: {models}")

    # Iterate models one by one (simpler, stable VRAM)
    for model_id in models:
        key = sanitize_model_id(model_id)
        base = defaults_for(model_id)
        cfg = merge_cfg(base, cli)

        print(f"\n==== Model: {model_id} ====")
        try:
            pipe = build_pipeline(model_id, args.device)
        except Exception as e:
            print(f"[LOAD-FAIL] {model_id}: {e}")
            continue

        for idx in range(start, end):
            prompt = prompts[idx]
            prompt_id = str(idx)  # use global line index as id
            gcs_path = f"{args.gcs_prefix}/{key}/{prompt_id}.mp4"

            # resume: skip existing
            if args.skip_existing and blob_exists(bucket, gcs_path):
                print(f"[SKIP] exists gs://{args.gcs_bucket}/{gcs_path}")
                continue

            local_dir = os.path.join(args.local_dir, key)
            os.makedirs(local_dir, exist_ok=True)
            local_mp4 = os.path.join(local_dir, f"{prompt_id}.mp4")
            if os.path.exists(local_mp4):
                os.remove(local_mp4)

            # Per-sample seed for reproducibility
            generator = None
            if args.seed is not None:
                generator = torch.Generator(device=args.device).manual_seed((args.seed + idx) & 0xFFFFFFFF)

            # Build call kwargs minimally
            kwargs = dict(prompt=prompt)
            if cfg.f is not None:
                kwargs["num_frames"] = cfg.f
            if cfg.steps is not None:
                kwargs["num_inference_steps"] = cfg.steps
            if cfg.gs is not None:
                kwargs["guidance_scale"] = cfg.gs
            if cfg.h is not None:
                kwargs["height"] = cfg.h
            if cfg.w is not None:
                kwargs["width"] = cfg.w
            if generator is not None:
                kwargs["generator"] = generator

            # LTX small extras
            if isinstance(pipe, LTXPipeline):
                kwargs.setdefault("decode_timestep", 0.03)
                kwargs.setdefault("decode_noise_scale", 0.025)

            try:
                out = pipe(**kwargs)
                frames = out.frames[0] if hasattr(out, "frames") else out[0]
                fps = cfg.fps or 16
                export_to_video(frames, local_mp4, fps=fps)
                upload_file(bucket, local_mp4, gcs_path)
                print(f"[OK] {key}:{prompt_id} -> gs://{args.gcs_bucket}/{gcs_path}")
            except Exception as e:
                print(f"[ERR] {key}:{prompt_id}: {e}")
            finally:
                if os.path.exists(local_mp4):
                    try:
                        os.remove(local_mp4)
                    except Exception:
                        pass

        # free
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Done.")
