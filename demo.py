import argparse
import os
import sys
import hashlib
from typing import List, Optional

import torch
from google.cloud import storage

from diffusers import (
    DiffusionPipeline,
    CogVideoXPipeline,
    HunyuanVideoPipeline,
    LTXPipeline,
    WanPipeline,
    SkyReelsV2Pipeline,
    SkyReelsV2DiffusionForcingPipeline,
    MochiPipeline,
)
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan


def parse_args():
    p = argparse.ArgumentParser(description="Run HF T2V models on a slice of prompts and upload to GCS")
    p.add_argument("--prompts-file", required=True, help="txt prompts")
    p.add_argument("--batch-id", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--models", required=True, help="(e.g. THUDM/CogVideoX-5b, Lightricks/LTX-Video, ...)")
    p.add_argument("--gcs-bucket", default="oe-training-jieyu")
    p.add_argument("--gcs-prefix", default="ziqigao/video_defection")

    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=None, help="num_inference_steps")
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--local-dir", default="./_t2v_tmp")
    return p.parse_args()


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_").replace(" ", "_")


def blob_exists(bucket: storage.Bucket, path: str) -> bool:
    blob = bucket.blob(path)
    return blob.exists()


def upload_file(bucket: storage.Bucket, src: str, dst_path: str):
    os.makedirs(os.path.dirname(src), exist_ok=True)
    blob = bucket.blob(dst_path)
    blob.upload_from_filename(src)


def read_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]



class T2VConfig:
    def __init__(self, height=None, width=None, num_frames=None, num_steps=None, guidance_scale=None, fps=None):
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.fps = fps


def default_cfg_for_model(model_id: str) -> T2VConfig:
    mid = model_id.lower()
    if "cogvideox" in mid:
        return T2VConfig(height=480, width=768, num_frames=81, num_steps=50, guidance_scale=6.0, fps=16)
    if "hunyuanvideo" in mid:
        return T2VConfig(height=720, width=1280, num_frames=61, num_steps=30, guidance_scale=6.0, fps=15)
    if "ltx" in mid:
        return T2VConfig(height=512, width=768, num_frames=161, num_steps=50, guidance_scale=5.0, fps=24)
    if "wan" in mid:
        return T2VConfig(height=480, width=848, num_frames=81, num_steps=30, guidance_scale=6.0, fps=16)
    if "skyreels" in mid:
        return T2VConfig(height=544, width=960, num_frames=97, num_steps=30, guidance_scale=6.0, fps=24)
    if ("zeroscope" in mid) or ("text-to-video-ms" in mid) or ("text_to_video" in mid):
        return T2VConfig(height=None, width=None, num_frames=16, num_steps=25, guidance_scale=7.5, fps=8)
    if "mochi" in mid:
        return T2VConfig(height=480, width=848, num_frames=84, num_steps=64, guidance_scale=None, fps=30)
    return T2VConfig(num_frames=32, num_steps=30, guidance_scale=6.0, fps=12)


def merge_cfg(base: T2VConfig, override: T2VConfig) -> T2VConfig:
    out = T2VConfig(
        height=override.height if override.height is not None else base.height,
        width=override.width if override.width is not None else base.width,
        num_frames=override.num_frames if override.num_frames is not None else base.num_frames,
        num_steps=override.num_steps if override.num_steps is not None else base.num_steps,
        guidance_scale=override.guidance_scale if override.guidance_scale is not None else base.guidance_scale,
        fps=override.fps if override.fps is not None else base.fps,
    )
    return out


def build_pipeline(model_id: str, device: str):
    mid = model_id.lower()
    dtype_bf16 = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dtype_fp16 = torch.float16 if torch.cuda.is_available() else torch.float32

    if "cogvideox" in mid:
        pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=dtype_bf16)
        pipe.to(device)
        return pipe

    if "hunyuanvideo" in mid:
        pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=dtype_bf16)
        pipe.enable_model_cpu_offload()
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        return pipe

    if "ltx" in mid:
        pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=dtype_bf16)
        pipe.to(device)
        return pipe

    if "wan" in mid:
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype_bf16)
        pipe.to(device)
        return pipe

    if "skyreels" in mid:
        if "-df-" in mid:
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype_bf16)
        else:
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            pipe = SkyReelsV2Pipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype_bf16)
        pipe.to(device)
        return pipe

    if ("zeroscope" in mid) or ("text-to-video-ms" in mid) or ("text_to_video" in mid):
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype_fp16, variant="fp16" if "text-to-video-ms" in mid else None)
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        return pipe

    if "mochi" in mid:
        try:
            pipe = MochiPipeline.from_pretrained(model_id, variant="bf16", torch_dtype=dtype_bf16)
        except Exception:
            pipe = MochiPipeline.from_pretrained(model_id)
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        return pipe

    pipe = DiffusionPipeline.from_pretrained(model_id)
    if hasattr(pipe, "to"):
        pipe.to(device)
    return pipe


def run_one(model_id: str, prompt: str, cfg: T2VConfig, seed: Optional[int] = None):
    pipe = build_pipeline(model_id, args.device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(seed)

    kwargs = dict(prompt=prompt)
    if cfg.num_frames is not None:
        kwargs["num_frames"] = cfg.num_frames
    if cfg.num_steps is not None:
        kwargs["num_inference_steps"] = cfg.num_steps
    if cfg.guidance_scale is not None:
        kwargs["guidance_scale"] = cfg.guidance_scale
    if cfg.height is not None:
        kwargs["height"] = cfg.height
    if cfg.width is not None:
        kwargs["width"] = cfg.width
    if generator is not None:
        kwargs["generator"] = generator

    if isinstance(pipe, LTXPipeline):
        kwargs.setdefault("decode_timestep", 0.03)
        kwargs.setdefault("decode_noise_scale", 0.025)

    out = pipe(**kwargs)
    frames = out.frames[0] if hasattr(out, "frames") else out[0]

    fps = cfg.fps or 16

    return frames, fps, pipe


if __name__ == "__main__":
    args = parse_args()

    prompts = read_prompts(args.prompts_file)
    n = len(prompts)
    start = args.batch_id * args.batch_size
    end = min(start + args.batch_size, n)
    if start >= n:
        print(f"Nothing to do: start={start} >= total={n}")
        sys.exit(0)

    os.makedirs(args.local_dir, exist_ok=True)
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(args.gcs_bucket)

    cli_cfg = T2VConfig(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
    )

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"Total prompts: {n}; processing slice [{start}:{end}) => {end-start} items")
    print(f"Models: {models}")

    for model_id in models:
        model_key = sanitize_model_id(model_id)
        print(f"\n==== Running model: {model_id} ====")
        base_cfg = default_cfg_for_model(model_id)
        cfg = merge_cfg(base_cfg, cli_cfg)

        pipe = build_pipeline(model_id, args.device)

        for global_idx in range(start, end):
            prompt = prompts[global_idx]
            prompt_id = str(global_idx) 
            gcs_path = f"{args.gcs_prefix}/{model_key}/{prompt_id}.mp4"

            if args.skip_existing and blob_exists(bucket, gcs_path):
                print(f"[SKIP] exists on GCS: {gcs_path}")
                continue

            local_dir = os.path.join(args.local_dir, model_key)
            os.makedirs(local_dir, exist_ok=True)
            local_mp4 = os.path.join(local_dir, f"{prompt_id}.mp4")

            seed = None
            if args.seed is not None:
                seed = (args.seed + global_idx) & 0xFFFFFFFF

            try:
                kwargs = dict(prompt=prompt)
                if cfg.num_frames is not None:
                    kwargs["num_frames"] = cfg.num_frames
                if cfg.num_steps is not None:
                    kwargs["num_inference_steps"] = cfg.num_steps
                if cfg.guidance_scale is not None:
                    kwargs["guidance_scale"] = cfg.guidance_scale
                if cfg.height is not None:
                    kwargs["height"] = cfg.height
                if cfg.width is not None:
                    kwargs["width"] = cfg.width
                if seed is not None:
                    kwargs["generator"] = torch.Generator(device=args.device).manual_seed(seed)
                if isinstance(pipe, LTXPipeline):
                    kwargs.setdefault("decode_timestep", 0.03)
                    kwargs.setdefault("decode_noise_scale", 0.025)

                out = pipe(**kwargs)
                frames = out.frames[0] if hasattr(out, "frames") else out[0]

                fps = cfg.fps or 16
                export_to_video(frames, local_mp4, fps=fps)

                upload_file(bucket, local_mp4, gcs_path)
                print(f"[OK] {model_key}:{prompt_id} -> gs://{args.gcs_bucket}/{gcs_path}")

            except Exception as e:
                print(f"[ERR] {model_key}:{prompt_id} :: {e}", file=sys.stderr)
                continue

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("All done.")
