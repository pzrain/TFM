import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import os
import argparse

args = argparse.ArgumentParser()
args.add_argument('--seed', default=0, type=int)
args.add_argument('--prompt_file', default='prompts')
args.add_argument('--output_path', default='results')
args = args.parse_args()

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    redirect_common_files=False,
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", skip_download=True),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", skip_download=True),
    ],
)

lora_path = f'models/Wan-AI/tfm-lora.safetensors'
pipe.load_lora(module=pipe.dit, path=lora_path)
pipe.enable_vram_management()

seed=int(args.seed)
prompt_file = f'examples/wanvideo/model_inference/{args.prompt_file}.txt'
prompts = []
with open(prompt_file) as f:
    lines = f.readlines()
    for line in lines:
        prompts.append(line.strip())

os.makedirs(args.output_path, exist_ok=True)
for index, prompt in enumerate(prompts):
    print(index, prompt)
    video = pipe(
        prompt=prompt,
        negative_prompt="overbright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression artifact, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        seed=seed, tiled=True,
    )
    save_video(video, os.path.join(args.output_path, f"{str(index).zfill(3)}.mp4"), fps=15, quality=5)

