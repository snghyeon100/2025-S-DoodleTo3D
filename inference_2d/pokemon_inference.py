# --- inference_2d/pokemon_inference.py ---
import os
import sys
from typing import Optional
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODAL_LORA_PATH = "/vol/models/model/pokemon"
LOCAL_LORA_PATH = "/home/aikusrv04/Doodle/FINAL/model/pokemon"

LORA_WEIGHTS_PATH = os.getenv("POKEMON_LORA_PATH", None)

if LORA_WEIGHTS_PATH is None:
    if os.path.exists(MODAL_LORA_PATH):
        LORA_WEIGHTS_PATH = MODAL_LORA_PATH
    else:
        LORA_WEIGHTS_PATH = LOCAL_LORA_PATH

SEED = 456
PROMPT_PREFIX = "pokemon style, realistic"
NEGATIVE_PROMPT = "background, blurry, low quality, distorted, ugly, bad anatomy, extra limbs"

# I/O 경로 (CLI에서 사용)
INPUT_DIR  = "/home/aikusrv04/Doodle/FINAL/inputs"
OUTPUT_DIR = "/home/aikusrv04/Doodle/FINAL/outputs_2d/pokemon"


def load_pokemon_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_scribble",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )

    if os.path.exists(LORA_WEIGHTS_PATH):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_WEIGHTS_PATH)
    else:
        print("[WARN] Pokemon LoRA not found. Using base model.")

    pipe = pipe.to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def run_pokemon_inference(pipe, image: Image.Image, user_prompt: str) -> Image.Image:
    prompt = f"{PROMPT_PREFIX}, cute {user_prompt} pokemon character, no background"
    image = image.convert("RGB").resize((512, 512))

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image,
        num_inference_steps=20,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        generator=torch.manual_seed(SEED)
    )

    return result.images[0]

def _find_input_path(num: str) -> Optional[str]:
    """inputs/input_{num}.(png|jpg|jpeg|webp) 중 존재하는 경로를 반환"""
    bases = [f"input_{num}"]
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    for base in bases:
        for ext in exts:
            p = os.path.join(INPUT_DIR, base + ext)
            if os.path.exists(p):
                return p
    return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pokemon_inference.py <NUM> <USER_PROMPT>")
        print("Example: python pokemon_inference.py 7 \"blue dragon\"")
        sys.exit(1)

    num = str(sys.argv[1])
    user_prompt = " ".join(sys.argv[2:])

    in_path = _find_input_path(num)
    if not in_path:
        print(f"[ERROR] Input image not found: {INPUT_DIR}/input_{num}.(png|jpg|jpeg|webp)")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"output_{num}.png")

    print(f"[INFO] Loading pipeline on {device} ...")
    pipe = load_pokemon_pipeline()

    print(f"[INFO] Reading: {in_path}")
    img = Image.open(in_path).convert("RGB")

    print(f"[INFO] Running inference (prompt: {user_prompt}) ...")
    out = run_pokemon_inference(pipe, img, user_prompt)

    print(f"[INFO] Saving to: {out_path}")
    out.save(out_path)

    print("[DONE] 2D generation finished.")