# inference_3d/pokemon_inference.py
import os
import sys
import numpy as np
import torch
from PIL import Image
import rembg

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

# ---- 공통 설정 ----
MODEL_REPO = "stabilityai/TripoSR"
CHUNK_SIZE = 8192
MC_RESOLUTION = 400
FOREGROUND_RATIO = 0.85
USE_BG_REMOVAL = True

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TSR_POKEMON = None


def load_pokemon_tsr():
    """TripoSR 모델을 로드(레플리카 캐시)."""
    global _TSR_POKEMON
    if _TSR_POKEMON is not None:
        return _TSR_POKEMON

    model = TSR.from_pretrained(
        MODEL_REPO,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(CHUNK_SIZE)
    model.to(_DEVICE)
    _TSR_POKEMON = model
    return _TSR_POKEMON


def _prep_image(pil_img: Image.Image) -> Image.Image:
    """입력 RGB PIL 이미지를 TSR 입력에 맞게 전처리(배경제거/전경 비율 조정)."""
    if not USE_BG_REMOVAL:
        return pil_img.convert("RGB")

    session = rembg.new_session()
    image = remove_background(pil_img.convert("RGB"), session)
    image = resize_foreground(image, FOREGROUND_RATIO)

    # 알파를 회색 배경(0.5)로 합성
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1.0 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image


def run_pokemon_to_obj(pil_img: Image.Image, out_path: str | None = None) -> bytes:
    """
    입력 PIL 이미지를 기반으로 TSR로 3D 복원 후 OBJ bytes 반환.
    out_path를 주면 파일도 함께 저장.
    """
    model = load_pokemon_tsr()
    image = _prep_image(pil_img)

    with torch.no_grad():
        scene_codes = model([image], device=_DEVICE)

    meshes = model.extract_mesh(scene_codes, True, resolution=MC_RESOLUTION)

    # ⬇️ OBJ 문자열을 직접 받는 방식(텍스트)
    obj_str = meshes[0].export(file_type="obj")  # returns str

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(obj_str)

    return obj_str.encode("utf-8")  # API에서는 bytes로 넘기기 좋음


# ----------------------------- CLI 실행 -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pokemon_inference.py <NUM>")
        sys.exit(1)

    num = str(sys.argv[1])

    # 2D 출력물을 입력으로 사용
    INPUT_IMAGE = f"output_{num}.png"
    INPUT_PATH = f"/home/aikusrv04/Doodle/FINAL/outputs_2d/pokemon/{INPUT_IMAGE}"
    OUTPUT_PATH = "/home/aikusrv04/Doodle/FINAL/outputs_3d/pokemon"
    base_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]
    out_obj_path = os.path.join(OUTPUT_PATH, f"{base_name}.obj")

    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input image not found: {INPUT_PATH}")
        sys.exit(1)

    print(f"[INFO] Device: {_DEVICE}")
    print(f"[INFO] Reading: {INPUT_PATH}")
    pil_img = Image.open(INPUT_PATH).convert("RGB")

    print("[INFO] Running TSR (TripoSR) ...")
    _ = run_pokemon_to_obj(pil_img, out_obj_path)

    print(f"[DONE] Export finished: {out_obj_path}")
