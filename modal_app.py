# modal_app.py
import base64, io, re
from typing import Literal

import modal
from PIL import Image

app = modal.App("doodle-to-magic")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "diffusers", "transformers", "peft", "accelerate",
        "fastapi[standard]", "pillow",
    )
    .add_local_python_source("inference_2d")
    # .add_local_python_source("inference_3d")
)

MODELS_DIR = "/vol/models"
weights_vol = modal.Volume.from_name("sd-lora-weights", create_if_missing=True)

@app.cls(image=image, gpu="T4", volumes={MODELS_DIR: weights_vol}, timeout=3600)
class SDInference:
    _pokemon = None
    _amateur = None

    @modal.enter()
    def load_models(self):
        from inference_2d.pokemon_inference import load_pokemon_pipeline
        from inference_2d.amateur_inference import load_amateur_pipeline
        self._pokemon = load_pokemon_pipeline()
        assert self._pokemon is not None, "Pokemon pipeline failed to load"
        self._amateur = load_amateur_pipeline()
        assert self._amateur is not None, "Amateur pipeline failed to load"

    @staticmethod
    def _decode_image(data_url_or_b64: str) -> Image.Image:
        m = re.match(r"^data:image/\w+;base64,(.*)$", data_url_or_b64)
        b64 = m.group(1) if m else data_url_or_b64
        img_bytes = base64.b64decode(b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    @modal.method()
    def infer(self, mode: Literal["pokemon", "amateur"], prompt: str, data_url_or_b64: str) -> str:
        image = self._decode_image(data_url_or_b64)

        if mode == "pokemon":
            from inference_2d.pokemon_inference import run_pokemon_inference
            assert self._pokemon is not None, "Pokemon pipeline not initialized"
            out_img = run_pokemon_inference(self._pokemon, image, prompt)
        else:
            from inference_2d.amateur_inference import run_amateur_inference
            assert self._amateur is not None, "Amateur pipeline not initialized"
            out_img = run_amateur_inference(self._amateur, image, prompt)

        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    
    # @modal.method()
    # def export_obj(self, mode: Literal["pokemon", "amateur"], data_url_or_b64: str) -> str:
    #     """
    #     프런트에서 보낸 결과 이미지(dataURL/base64)를 받아 OBJ(base64)로 반환.
    #     현재는 pokemon만 지원.
    #     """
    #     if mode != "pokemon":
    #         raise ValueError("OBJ export currently supported only for 'pokemon' mode.")

    #     image = self._decode_image(data_url_or_b64)

    #     # 3D 복원
    #     from inference_3d.pokemon_inference import run_pokemon_to_obj
    #     obj_bytes = run_pokemon_to_obj(image)

    #     # base64로 인코딩해 반환
    #     obj_b64 = base64.b64encode(obj_bytes).decode("utf-8")
    #     return obj_b64


# -------- FastAPI endpoints --------
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate(body: dict):
    """
    요청:
    { "image": "<dataURL or base64>", "prompt": "...", "theme": "pokemon|amateur" }

    응답:
    { "resultImage": "data:image/png;base64,..." }
    """
    theme = body.get("theme", "pokemon")
    prompt = body.get("prompt", "")
    img    = body.get("image", "")

    inferencer = SDInference()
    result_b64 = inferencer.infer.remote(
        "pokemon" if theme == "pokemon" else "amateur",
        prompt,
        img,
    )
    return {"resultImage": f"data:image/png;base64,{result_b64}"}

# @app.function(image=image)
# @modal.fastapi_endpoint(method="POST")
# def export_obj(body: dict):
#     """
#     요청:
#     { "image": "<dataURL or base64>", "mode": "pokemon|amateur" }

#     응답:
#     { "obj": "data:model/obj;base64,...." }
#     """
#     mode = body.get("mode", "pokemon")
#     img  = body.get("image", "")

#     inferencer = SDInference()
#     obj_b64 = inferencer.export_obj.remote(mode, img)

#     return {"obj": f"data:model/obj;base64,{obj_b64}"}


# -------- Local entrypoints (for development) --------
@app.local_entrypoint()
def test_2d():
    """
    2D 테스트: modal run modal_app.py::test_2d
    """
    import os
    sample_path = "sample_scribble.png"
    if not os.path.exists(sample_path):
        print("[INFO] Put a small PNG as 'sample_scribble.png' to test 2D.")
        return

    with open(sample_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    inferencer = SDInference()
    out = inferencer.infer.remote("pokemon", "monster", f"data:image/png;base64,{b64}")
    print("2D result b64 length:", len(out))


# @app.local_entrypoint()
# def test_3d():
#     """
#     3D 테스트: modal run modal_app.py::test_3d
#     - sample_scribble.png를 입력으로 사용
#     - OBJ base64 길이 출력
#     """
#     import os
#     sample_path = "sample_scribble.png"
#     if not os.path.exists(sample_path):
#         print("[INFO] Put a small PNG as 'sample_scribble.png' to test 3D.")
#         return

#     with open(sample_path, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode("utf-8")
#         data_url = f"data:image/png;base64,{b64}"

#     inferencer = SDInference()
#     obj_b64 = inferencer.export_obj.remote("pokemon", data_url)
#     print("3D OBJ b64 length:", len(obj_b64))
