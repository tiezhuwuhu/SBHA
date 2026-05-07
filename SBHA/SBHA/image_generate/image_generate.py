import os
import json
import torch
import streamlit as st
import numpy as np
from einops import repeat
from pytorch_lightning import seed_everything

from scripts.demo.streamlit_helpers import *

SAVE_PATH = "outputs/demo/txt2img/"

# SDXL base ratios
SD_XL_BASE_RATIOS = {
    "1.0": (1024, 1024),
    "0.94": (960, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
}

VERSION2SPECS = {
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
    },
    "SDXL-refiner-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_1.0.safetensors",
    },
}

# ===========================
# Helper function: load JSON
# ===========================
def load_prompt_from_json(json_file="input.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("prompt", "")

# ===========================
# Main Streamlit App
# ===========================
if __name__ == "__main__":
    st.title("Stable Diffusion JSON Prompt Generator")

    # ===========================
    # 读取 JSON prompt
    # ===========================
    prompt = load_prompt_from_json("input.json")
    st.write(f"**Using prompt from JSON:** {prompt}")

    # ===========================
    # 模型选择
    # ===========================
    version = st.selectbox("Model Version", list(VERSION2SPECS.keys()), 0)
    version_dict = VERSION2SPECS[version]
    is_legacy = version_dict["is_legacy"]

    # Low VRAM 模式
    set_lowvram_mode(st.checkbox("Low vram mode", True))

    # 初始化保存路径
    save_locally, save_path = init_save_locally(os.path.join(SAVE_PATH, version))

    # 初始化模型
    state = init_st(version_dict, load_filter=True)
    if state["msg"]:
        st.info(state["msg"])
    model = state["model"]

    # ===========================
    # 随机种子
    # ===========================
    seed = st.sidebar.number_input("Seed", value=42, min_value=0, max_value=int(1e9))
    seed_everything(seed)

    # ===========================
    # txt2img 生成
    # ===========================
    if version.startswith("SDXL-base"):
        W, H = SD_XL_BASE_RATIOS["1.0"]  # 默认 1024x1024
    else:
        W, H = version_dict["W"], version_dict["H"]

    C = version_dict["C"]
    F = version_dict["f"]

    init_dict = {"orig_width": W, "orig_height": H, "target_width": W, "target_height": H}
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        init_dict,
        prompt=prompt,
        negative_prompt=""
    )

    sampler, num_rows, num_cols = init_sampling()
    num_samples = num_rows * num_cols

    if st.button("Generate Image"):
        st.write(f"**Model:** {version}")
        out = do_sample(
            model,
            sampler,
            value_dict,
            num_samples,
            H,
            W,
            C,
            F,
            force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        )

        if isinstance(out, (tuple, list)):
            samples, _ = out
        else:
            samples = out


        # 保存图片
        if save_locally:
            perform_save_locally(save_path, samples)
            st.success(f"Saved {len(samples)} image(s) to {save_path}")
