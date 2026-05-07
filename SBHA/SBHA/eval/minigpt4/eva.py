import os
import json
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import StoppingCriteriaList
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
from PIL import Image
from tqdm import tqdm


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MiniGPT-4 Batch Inference")
    parser.add_argument("--cfg-path", required=True, help="path to configuration yaml")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--input-json", required=True, help="path to input JSON file")
    parser.add_argument("--output-json", required=True, help="path to save output JSON file")
    args = parser.parse_args()

    conv_dict = {
        "pretrain_vicuna0": CONV_VISION_Vicuna0,
        "pretrain_llama2": CONV_VISION_LLama2,
    }

    print("🔹 Initializing MiniGPT-4 ...")
    cfg = Config(args)
    setup_seeds(cfg)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f"cuda:{args.gpu_id}")

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device=f"cuda:{args.gpu_id}") for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device=f"cuda:{args.gpu_id}", stopping_criteria=stopping_criteria)
    print("✅ Model initialized.\n")

    # 读取 JSON 文件
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 必须是数组，每条记录包含 id, task, image, prompt, ground")

    results = []

    for item in tqdm(data, desc="推理中"):
        _id = item.get("id")
        task = item.get("task")
        image_path = item.get("image")
        prompt = item.get("prompt", "")

        if not os.path.exists(image_path):
            print(f"❌ 图片不存在: {image_path}")
            results.append({
                "id": _id,
                "task": task,
                "image": image_path,
                "output": f"ERROR: image not found"
            })
            continue

        try:
            image = load_image(image_path)
        except Exception as e:
            print(f"⚠️ 载入图片出错 {image_path}: {e}")
            results.append({
                "id": _id,
                "task": task,
                "image": image_path,
                "output": f"ERROR: cannot open image"
            })
            continue

        # 初始化对话状态
        chat_state = CONV_VISION.copy()
        img_list = []
        chat.upload_img(image, chat_state, img_list)
        chat.encode_img(img_list)

        # 问题
        user_message = prompt.strip()
        chat.ask(user_message, chat_state)

        # 生成回答
        try:
            llm_message = chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=1,
                temperature=1.0,
                max_new_tokens=300,
                max_length=2000,
            )[0]
        except Exception as e:
            llm_message = f"ERROR: model inference failed ({e})"

        results.append({
            "id": _id,
            "task": task,
            "image": image_path,
            "output": llm_message
        })

    # 保存输出
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 推理完成，共 {len(results)} 条样本，结果保存在：{args.output_json}")


if __name__ == "__main__":
    main()