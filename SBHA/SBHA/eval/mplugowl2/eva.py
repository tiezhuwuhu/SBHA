import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import TextStreamer
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="mPLUG-Owl2 Batch Inference")
    parser.add_argument("--input-json", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output-json", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--model-path", default="MAGAer13/mplug-owl2-llama2-7b", help="HuggingFace 模型路径")
    parser.add_argument("--device", default="cuda", help="推理设备 (cuda/cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # === 1. 加载模型 ===
    print(f"🔹 Loading model from {args.model_path} ...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        load_8bit=False,
        load_4bit=False,
        device=args.device
    )

    model.eval()
    print("✅ Model initialized.\n")

    # === 2. 读取输入 JSON ===
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 必须是一个列表，每个元素包含 id/task/image/prompt")

    results = []

    # === 3. 遍历样本并推理 ===
    for item in tqdm(data, desc="推理中"):
        _id = item.get("id")
        task = item.get("task", "")
        image_path = item.get("image")
        query = item.get("prompt", "")

        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"⚠️ 图片不存在: {image_path}")
            results.append({"id": _id, "task": task, "image": image_path, "output": "ERROR: image not found"})
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            max_edge = max(image.size)
            image = image.resize((max_edge, max_edge))
        except Exception as e:
            print(f"❌ 无法打开图片 {image_path}: {e}")
            results.append({"id": _id, "task": task, "image": image_path, "output": "ERROR: failed to open image"})
            continue

        # 构造对话模板
        conv = conv_templates["mplug_owl2"].copy()
        inp = DEFAULT_IMAGE_TOKEN + query
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 处理输入
        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

        stop_str = conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # === 4. 模型推理 ===
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            outputs = f"ERROR: inference failed ({e})"

        results.append({
            "id": _id,
            "task": task,
            "image": image_path,
            "output": outputs
        })

    # === 5. 保存结果 ===
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 推理完成，共 {len(results)} 条样本，结果已保存至: {args.output_json}")


if __name__ == "__main__":
    main()