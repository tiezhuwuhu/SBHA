import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor


def main():
    import argparse
    parser = argparse.ArgumentParser(description="mPLUG-Owl Batch Inference")
    parser.add_argument("--input-json", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output-json", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--model-path", default="MAGAer13/mplug-owl-llama-7b", help="Hugging Face 模型路径")
    parser.add_argument("--device", default="cuda", help="推理设备 (cuda / cpu)")
    args = parser.parse_args()

    # === 1. 加载模型 ===
    print(f"🔹 Loading model from {args.model_path} ...")
    model = MplugOwlForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    image_processor = MplugOwlImageProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    model.eval()

    # === 2. 读取输入 JSON ===
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 文件格式错误，应为包含多条样本的列表。")

    results = []

    # === 3. 遍历每个样本 ===
    for item in tqdm(data, desc="推理中"):
        sample_id = item.get("id")
        task = item.get("task", "")
        image_path = item.get("image")
        prompt = item.get("prompt", "")

        # 检查图像路径
        if not os.path.exists(image_path):
            print(f"⚠️ 图片不存在: {image_path}")
            results.append({
                "id": sample_id,
                "task": task,
                "image": image_path,
                "output": "ERROR: image not found"
            })
            continue

        # 读取图像
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ 无法读取图片 {image_path}: {e}")
            results.append({
                "id": sample_id,
                "task": task,
                "image": image_path,
                "output": "ERROR: failed to load image"
            })
            continue

        # 构造对话模板
        text_prompt = f"The following is a conversation between a curious human and AI assistant. " \
                      f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n" \
                      f"Human: <image>\nHuman: {prompt}\nAI: "

        # 处理输入
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
        inputs = {k: (v.bfloat16() if v.dtype == torch.float else v) for k, v in inputs.items()}
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    top_k=5,
                    max_length=512
                )
                sentence = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
            except Exception as e:
                sentence = f"ERROR: inference failed ({e})"

        # 保存结果
        results.append({
            "id": sample_id,
            "task": task,
            "image": image_path,
            "output": sentence.strip()
        })

    # === 4. 写出结果 ===
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 推理完成，共 {len(results)} 条样本，结果保存至：{args.output_json}")


if __name__ == "__main__":
    main()