import os
import json
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch inference with Qwen2-VL")
    parser.add_argument("--input-json", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output-json", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-72B-Instruct", help="模型路径或名称")
    parser.add_argument("--device", default="cuda", help="推理设备: cuda / cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    # =============== 加载模型 ===============
    print(f"🔹 Loading model: {args.model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("✅ Model loaded.\n")

    # =============== 读取输入 JSON ===============
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📄 Loaded {len(data)} samples.\n")

    results = []

    # =============== 批量推理 ===============
    for item in tqdm(data, desc="推理中"):
        _id = item.get("id")
        task = item.get("task", "")
        image_path = item.get("image")
        prompt = item.get("prompt", "Describe the image.")

        # 检查图片
        if not os.path.exists(image_path):
            print(f"⚠️ 图像文件不存在: {image_path}")
            results.append({"id": _id, "task": task, "image": image_path, "output": "ERROR: image not found"})
            continue

        # 构造输入格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            # 处理输入
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(args.device)

            # 推理
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                generated_ids_trimmed = [
                    out[len(ipt):] for ipt, out in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()
        except Exception as e:
            output_text = f"ERROR: inference failed ({e})"

        results.append({
            "id": _id,
            "task": task,
            "image": image_path,
            "output": output_text
        })

    # =============== 保存结果 ===============
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 推理完成，共 {len(results)} 条样本。结果已保存至: {args.output_json}")

if __name__ == "__main__":
    main()