import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from mmengine import Config

from mmgpt import create_model_and_transforms
from mmgpt.models.builder import create_toy_model_and_transforms


@torch.no_grad()
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch inference with mmGPT / OpenFlamingo")
    parser.add_argument("--input-json", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output-json", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--vision-encoder-path", default="ViT-L-14", type=str)
    parser.add_argument("--vision-encoder-pretrained", default="openai", type=str)
    parser.add_argument("--lm-path", default="checkpoints/llama-7b_hf", type=str)
    parser.add_argument("--tokenizer-path", default="checkpoints/llama-7b_hf", type=str)
    parser.add_argument("--pretrained-path", default="checkpoints/OpenFlamingo-9B/checkpoint.pt", type=str)
    parser.add_argument("--tuning-config", required=True, help="path to tuning config file (.py)")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # 1️⃣ 加载模型
    print(f"🔹 Loading model from {args.pretrained_path}")
    tuning_config = Config.fromfile(args.tuning_config)

    model, image_processor, tokenizer = create_model_and_transforms(
        model_name="open_flamingo",
        clip_vision_encoder_path=args.vision_encoder_path,
        clip_vision_encoder_pretrained=args.vision_encoder_pretrained,
        lang_encoder_path=args.lm_path,
        tokenizer_path=args.tokenizer_path if args.tokenizer_path else args.lm_path,
        pretrained_model_path=args.pretrained_path,
        tuning_config=tuning_config.tuning_config,
    )

    model = model.to(args.device)
    model.eval()
    print("✅ Model initialized.\n")

    # 2️⃣ 加载输入 JSON
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    # 3️⃣ 遍历并推理
    for item in tqdm(data, desc="推理中"):
        _id = item.get("id")
        task = item.get("task", "")
        image_path = item.get("image")
        prompt = item.get("prompt", "Describe the image.")

        if not os.path.exists(image_path):
            print(f"⚠️ 图片不存在: {image_path}")
            results.append({"id": _id, "task": task, "image": image_path, "output": "ERROR: image not found"})
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ 无法打开图片 {image_path}: {e}")
            results.append({"id": _id, "task": task, "image": image_path, "output": "ERROR: failed to open image"})
            continue

        # 处理图像
        image_tensor = image_processor(image).unsqueeze(0).to(args.device)

        # 构建输入 prompt
        text_input = [prompt]
        inputs = tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(args.device)

        # 推理
        try:
            output = model.generate(
                vision_x=image_tensor.unsqueeze(1).unsqueeze(1),
                lang_x=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
            )

            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            result_text = decoded.strip()
        except Exception as e:
            result_text = f"ERROR: inference failed ({e})"

        results.append({
            "id": _id,
            "task": task,
            "image": image_path,
            "output": result_text
        })

    # 4️⃣ 保存结果
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 推理完成，共 {len(results)} 条样本，结果已保存至: {args.output_json}")


if __name__ == "__main__":
    main()