import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch inference with Qwen-VL")
    parser.add_argument("--input-json", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output-json", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--device", default="cuda", help="推理设备: cuda 或 cpu")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"], help="推理精度")
    args = parser.parse_args()

    # =============== 加载模型 ===============
    print("🔹 Loading Qwen-VL model...")
    torch.manual_seed(1234)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16 if args.precision == "bf16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-VL",
        device_map=args.device,
        trust_remote_code=True,
        torch_dtype=dtype
    ).eval()
    print("✅ Model loaded.\n")

    # =============== 读取输入 JSON ===============
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📄 Loaded {len(data)} samples.")

    results = []

    # =============== 批量推理 ===============
    for item in tqdm(data, desc="推理中"):
        _id = item.get("id")
        task = item.get("task", "")
        image_path = item.get("image")
        prompt = item.get("prompt", "Describe the image.")

        # 检查图片路径
        if not os.path.exists(image_path):
            print(f"⚠️ 图片不存在: {image_path}")
            results.append({"id": _id, "task": task, "image": image_path, "output": "ERROR: image not found"})
            continue

        # 构建 Qwen-VL 输入格式
        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': prompt}
        ])

        try:
            inputs = tokenizer(query, return_tensors='pt')
            inputs = inputs.to(model.device)

            pred = model.generate(**inputs)
            response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
            response_clean = response.split("<|endoftext|>")[0].strip()
        except Exception as e:
            response_clean = f"ERROR: inference failed ({e})"

        results.append({
            "id": _id,
            "task": task,
            "image": image_path,
            "output": response_clean
        })

    # =============== 保存结果 ===============
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 推理完成，共 {len(results)} 条样本。结果已保存至: {args.output_json}")

if __name__ == "__main__":
    main()