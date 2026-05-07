import json
import os
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

# ========================
# 配置部分
# ========================
MODEL_NAME = "meta-llama/LLaMA-3.2-VL"  # 替换为实际 LLaMA3.2-VL 模型路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_JSON = "input.json"  # [{"id":1,"task":"Describe this image","image":"path/to/img.jpg"}, ...]
OUTPUT_JSON = "output.json"

# ========================
# 模型加载
# ========================
print("加载模型和处理器...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto",
                                             torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32)
model.eval()

# ========================
# 读取输入 JSON
# ========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

# ========================
# 遍历每条数据
# ========================
for item in tqdm(data, desc="Running inference"):
    item_id = item.get("id")
    task = item.get("task", "")
    image_path = item.get("image", "")
    output_text = ""

    try:
        if not os.path.exists(image_path):
            output_text = f"Error: image not found at {image_path}"
        else:
            # 打开图片
            image = Image.open(image_path).convert("RGB")
            max_edge = max(image.size)
            image = image.resize((max_edge, max_edge))

            # 准备输入
            inputs = processor(
                text=task,
                images=image,
                return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 生成输出
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=256)

            # 解码
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    except Exception as e:
        output_text = f"Error: {e}"

    results.append({
        "id": item_id,
        "task": task,
        "image": image_path,
        "output": output_text
    })

# ========================
# 保存结果
# ========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 推理完成，结果已保存到 {OUTPUT_JSON}")
