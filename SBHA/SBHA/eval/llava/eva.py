
import os
import json
import torch
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

# --------------------- 模型路径与文件路径 ---------------------
model_path = "/home/liudongdong/LLaVA-main/llava-v1.5-13b/"
input_json = "/path/to/input.json"      # ✅ 修改为你的输入 JSON 路径
output_json = "/path/to/output.json"    # ✅ 修改为你的输出 JSON 路径

# --------------------- 加载模型 ---------------------
print("🔹 Loading LLaVA model...")
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name, load_8bit=False, load_4bit=True, device="cuda:0"
)
print("✅ Model loaded.\n")

# --------------------- 读取输入 JSON ---------------------
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, list):
    raise ValueError("输入 JSON 必须是一个数组，每个元素包含 id, task, image, prompt, ground 字段")

results = []

# --------------------- 遍历并推理 ---------------------
for idx, item in enumerate(data):
    _id = item.get("id")
    task = item.get("task")
    image_path = item.get("image")
    prompt = item.get("prompt", "")

    print(f"[{idx}] 推理中: id={_id}, image={image_path}")

    # 检查图片路径
    if not os.path.exists(image_path):
        print(f"❌ 找不到图片: {image_path}")
        results.append({
            "id": _id,
            "task": task,
            "image": image_path,
            "output": f"ERROR: image not found: {image_path}"
        })
        continue

    # 读取并处理图像
    image = Image.open(image_path).convert("RGB")
    max_edge = max(image.size)
    image = image.resize((max_edge, max_edge))
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # 构造 prompt
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{prompt}")
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    # 编码并生成
    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.2,
            max_new_tokens=512,
            use_cache=True,
        )

    output_text = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True).strip()

    results.append({
        "id": _id,
        "task": task,
        "image": image_path,
        "output": output_text
    })

# --------------------- 保存结果 ---------------------
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ 推理完成，共 {len(results)} 条结果，已保存到 {output_json}")
