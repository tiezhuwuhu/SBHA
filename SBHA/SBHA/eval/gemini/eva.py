#import json
import os
from tqdm import tqdm
from google import genai
from google.genai import types

# 初始化 Gemini 客户端（推荐设置环境变量 GOOGLE_API_KEY）
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# 输入和输出文件路径
input_file = "input.json"   # 例如 [{"id":1,"task":"Describe the image","image":"path/to/img.jpg"}, ...]
output_file = "output.json"

# 读取输入 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

for item in tqdm(data, desc="Processing images"):
    item_id = item.get("id")
    task = item.get("task", "")
    image_path = item.get("image", "")

    # 默认输出
    output_text = ""

    try:
        # 检查文件存在
        if not os.path.exists(image_path):
            output_text = f"Error: image not found at {image_path}"
        else:
            # 读取图片字节
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # 调用 Gemini 生成内容
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    ),
                    task  # 将任务作为 prompt
                ]
            )
            output_text = response.text.strip()

    except Exception as e:
        output_text = f"Error: {e}"

    # 保存结果
    results.append({
        "id": item_id,
        "task": task,
        "image": image_path,
        "output": output_text
    })

# 写入输出文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 推理完成，结果已保存到 {output_file}")