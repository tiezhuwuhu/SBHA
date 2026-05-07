import json
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import re

# --------- 模型加载 ---------
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen3-VL-8B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(model_name, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_name)

# --------- 加载 JSON 数据 ---------
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

scores_only = []

# --------- 循环处理每条数据 ---------
for item in data:
    image_path = item["imagepath"]
    task = item["task"]
    output = item["output"]

    # 构建评分消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Task: {task}\nModel output: {output}\nPlease score this output from 0 to 100 based on accuracy and relevance to the image."}
            ],
        }
    ]

    # 准备输入
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # 生成评分
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=32)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        score_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

    # 尝试从文本中提取数字评分
    match = re.search(r"\d{1,3}", score_text)
    score = int(match.group(0)) if match else 0
    score = max(0, min(score, 100))  # 限制在 0-100

    # 添加到原数据
    item["score"] = score
    scores_only.append({"id": item["id"], "score": score})

# --------- 保存两个 JSON 文件 ---------
with open("scores_only.json", "w", encoding="utf-8") as f:
    json.dump(scores_only, f, ensure_ascii=False, indent=4)

with open("data_with_scores.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("打分完成，生成 files: scores_only.json & data_with_scores.json")
