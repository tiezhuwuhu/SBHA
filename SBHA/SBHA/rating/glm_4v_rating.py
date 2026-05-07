import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

device = "cuda"

# 模型加载
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/glm-4v-9b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

# 加载 JSON
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

scores_only = []

for item in data:
    image_path = item["imagepath"]
    task = item["task"]
    output = item["output"]

    # 打开图像
    image = Image.open(image_path).convert('RGB')

    # 构建评分提示
    query = f"Task: {task}\nModel output: {output}\n请对这个输出的准确性和与图像相关性打分（0-100分），只输出数字。"

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": query}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)

    # 推理生成评分
    with torch.no_grad():
        gen_kwargs = {"max_length": 32, "do_sample": False}  # 分数很短，不采样
        outputs_ids = model.generate(**inputs, **gen_kwargs)
        outputs_ids = outputs_ids[:, inputs['input_ids'].shape[1]:]
        score_text = tokenizer.decode(outputs_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # 提取数字评分
    match = re.search(r"\d{1,3}", score_text)
    score = int(match.group(0)) if match else 0
    score = max(0, min(score, 100))

    # 保存到数据结构
    item["score"] = score
    scores_only.append({"id": item["id"], "score": score})

# 保存 JSON 文件
with open("scores_only.json", "w", encoding="utf-8") as f:
    json.dump(scores_only, f, ensure_ascii=False, indent=4)

with open("data_with_scores.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("打分完成，生成 scores_only.json & data_with_scores.json")
