#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSeek-VL2 JSON 批量推理脚本
直接读取 JSON 文件，每条记录包含 id, task, image, prompt
输出 JSON 文件，每条记录包含 id, task, image, output
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# ---------------------- 模型初始化 ----------------------
def load_model(model_path="deepseek-ai/deepseek-vl2-small"):
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(torch.bfloat16).cuda().eval()
    return processor, tokenizer, model


# ---------------------- 单条记录推理 ----------------------
def infer_record(model, processor, tokenizer, record: dict) -> str:
    """
    record 示例：
    {
        "id": "001",
        "task": "describe_image",
        "image": "images/example1.jpg",
        "prompt": "Describe this image."
    }
    """
    image_path = record.get("image")
    prompt = record.get("prompt", "")

    if not image_path or not os.path.exists(image_path):
        return f"ERROR: image not found -> {image_path}"

    # 构造 conversation 格式，DeepSeek-VL2 要求
    conversation = [
        {"role": "<|User|>", "content": "<image>\n" + prompt, "images": [image_path]},
        {"role": "<|Assistant|>", "content": ""}
    ]

    try:
        pil_images = load_pil_images(conversation)
        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device)

        with torch.no_grad():
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            inputs_embeds, past_key_values = model.incremental_prefilling(
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                chunk_size=512
            )

            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                past_key_values=past_key_values,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )

        answer = tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
            skip_special_tokens=True
        )
        return answer.strip()
    except Exception as e:
        return f"ERROR during inference: {e}"


# ---------------------- 批量处理 JSON ----------------------
def process_json(input_path: str, output_path: str, processor, tokenizer, model):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 必须是数组，每个元素包含 id, task, image, prompt")

    results = []
    for idx, record in enumerate(data):
        print(f"[{idx}] id={record.get('id')}, image={record.get('image')} -> infer ...")
        output_text = infer_record(model, processor, tokenizer, record)
        results.append({
            "id": record.get("id"),
            "task": record.get("task"),
            "image": record.get("image"),
            "output": output_text
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 推理完成，共 {len(results)} 条结果，已保存到 {output_path}")


# ---------------------- CLI ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek-VL2 JSON 批量推理")
    parser.add_argument("--input", "-i", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--model-path", default="deepseek-ai/deepseek-vl2-small", help="模型路径或名称")
    args = parser.parse_args()

    processor, tokenizer, model = load_model(args.model_path)
    process_json(args.input, args.output, processor, tokenizer, model)
