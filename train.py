# train.py - Medical VLM Fine-tuning Script
# 
# 🏥 医疗视觉大模型微调脚本
# 基于 Unsloth 和 Qwen2-VL
#
# 功能：
# 1. 加载 4-bit 量化的 Qwen2-VL 模型
# 2. 配置 LoRA 适配器
# 3. 加载并处理医疗数据集
# 4. 执行监督微调 (SFT)
# 5. 保存微调后的 LoRA 权重

import os
import torch
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TextStreamer

def main():
    print(" Starting Medical VLM Fine-tuning...")

    # =================================================================
    # 1. 配置与模型加载
    # =================================================================
    # 模型路径 (请修改为你的本地路径或 HuggingFace 模型 ID)
    # 注意：这里使用 4-bit 量化版本以节省显存
    MODEL_NAME = "/root/autodl-tmp/models/unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
    OUTPUT_DIR = "outputs"
    LORA_OUTPUT_DIR = "lora_model"

    print(f" Loading model from: {MODEL_NAME}")
    
    # 加载模型和分词器
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        use_gradient_checkpointing="unsloth",
        local_files_only=True,
    )

    # =================================================================
    # 2. 配置 LoRA 适配器
    # =================================================================
    print(" Configuring LoRA adapter...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # 不微调视觉层
        finetune_language_layers=True, # 重点微调语言层
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,           # LoRA rank
        lora_alpha=16,  # Alpha 参数
        lora_dropout=0,
        bias="none",
        use_rslora=False,
        loftq_config=None,
    )

    # =================================================================
    # 3. 数据集加载与处理
    # =================================================================
    print("Loading and processing dataset...")
    # 加载本地数据集
    # 假设 ./data 目录下有正确的 train 数据
    try:
        dataset = load_dataset("./data", split="train")
    except Exception as e:
        print(f" Error loading dataset: {e}")
        print("Please ensure your dataset is in the './data' directory.")
        return

    # 定义系统指令
    instruction = "你是一名专业的放射科医生，请准确描述你在图片看到的内容。"

    # 数据转换函数
    def convert_to_conversation(sample):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample['image']}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample['caption']}
                ]
            }
        ]
        return {"messages": conversation}

    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    print(f" Processed {len(converted_dataset)} samples.")

    # =================================================================
    # 4. 执行微调 (Training)
    # =================================================================
    print("Starting training...")
    
    # 切换到训练模式
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,  # 显存较小可设为 1
            gradient_accumulation_steps=4,
            max_steps=30,                   # 演示用步数，实际训练请调大 (e.g., 60-100)
            learning_rate=2e-4,
            warmup_steps=5,
            lr_scheduler_type="cosine",
            bf16=is_bf16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            seed=3407,
            logging_steps=1,
            output_dir=OUTPUT_DIR,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=2048,
        )
    )

    trainer_stats = trainer.train()
    print("✅ Training completed.")

    # =================================================================
    # 5. 保存模型
    # =================================================================
    print(f"💾 Saving LoRA model to '{LORA_OUTPUT_DIR}'...")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print("✅ Model saved successfully!")

    # =================================================================
    # 6. (可选) 简单的推理测试
    # =================================================================
    print("\n🔍 Running post-training inference test...")
    FastVisionModel.for_inference(model)
    
    image = dataset[0]['image']
    test_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image"}
            ]
        }
    ]
    
    input_text = tokenizer.apply_chat_template(test_messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )

if __name__ == "__main__":
    main()
