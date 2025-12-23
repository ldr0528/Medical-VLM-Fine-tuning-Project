# train_grpo.py - Medical VLM Reinforcement Learning (GRPO) Script
# 
# 🏥 医疗视觉大模型强化学习脚本 (GRPO)
# 基于 Unsloth 和 Qwen3-VL
#
# 功能：
# 1. 加载 SFT 后的模型作为初始策略
# 2. 定义奖励函数 (Reward Functions)：
#    - XML 格式奖励：强制模型使用 <reasoning>...</reasoning> <answer>...</answer> 格式
#    - 长度奖励：鼓励更详细的推理过程
# 3. 执行 GRPO 训练：让模型学会“先思考，再回答”
# 4. 保存 RL 后的模型权重

import os
import re
import torch
from unsloth import FastVisionModel, is_bf16_supported
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    print("🚀 Starting Medical VLM GRPO Training...")

    # =================================================================
    # 1. 配置与模型加载
    # =================================================================
    # 直接加载 SFT 后的 LoRA 模型作为起点
    # 如果 lora_model 存在，直接加载它；否则加载基座
    if os.path.exists("lora_model"):
        print(f"Loading SFT model from: lora_model")
        MODEL_NAME = "lora_model" # Unsloth 支持直接加载 LoRA 目录
    else:
        MODEL_NAME = "/root/autodl-tmp/models/unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
        print(f"'lora_model' not found! Using base model: {MODEL_NAME}")

    OUTPUT_DIR = "outputs_grpo"

    # 加载模型
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        use_gradient_checkpointing="unsloth",
        local_files_only=True,
    )
    
    # 配置 LoRA (GRPO 也需要 LoRA 来节省显存)
    print(" Configuring LoRA for GRPO...")
    
    # 检查模型是否已经加载了 Adapter (从 lora_model 加载时会自动带上)
    # 如果已经有 adapter，我们只需要确保它处于训练模式
    if hasattr(model, "peft_config") and len(model.peft_config) > 0:
        print("✅ Model already has LoRA adapters. Enabling training mode...")
        FastVisionModel.for_training(model)
    else:
        # 只有当模型是纯基座时，才需要添加新的 LoRA
        print("🆕 Adding new LoRA adapters...")
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_rslora=False,
        )

    # =================================================================
    # 2. 准备数据集与 Prompt 格式
    # =================================================================
    print(" Loading dataset...")
    # 这里我们复用 Radiology-mini 数据集，但我们需要构造不带 Answer 的 Prompt
    # 让模型自己生成推理过程和答案，然后通过奖励函数来评估
    dataset = load_dataset("./data", split="train")

    # 定义系统提示词，强制要求特定的输出格式
    SYSTEM_PROMPT = """
    你是一名专业的放射科医生。请分析给定的医疗图像。
    请严格按照以下格式输出你的诊断结果，并且只输出这两个标签的内容：
    
    <reasoning>
    在这里写下你的观察过程、推理逻辑和分析细节。
    </reasoning>
    <answer>
    在这里给出最终的诊断结论。
    </answer>
    """

    # GRPO 需要的数据格式通常是 prompt 列
    def format_data(sample):
        # 构造输入 Prompt
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample['image']},
                    {"type": "text", "text": "请分析这张图片。"}
                ]
            }
        ]
        return {
            "prompt": messages,
            "ground_truth": sample['caption'] # 1) 改名 target -> ground_truth
        }

    # 1) 改名 target -> ground_truth, 并增加 num_proc=4 加速
    dataset = dataset.map(format_data, remove_columns=["image", "caption", "image_id", "cui"], num_proc=4)

    # =================================================================
    # 3. 定义奖励函数 (Reward Functions)
    # =================================================================
    print("⚖️ Defining Reward Functions...")

    # 1. 格式奖励：检查是否包含 XML 标签，且内容充实
    def xml_format_reward(completions, **kwargs):
        rewards = []
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>(.*?)</answer>"
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            match = re.search(pattern, text, re.DOTALL)
            
            if match:
                # 检查 <answer> 内容长度
                answer_content = match.group(1).strip()
                if len(answer_content) > 10: # 至少有 10 个字符
                    rewards.append(1.0)
                else:
                    rewards.append(0.5) # 格式对但内容太短
            else:
                rewards.append(0.0)
        return rewards

    # 2. 长度奖励 (Length Reward)：鼓励适中长度的推理
    # 分段/门槛式设计，严厉打击过短回复
    def length_reward(completions, **kwargs):
        rewards = []
        min_len = 80
        max_len = 250
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
            if reasoning_match:
                reasoning_text = reasoning_match.group(1)
                length = len(reasoning_text)
                
                # 分段奖励逻辑
                if length < min_len:
                    # 严厉惩罚过短回复 (如 19 tokens)
                    rewards.append(-0.5)
                elif min_len <= length <= max_len:
                    # 舒适区给正奖励
                    rewards.append(0.5)
                else: # length > max_len
                    # 超过上限给轻微负分，防止废话
                    rewards.append(-0.1)
            else:
                rewards.append(0.0)
        return rewards
    
    # 3. 步骤奖励 (Step Reward)：鼓励结构化推理 (新增)
    def step_reward(completions, **kwargs):
        rewards = []
        # 检测 "1.", "Step 1", "首先", "第一" 等步骤词
        step_patterns = [r"\d+\.", r"Step \d+", r"首先", r"其次", r"最后", r"第一", r"第二"]
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
            if reasoning_match:
                reasoning_text = reasoning_match.group(1)
                step_count = 0
                for p in step_patterns:
                    step_count += len(re.findall(p, reasoning_text))
                # 每个步骤加 0.1 分，上限 0.5 分
                rewards.append(min(step_count * 0.1, 0.5))
            else:
                rewards.append(0.0)
        return rewards

    # 4. 准确率奖励 (Accuracy)：主目标 (改进版 - 实体关键词覆盖率)
    # 1) 签名修改：target -> ground_truth, 兼容 **kwargs
    def accuracy_reward(completions, ground_truth, **kwargs):
        rewards = []
        for completion, ref_answer in zip(completions, ground_truth):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            # 尝试提取 <answer> 内容
            answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            
            # 提取预测文本：如果有标签取标签内，否则取最后一段，再否则取全文
            if answer_match:
                pred_answer = answer_match.group(1).lower().strip()
            elif "<answer>" in text:
                pred_answer = text.split("<answer>")[-1].lower().strip()
            else:
                pred_answer = text.lower().strip()
            
            # 预处理：移除标点符号，只保留字母数字和空格
            pred_clean = re.sub(r'[^\w\s]', ' ', pred_answer)
            ref_clean = re.sub(r'[^\w\s]', ' ', ref_answer.lower())
            
            # 分词并过滤停用词
            stop_words = {"the", "is", "a", "an", "of", "in", "on", "at", "and", "with", "to", "for", "it", "this", "that"}
            ref_tokens = set([w for w in ref_clean.split() if w not in stop_words and len(w) > 2])
            pred_tokens = set([w for w in pred_clean.split() if w not in stop_words and len(w) > 2])
            
            # 只要有任何重叠就给基础分，避免全0
            intersection = ref_tokens.intersection(pred_tokens)
            
            if not ref_tokens:
                 # 参考答案无效时，给一个中间分保底
                 rewards.append(0.5)
                 continue

            if not intersection:
                 rewards.append(0.0)
            else:
                 # 计算覆盖率
                 recall = len(intersection) / len(ref_tokens)
                 
                 # 阶梯奖励设计：更密集的阶梯，确保有分可得
                 if recall >= 0.9:
                     score = 2.0
                 elif recall >= 0.6:
                     score = 1.5
                 elif recall >= 0.3:
                     score = 1.0
                 else:
                     # 只要有命中 (0 < recall < 0.3)，就给 0.5 分
                     score = 0.5
                     
                 rewards.append(score)
        return rewards

    # =================================================================
    # 4. 执行 GRPO 训练
    # =================================================================
    print(" Starting GRPO training...")
    
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="grpo_medical_vlm",
        learning_rate=5e-6,          # RL 通常需要更低的学习率 (MD 建议 1e-6 ~ 1e-5)
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,           # 每个 prompt 生成多少个样本用于对比 (Group Size)
        max_prompt_length=512,
        max_completion_length=384,   # 允许生成的最大长度，从 512 降低到 384 以减少截断概率
        max_steps=50,                # 演示用
        save_steps=10,
        report_to="none",
        use_vllm=False,              # 如果显存够大且安装了 vLLM 可以开启加速
        bf16=is_bf16_supported(),
        
        # 1) 训练目标与“参考策略 + KL 约束”
        # GRPO 的核心稳定器配置
        beta=0.04,                   # KL coefficient (trl 中通常叫 beta)，MD 建议 0.01-0.1
        # clip_range=0.2,            # TRL 的 GRPOConfig 可能不直接暴露 clip_range，通常内置处理或默认值
        # temperature=0.8,           # 生成采样温度，影响探索多样性
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[xml_format_reward, length_reward, step_reward, accuracy_reward],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    print(" GRPO Training completed.")

    # =================================================================
    # 5. 保存模型
    # =================================================================
    GRPO_OUTPUT_DIR = "grpo_model"
    print(f" Saving GRPO model to '{GRPO_OUTPUT_DIR}'...")
    model.save_pretrained(GRPO_OUTPUT_DIR)
    tokenizer.save_pretrained(GRPO_OUTPUT_DIR)
    print(" Model saved successfully!")

if __name__ == "__main__":
    main()
