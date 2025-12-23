# train_grpo.py - Medical VLM Reinforcement Learning (GRPO) Script (Revised)
import os
import re
import torch
from unsloth import FastVisionModel, is_bf16_supported
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# =========================
# 0) 训练目标开关
# =========================
TARGET_LANG = "zh"   # "zh" 或 "en"
USE_ACCURACY_REWARD = False  #  GT 是英文时：强烈建议先关掉，否则模型会被拉回英文
ACC_WEIGHT = 0.3     # 如果你坚持开 accuracy_reward，把权重压低（例如 0.2~0.4）

# 奖励权重（可调）
W_FORMAT = 1.0
W_STOP   = 0.6
W_LEN    = 0.8
W_STEP   = 0.2
W_LANG   = 0.8
W_EARLY  = 1.2
W_ACC    = ACC_WEIGHT

# 最小长度约束
MIN_REASONING_CHARS = 120
MIN_ANSWER_CHARS    = 30
MIN_TOTAL_CHARS     = 220    
MAX_REASONING_CHARS = 800    

STRICT_XML_PATTERN = re.compile(
    r"^\s*<reasoning>(?P<r>.*?)</reasoning>\s*<answer>(?P<a>.*?)</answer>\s*$",
    re.DOTALL
)

def _get_text(completion):
    # TRL/Unsloth 里 completion 可能是 list[{"content": "..."}]
    if isinstance(completion, list):
        return completion[0].get("content", "")
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)

def _parse_xml(text: str):
    m = STRICT_XML_PATTERN.search(text)
    if not m:
        return False, "", ""
    reasoning = m.group("r").strip()
    answer = m.group("a").strip()
    return True, reasoning, answer

def _count_zh_chars(s: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", s))

def _count_en_letters(s: str) -> int:
    return len(re.findall(r"[A-Za-z]", s))

# =========================
# 1) 主程序
# =========================
def main():
    print(" Starting Medical VLM GRPO Training (Revised)...")

    # 1) 模型加载
    if os.path.exists("lora_model"):
        print(" Loading SFT model from: lora_model")
        MODEL_NAME = "lora_model"
    else:
        MODEL_NAME = "/root/autodl-tmp/models/unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
        print(f"'lora_model' not found! Using base model: {MODEL_NAME}")

    OUTPUT_DIR = "outputs_grpo"

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        use_gradient_checkpointing="unsloth",
        local_files_only=True,
    )

    # LoRA 确保可训练
    if hasattr(model, "peft_config") and len(model.peft_config) > 0:
        print("✅ Model already has LoRA adapters. Enabling training mode...")
        FastVisionModel.for_training(model)
    else:
        print("🆕 Adding new LoRA adapters...")
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,   # 稍微加点 dropout，RL 更稳一些
            bias="none",
            use_rslora=False,
        )

    # 2) 数据
    print(" Loading dataset...")
    dataset = load_dataset("./data", split="train")

    if TARGET_LANG == "zh":
        system_prompt = """
你是一名专业的放射科医生。请分析给定的医疗图像。
要求：
1) 必须使用中文作答（医学名词可保留英文缩写）。
2) 严格只输出以下两个标签，且不要输出多余文本：

<reasoning>
写下观察要点、推理过程、依据（不少于120字）。
</reasoning>
<answer>
给出最终诊断结论与关键发现（不少于30字）。
</answer>
"""
        user_text = "请用中文分析这张图片。"
    else:
        system_prompt = """
You are a professional radiologist. Analyze the given medical image.
Strictly output ONLY the following two tags:

<reasoning>
Write your observations and reasoning.
</reasoning>
<answer>
Write your final diagnosis.
</answer>
"""
        user_text = "Please analyze this image."

    def format_data(sample):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": sample["image"]},
                                         {"type": "text", "text": user_text}]},
        ]
        return {
            "prompt": messages,
            "ground_truth": sample["caption"],  # 仍然保留，以便未来切换
        }

    # 备注：如果image 是 PIL/Arrow Image，num_proc 多进程有时会不稳定；不稳就改回 num_proc=1
    dataset = dataset.map(
        format_data,
        remove_columns=["image", "caption", "image_id", "cui"],
        num_proc=4
    )

    # =========================
    # 3) Reward functions
    # =========================
    def format_reward(completions, **kwargs):
        rewards = []
        for c in completions:
            text = _get_text(c)
            ok, r, a = _parse_xml(text)
            if not ok:
                rewards.append(0.0)
                continue

            # 强约束：reasoning/answer 都要够长
            if len(r) < MIN_REASONING_CHARS or len(a) < MIN_ANSWER_CHARS:
                rewards.append(0.2)  # 结构对但太短：只给很小的保底，避免投机
            else:
                rewards.append(1.0)
        return [x * W_FORMAT for x in rewards]

    def stop_reward(completions, **kwargs):
        rewards = []
        for c in completions:
            text = _get_text(c)
            ok, r, a = _parse_xml(text)
            if not ok:
                rewards.append(-0.2)  
                continue

            # 只有在满足最小内容后闭合才奖励；否则闭合也扣分
            if len(r) >= MIN_REASONING_CHARS and len(a) >= MIN_ANSWER_CHARS:
                rewards.append(0.5)
            else:
                rewards.append(-0.5)
        return [x * W_STOP for x in rewards]

    def length_reward(completions, **kwargs):
        rewards = []
        for c in completions:
            text = _get_text(c)
            ok, r, a = _parse_xml(text)
            if not ok:
                rewards.append(0.0)
                continue

            # reasoning 舒适区奖励；太短重罚；太长轻罚
            L = len(r)
            if L < MIN_REASONING_CHARS:
                rewards.append(-1.0)
            elif L <= MAX_REASONING_CHARS:
                rewards.append(0.6)
            else:
                rewards.append(-0.2)
        return [x * W_LEN for x in rewards]

    def early_stop_penalty(completions, **kwargs):
        rewards = []
        for c in completions:
            text = _get_text(c)
            # 直接对“整体文本长度”做硬惩罚
            if len(text.strip()) < MIN_TOTAL_CHARS:
                rewards.append(-1.0)
            else:
                rewards.append(0.0)
        return [x * W_EARLY for x in rewards]

    def step_reward(completions, **kwargs):
        rewards = []
        step_patterns = [r"\d+\.", r"Step\s*\d+", r"首先", r"其次", r"最后", r"第一", r"第二", r"第三"]
        for c in completions:
            text = _get_text(c)
            ok, r, _ = _parse_xml(text)
            if not ok:
                rewards.append(0.0)
                continue
            step_count = 0
            for p in step_patterns:
                step_count += len(re.findall(p, r))
            rewards.append(min(step_count * 0.1, 0.3))
        return [x * W_STEP for x in rewards]

    def language_reward(completions, **kwargs):
        rewards = []
        if TARGET_LANG != "zh":
            return [0.0 for _ in completions]

        for c in completions:
            text = _get_text(c)
            ok, r, a = _parse_xml(text)
            if not ok:
                rewards.append(0.0)
                continue

            s = (r + "\n" + a)
            zh = _count_zh_chars(s)
            en = _count_en_letters(s)
            total = max(len(s), 1)

            zh_ratio = zh / total
            # 典型英文回答：zh_ratio 很低、en 很多
            if zh_ratio >= 0.10:
                rewards.append(0.6)
            elif en >= 50 and zh_ratio < 0.03:
                rewards.append(-0.8)
            else:
                rewards.append(0.0)
        return [x * W_LANG for x in rewards]

    # 你原来的 accuracy_reward（英文 GT）会把模型拉回英文，默认先关闭
    def accuracy_reward(completions, ground_truth, **kwargs):
        rewards = []
        stop_words = {"the", "is", "a", "an", "of", "in", "on", "at", "and", "with", "to", "for", "it", "this", "that"}
        for completion, ref_answer in zip(completions, ground_truth):
            text = _get_text(completion)
            ok, _, a = _parse_xml(text)
            pred = (a if ok else text).lower().strip()

            pred_clean = re.sub(r"[^\w\s]", " ", pred)
            ref_clean  = re.sub(r"[^\w\s]", " ", str(ref_answer).lower())

            ref_tokens  = set([w for w in ref_clean.split() if w not in stop_words and len(w) > 2])
            pred_tokens = set([w for w in pred_clean.split() if w not in stop_words and len(w) > 2])

            if not ref_tokens:
                rewards.append(0.2)
                continue
            inter = ref_tokens.intersection(pred_tokens)
            if not inter:
                rewards.append(0.0)
            else:
                recall = len(inter) / len(ref_tokens)
                if recall >= 0.6:
                    rewards.append(1.0)
                elif recall >= 0.3:
                    rewards.append(0.6)
                else:
                    rewards.append(0.3)
        return [x * W_ACC for x in rewards]

    reward_funcs = [format_reward, stop_reward, length_reward, early_stop_penalty, step_reward, language_reward]
    if USE_ACCURACY_REWARD:
        reward_funcs.append(accuracy_reward)

    # =========================
    # 4) GRPO Config
    # =========================
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="grpo_medical_vlm_revised",
        learning_rate=2e-6,           
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.05,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=512,    
        max_steps=60,                 # 先跑更久看趋势
        save_steps=20,
        report_to="none",
        use_vllm=False,
        bf16=is_bf16_supported(),
        beta=0.08,                    # KL 约束稍微加强，压住发散/投机
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    GRPO_OUTPUT_DIR = "grpo_model"
    print(f" Saving GRPO model to '{GRPO_OUTPUT_DIR}'...")
    model.save_pretrained(GRPO_OUTPUT_DIR)
    tokenizer.save_pretrained(GRPO_OUTPUT_DIR)
    print("✅ Done.")

if __name__ == "__main__":
    main()
