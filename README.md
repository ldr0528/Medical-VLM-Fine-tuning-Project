# Medical VLM Fine-tuning & Inference Assistant

这是一个基于 [Unsloth](https://github.com/unslothai/unsloth) 和 [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) 的医疗视觉大模型微调与部署项目。本项目演示了如何使用 LoRA 高效微调多模态大模型，使其具备专业的医疗影像诊断能力，并提供了一个基于 Streamlit 的可视化对话界面。

## 🚀 项目功能

*   **高效微调**：使用 Unsloth 加速 Qwen2-VL-7B 的 LoRA 微调，大幅降低显存需求并提升训练速度。
*   **医疗场景适配**：针对医疗影像（如 X 光、CT 等）进行指令微调，使模型能够像放射科医生一样描述病灶。
*   **可视化交互**：提供 Streamlit Web 应用，支持上传图片进行多轮医学对话。
*   **多轮对话支持**：微调后的模型支持结合上下文的多轮问答。

## 📂 项目结构

```
.
├── train.py            # 微调脚本 (Python 版本)
├── app.py              # Streamlit 可视化部署应用
├── requirements.txt    # 项目依赖文件
├── README.md           # 项目说明文档
├── data/               # 训练数据集目录
└── lora_model/         # (自动生成) 微调后的 LoRA 权重
```

## 🛠️ 环境安装

1.  **克隆项目**
    ```bash
    git clone https://github.com/your-username/medical-vlm-finetune.git
    cd medical-vlm-finetune
    ```

2.  **安装依赖**
    建议使用 Conda 创建虚拟环境：
    ```bash
    conda create -n vlm python=3.10
    conda activate vlm
    pip install -r requirements.txt
    ```

    *注意：Unsloth 的安装可能需要特定的 CUDA 版本，请参考 [Unsloth 官方文档](https://github.com/unslothai/unsloth) 进行适配。*

## 🏃‍♂️ 快速开始

### 1. 模型微调

#### 使用 Python 脚本
直接运行以下命令进行训练：
```bash
python train.py
```
训练过程会自动加载模型、处理数据、微调并保存权重到 `lora_model/` 目录。

### 2. 启动 Web 应用
训练完成后，使用 Streamlit 启动可视化界面：
```bash
streamlit run app.py
```
访问终端显示的 URL（通常是 http://localhost:8501 或 http://localhost:6006）即可使用。

## 🧪 效果展示
![alt text](docs/image.png)
![alt text](docs/image-1.png)



## 📜 许可证

MIT License
