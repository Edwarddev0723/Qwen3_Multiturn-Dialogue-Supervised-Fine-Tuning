---
license: apache-2.0
language:
  - zh
base_model: Qwen/Qwen3-1.7B
tags:
  - conversational
  - sft
  - coffee
  - traditional-chinese
  - qwen3
  - task-oriented-dialogue
datasets:
  - your-username/coffee-order-zhtw
pipeline_tag: text-generation
---

# Qwen3-1.7B Coffee Order Assistant (繁體中文)

基於 [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) 進行全參數 SFT 微調的咖啡點餐助理模型。

## 模型描述

此模型專為繁體中文咖啡點餐場景設計，能夠：
- 理解用戶的點餐意圖
- 處理多輪對話
- 確認訂單細節（飲品種類、冰熱、加濃縮咖啡）
- 處理訂單修改與取消

### 支援的菜單項目
- 美式
- 拿鐵
- 燕麥奶拿鐵
- 鮮奶

### 可選項目
- 冰/熱
- 加一份濃縮咖啡

## 使用方式

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your-username/qwen3-1.7b-coffee-sft"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

messages = [
    {"role": "system", "content": "你是一位專業的咖啡點餐助理，負責協助使用者完成點餐。菜單包含：美式、拿鐵、燕麥奶拿鐵、鮮奶。"},
    {"role": "user", "content": "我想要一杯冰拿鐵"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 訓練細節

### 訓練配置
| 參數 | 值 |
|------|-----|
| 基礎模型 | Qwen/Qwen3-1.7B |
| 訓練方式 | Full Parameter SFT |
| 學習率 | 5e-6 |
| Batch Size | 1 |
| Gradient Accumulation | 16 |
| Epochs | 3 |
| Max Length | 512 |
| Optimizer | AdamW |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.1 |

### 訓練資料
- 資料集：[your-username/coffee-order-zhtw](https://huggingface.co/datasets/your-username/coffee-order-zhtw)
- 資料筆數：約 2900+ 筆多輪對話
- 語言：繁體中文（台灣）

### 硬體
- Apple Silicon (MPS)

## 限制與注意事項

- 此模型僅針對咖啡點餐場景進行訓練，不適用於一般對話
- 菜單項目固定，無法處理菜單外的飲品
- 每杯飲品最多只能加一份濃縮咖啡
- 所有飲品統一為大杯

## 授權

本模型基於 Apache 2.0 授權發布。

## 引用

如果您使用此模型，請引用：

```bibtex
@misc{qwen3-coffee-sft,
  author = {Your Name},
  title = {Qwen3-1.7B Coffee Order Assistant},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/your-username/qwen3-1.7b-coffee-sft}
}
```
