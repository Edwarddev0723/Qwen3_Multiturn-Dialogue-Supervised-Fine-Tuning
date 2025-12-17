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
  - multi-turn
  - taiwan
datasets:
  - your-username/coffee-order-zhtw
pipeline_tag: text-generation
model-index:
  - name: qwen3-1.7b-coffee-sft
    results:
      - task:
          type: text-generation
          name: Coffee Order Dialogue
        metrics:
          - type: eval_loss
            value: 0.0908
            name: Evaluation Loss
---

# Qwen3-1.7B Coffee Order Assistant (繁體中文)

基於 [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) 進行全參數 SFT 微調的咖啡點餐助理模型。

## 模型描述

此模型專為繁體中文（台灣）咖啡點餐場景設計，能夠：
- 理解用戶的點餐意圖（包含口語化表達如「ㄋㄟㄋㄟ」）
- 處理多輪對話，逐步確認訂單細節
- 確認訂單細節（飲品種類、冰熱、加濃縮咖啡）
- 處理訂單修改與取消
- 拒絕菜單外的品項並引導用戶選擇

### 支援的菜單項目
| 飲品 | 說明 |
|------|------|
| 美式 | 經典美式咖啡 |
| 拿鐵 | 濃縮咖啡加牛奶 |
| 燕麥奶拿鐵 | 濃縮咖啡加燕麥奶 |
| 鮮奶 | 純鮮奶 |

### 可選項目
- 冰 / 熱
- 加一份濃縮咖啡（每杯限加一份）
- 所有飲品統一為大杯

## 使用方式

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "your-username/qwen3-1.7b-coffee-sft"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

messages = [
    {
        "role": "system", 
        "content": "你是一位專業的咖啡點餐助理，負責協助使用者完成點餐。菜單包含：美式、拿鐵、燕麥奶拿鐵、鮮奶。"
    },
    {"role": "user", "content": "我想要一杯冰拿鐵"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

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

### 模型架構
| 參數 | 值 |
|------|-----|
| 架構 | Qwen3ForCausalLM |
| 隱藏層大小 | 2048 |
| 層數 | 28 |
| 注意力頭數 | 16 |
| 總參數量 | ~1.7B |

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

### 訓練結果
| 指標 | 值 |
|------|-----|
| 最終訓練 Loss | ~0.079 |
| 最佳驗證 Loss | 0.0908 |
| 總訓練步數 | 498 |

### 訓練資料
- 資料集：[your-username/coffee-order-zhtw](https://huggingface.co/datasets/your-username/coffee-order-zhtw)
- 資料筆數：2,939 筆多輪對話
- 訓練/驗證分割：90% / 10%
- 語言：繁體中文（台灣）

### 硬體環境
- 裝置：Apple Silicon (MPS)

## 對話範例

### 基本點餐
```
用戶：我要咖啡
助理：請問您想要哪一種咖啡呢？美式、拿鐵、燕麥奶拿鐵或鮮奶？
用戶：拿鐵熱的，加一份濃縮咖啡
助理：好的，一杯大杯熱拿鐵，並加一份濃縮咖啡。您的點單已完成，謝謝！
```

### 處理菜單外品項
```
用戶：我要一杯藍莓拿鐵
助理：抱歉，我們目前只提供美式、拿鐵、燕麥奶拿鐵和鮮奶，沒有藍莓拿鐵。請問您想要其他咖啡嗎？
```

### 訂單修改
```
用戶：先點一杯大杯美式冰
助理：好的，一杯大杯冰美式已加入您的點單。
用戶：改成大杯拿鐵熱
助理：好的，已將飲品改為一杯大杯熱拿鐵，謝謝！
```

## 限制與注意事項

- 此模型僅針對咖啡點餐場景進行訓練，不適用於一般對話
- 菜單項目固定，無法處理菜單外的飲品
- 每杯飲品最多只能加一份濃縮咖啡
- 所有飲品統一為大杯，無法調整杯型

## 授權

本模型基於 Apache 2.0 授權發布。

## 引用

```bibtex
@misc{qwen3-coffee-sft-2024,
  author = {Your Name},
  title = {Qwen3-1.7B Coffee Order Assistant},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/your-username/qwen3-1.7b-coffee-sft}
}
```
