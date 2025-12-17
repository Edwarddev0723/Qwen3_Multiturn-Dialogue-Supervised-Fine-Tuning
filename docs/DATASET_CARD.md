---
license: apache-2.0
language:
  - zh
task_categories:
  - conversational
  - text-generation
tags:
  - coffee
  - traditional-chinese
  - taiwan
  - task-oriented-dialogue
  - multi-turn
  - sft
  - chat
size_categories:
  - 1K<n<10K
---

# Coffee Order Dataset (繁體中文/Traditional Chinese)

專為咖啡點餐場景設計的繁體中文多輪對話資料集，適用於訓練任務導向對話系統。

## 資料集描述

此資料集包含模擬咖啡店點餐場景的多輪對話，涵蓋各種點餐情境，包括：
- 基本點餐流程
- 訂單修改與取消
- 處理菜單外品項
- 處理超出限制的請求（如加兩份濃縮）
- 口語化表達理解

### 語言
- 繁體中文（台灣）
- 包含台灣口語表達（如「ㄋㄟㄋㄟ」）

### 資料規模
| 項目 | 數量 |
|------|------|
| 總對話數 | 2,939 |
| 平均輪數 | 2-6 輪 |
| 語言 | 繁體中文 |

## 資料格式

每筆資料為 JSONL 格式，包含 `conversations` 欄位：

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "你是一位專業的助理，負責協助使用者完成「你是一位專業的咖啡點餐助理，負責協助使用者完成點餐。菜單包含：美式、拿鐵、燕麥奶拿鐵、鮮奶。規則：1. 所有飲品統一為「大杯」。2. 可選擇「冰」或「熱」。3. 每杯可選擇「加一份濃縮咖啡」。請用自然、友善的台灣繁體中文語氣回應。」相關的任務。請用自然、友善的語氣回應。"
    },
    {
      "role": "user",
      "content": "我要咖啡"
    },
    {
      "role": "assistant",
      "content": "請問您想要哪一種咖啡呢？美式、拿鐵、燕麥奶拿鐵或鮮奶？"
    }
  ]
}
```

### 欄位說明
| 欄位 | 類型 | 說明 |
|------|------|------|
| conversations | array | 對話輪次陣列 |
| conversations[].role | string | 角色：system / user / assistant |
| conversations[].content | string | 對話內容 |

## 菜單設定

### 飲品選項
| 飲品 | 說明 |
|------|------|
| 美式 | 經典美式咖啡 |
| 拿鐵 | 濃縮咖啡加牛奶 |
| 燕麥奶拿鐵 | 濃縮咖啡加燕麥奶 |
| 鮮奶 | 純鮮奶 |

### 客製化選項
| 選項 | 可選值 |
|------|--------|
| 溫度 | 冰 / 熱 |
| 濃縮咖啡 | 加一份（每杯限一份）|
| 杯型 | 大杯（固定）|

## 對話情境分類

### 1. 基本點餐
```
用戶：我要咖啡
助理：請問您想要哪一種咖啡呢？美式、拿鐵、燕麥奶拿鐵或鮮奶？
用戶：拿鐵
助理：好的，一杯大杯拿鐵。請問是熱的還是冰的？
用戶：冰的
助理：好的，冰拿鐵。還需要加一份濃縮咖啡嗎？
```

### 2. 一次性完整點餐
```
用戶：我想點三杯，第一杯熱美式加濃縮、第二杯冰燕麥奶拿鐵、第三杯熱鮮奶
助理：好的，您的訂單如下：
1. 一杯熱美式大杯加濃縮咖啡
2. 一杯冰燕麥奶拿鐵大杯
3. 一杯熱鮮奶大杯
請問還需要其他嗎？
```

### 3. 訂單修改
```
用戶：先點一杯大杯美式冰
助理：好的，一杯大杯冰美式已加入您的點單。
用戶：改成大杯拿鐵熱
助理：好的，已將飲品改為一杯大杯熱拿鐵，謝謝！
```

### 4. 訂單取消
```
用戶：我想取消剛剛點的那杯咖啡
助理：已經為您取消剛剛的訂單。若還有其他需要，隨時告訴我喔！
```

### 5. 處理菜單外品項
```
用戶：我要一杯藍莓拿鐵
助理：抱歉，我們目前只提供美式、拿鐵、燕麥奶拿鐵和鮮奶，沒有藍莓拿鐵。請問您想要其他咖啡嗎？
```

### 6. 處理超出限制的請求
```
用戶：我要一杯拿鐵加兩份濃縮咖啡
助理：每杯只能加一份濃縮咖啡，請問您要加一份還是不要加？
```

### 7. 口語化表達
```
用戶：ㄋㄟㄋㄟ
助理：請問您想點什麼飲品？我們提供美式、拿鐵、燕麥奶拿鐵、鮮奶...
```

## 使用方式

### 使用 Hugging Face Datasets

```python
from datasets import load_dataset

dataset = load_dataset("your-username/coffee-order-zhtw")
print(dataset)
print(dataset["train"][0])
```

### 本地載入 JSONL

```python
import json

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

data = load_jsonl("coffee_order_zhtw.jsonl")
print(f"載入 {len(data)} 筆對話")
```

### 轉換為訓練格式

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

def format_conversation(example):
    messages = [{"role": turn["role"], "content": turn["content"]} 
                for turn in example["conversations"]]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}
```

## 適用場景

- 任務導向對話系統訓練
- 咖啡點餐 Chatbot 開發
- 繁體中文 SFT 微調
- 多輪對話理解研究

## 授權

本資料集基於 Apache 2.0 授權發布。

## 引用

```bibtex
@misc{coffee-order-zhtw-2024,
  author = {Your Name},
  title = {Coffee Order Dataset (Traditional Chinese)},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/your-username/coffee-order-zhtw}
}
```
