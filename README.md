# Qwen3 Coffee Order SFT

基於 Qwen3-1.7B 的繁體中文咖啡點餐助理，使用全參數 SFT 微調。

## 專案簡介

此專案展示如何使用 Supervised Fine-Tuning (SFT) 訓練一個任務導向對話系統。模型能夠：

- 理解用戶點餐意圖（包含口語化表達）
- 處理多輪對話，逐步確認訂單
- 處理訂單修改與取消
- 拒絕菜單外品項並引導選擇

## 快速開始

### 環境安裝

```bash
pip install -r requirements.txt
```

### 使用訓練好的模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your-username/qwen3-1.7b-coffee-sft"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "你是一位專業的咖啡點餐助理..."},
    {"role": "user", "content": "我想要一杯冰拿鐵"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 訓練模型

```bash
# 開啟 Jupyter Notebook
jupyter notebook notebooks/train.ipynb
```

## 專案結構

```
.
├── README.md                 # 專案說明
├── LICENSE                   # 授權條款
├── requirements.txt          # Python 依賴
├── .gitignore               # Git 忽略規則
├── configs/
│   └── training_config.yaml # 訓練配置
├── data/
│   └── coffee_order_zhtw.jsonl  # 訓練資料
├── notebooks/
│   ├── train.ipynb          # 訓練 notebook
│   └── upload_to_hub.ipynb  # 上傳到 HuggingFace
├── src/
│   ├── __init__.py
│   ├── data_utils.py        # 資料處理工具
│   └── inference.py         # 推論腳本
├── docs/
│   ├── MODEL_CARD.md        # 模型卡
│   └── DATASET_CARD.md      # 資料集卡
└── outputs/                  # 訓練輸出（gitignore）
    └── qwen3_coffee_sft/
```

## 菜單設定

| 飲品 | 溫度選項 | 加購選項 |
|------|----------|----------|
| 美式 | 冰/熱 | 加一份濃縮 |
| 拿鐵 | 冰/熱 | 加一份濃縮 |
| 燕麥奶拿鐵 | 冰/熱 | 加一份濃縮 |
| 鮮奶 | 冰/熱 | 加一份濃縮 |

> 所有飲品統一為大杯，每杯最多加一份濃縮咖啡

## 訓練配置

| 參數 | 值 |
|------|-----|
| 基礎模型 | Qwen/Qwen3-1.7B |
| 訓練方式 | Full Parameter SFT |
| 學習率 | 5e-6 |
| Batch Size | 1 × 16 (gradient accumulation) |
| Epochs | 3 |
| 最佳驗證 Loss | 0.0908 |

## 對話範例

```
用戶：我要咖啡
助理：請問您想要哪一種咖啡呢？美式、拿鐵、燕麥奶拿鐵或鮮奶？

用戶：拿鐵
助理：好的，一杯大杯拿鐵。請問是熱的還是冰的？

用戶：冰的
助理：好的，冰拿鐵。還需要加一份濃縮咖啡嗎？

用戶：不需要
助理：好的，已為您安排一杯大杯冰拿鐵。謝謝！
```

## HuggingFace Hub

- 模型：[your-username/qwen3-1.7b-coffee-sft](https://huggingface.co/your-username/qwen3-1.7b-coffee-sft)
- 資料集：[your-username/coffee-order-zhtw](https://huggingface.co/datasets/your-username/coffee-order-zhtw)

## 授權

本專案基於 [Apache 2.0](LICENSE) 授權發布。

## 引用

```bibtex
@misc{qwen3-coffee-sft-2024,
  author = {Your Name},
  title = {Qwen3 Coffee Order SFT},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/your-username/qwen3-coffee-sft}
}
```

## 致謝

- [Qwen Team](https://github.com/QwenLM/Qwen) - 基礎模型
- [HuggingFace](https://huggingface.co) - 模型託管與訓練工具
