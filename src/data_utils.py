"""
資料處理工具
"""

import json
from typing import List, Dict, Any
from datasets import Dataset


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    載入 JSONL 格式的資料集
    
    Args:
        path: JSONL 檔案路徑
        
    Returns:
        資料列表
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def format_conversation(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """
    將對話轉換為訓練格式
    
    Args:
        example: 包含 conversations 欄位的字典
        tokenizer: HuggingFace tokenizer
        
    Returns:
        包含 text 欄位的字典
    """
    conversations = example["conversations"]
    messages = [{"role": turn["role"], "content": turn["content"]} 
                for turn in conversations]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}


def tokenize_function(examples: Dict[str, List], tokenizer, max_length: int = 512):
    """
    Tokenize 資料集
    
    Args:
        examples: 批次資料
        tokenizer: HuggingFace tokenizer
        max_length: 最大序列長度
        
    Returns:
        Tokenized 結果
    """
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    result["labels"] = result["input_ids"].copy()
    return result


def prepare_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    test_size: float = 0.1,
    seed: int = 42
) -> tuple:
    """
    準備訓練和驗證資料集
    
    Args:
        data_path: 資料檔案路徑
        tokenizer: HuggingFace tokenizer
        max_length: 最大序列長度
        test_size: 驗證集比例
        seed: 隨機種子
        
    Returns:
        (train_dataset, eval_dataset) 元組
    """
    # 載入原始資料
    raw_data = load_jsonl(data_path)
    print(f"載入 {len(raw_data)} 筆對話資料")
    
    # 格式化對話
    formatted_data = [
        format_conversation({"conversations": d["conversations"]}, tokenizer) 
        for d in raw_data
    ]
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # 分割訓練/驗證集
    split_dataset = tokenized_dataset.train_test_split(
        test_size=test_size, 
        seed=seed
    )
    
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"訓練集: {len(train_dataset)} 筆")
    print(f"驗證集: {len(eval_dataset)} 筆")
    
    return train_dataset, eval_dataset
