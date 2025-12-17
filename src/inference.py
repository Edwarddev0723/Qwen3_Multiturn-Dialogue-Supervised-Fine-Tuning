"""
推論工具
"""

import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class CoffeeOrderAssistant:
    """咖啡點餐助理"""
    
    DEFAULT_SYSTEM_PROMPT = (
        "你是一位專業的咖啡點餐助理，負責協助使用者完成點餐。"
        "菜單包含：美式、拿鐵、燕麥奶拿鐵、鮮奶。"
        "規則：1. 所有飲品統一為「大杯」。"
        "2. 可選擇「冰」或「熱」。"
        "3. 每杯可選擇「加一份濃縮咖啡」。"
        "請用自然、友善的台灣繁體中文語氣回應。"
    )
    
    def __init__(
        self,
        model_name: str = "your-username/qwen3-1.7b-coffee-sft",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        初始化助理
        
        Args:
            model_name: 模型名稱或路徑
            device: 運算裝置 (auto/cuda/mps/cpu)
            torch_dtype: 模型精度
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto" if device is None else device
        )
        
        self.model.eval()
        self.conversation_history: List[Dict[str, str]] = []
        
    def reset(self):
        """重置對話歷史"""
        self.conversation_history = []
        
    def chat(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        進行對話
        
        Args:
            user_input: 用戶輸入
            system_prompt: 系統提示詞（可選）
            max_new_tokens: 最大生成 token 數
            temperature: 生成溫度
            top_p: Top-p 採樣
            
        Returns:
            助理回應
        """
        # 建立訊息列表
        messages = []
        
        # 加入系統提示
        if not self.conversation_history:
            messages.append({
                "role": "system",
                "content": system_prompt or self.DEFAULT_SYSTEM_PROMPT
            })
        
        # 加入歷史對話
        messages.extend(self.conversation_history)
        
        # 加入當前用戶輸入
        messages.append({"role": "user", "content": user_input})
        
        # 生成回應
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # 解碼回應
        full_response = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # 提取助理回應（移除輸入部分）
        input_decoded = self.tokenizer.decode(
            inputs["input_ids"][0], 
            skip_special_tokens=True
        )
        assistant_response = full_response[len(input_decoded):].strip()
        
        # 更新對話歷史
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        return assistant_response


def main():
    """互動式對話示範"""
    print("載入模型中...")
    assistant = CoffeeOrderAssistant()
    
    print("\n" + "="*50)
    print("☕ 咖啡點餐助理")
    print("輸入 'quit' 結束對話，'reset' 重置對話")
    print("="*50 + "\n")
    
    while True:
        user_input = input("你：").strip()
        
        if user_input.lower() == "quit":
            print("謝謝光臨，再見！")
            break
        elif user_input.lower() == "reset":
            assistant.reset()
            print("[對話已重置]\n")
            continue
        elif not user_input:
            continue
            
        response = assistant.chat(user_input)
        print(f"助理：{response}\n")


if __name__ == "__main__":
    main()
