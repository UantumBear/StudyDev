# models/llama3_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from data.prompts import pmpt


class LlamaChatModel:
    def __init__(self, model_path: str, prompt_template: str, is_bf16=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
            legacy=False # LLaMA3 tokenizer 최신 방식 권장
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.chat_template = prompt_template

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if (is_bf16 and torch.cuda.is_bf16_supported()) else torch.float16,
            device_map="auto"
        )
        self.model.eval()

        if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer)) # 모델을 resize

        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id
        )

    def chat(self, user_input: str, system_prompt: str) -> str:
        """ 모델을 사용 하여 대화를 하는 함수 """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config
            )

        output_text = self.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

        return output_text


if __name__ == "__main__":
    # main.py

    from data.DevBear import system_prompt as sys_pmpt

    model_dir = "models/meta-llama/Llama-3.2-1B-Instruct"
    llm = LlamaChatModel(model_path=model_dir, prompt_template=pmpt.PROMPT_TEMPLATE)

    print("🌟 LLaMA3 구조화 챗봇 시작 (exit 입력 시 종료) 🌟")
    while True:
        user_utterance = input("You: ").strip()
        if user_utterance.lower() in ["exit", "quit"]:
            break
        answer = llm.chat(user_input=user_utterance, system_prompt=sys_pmpt.PROMPT_V1)
        print(f"Assistant: {answer}")


# python model/llama_chat_model.py