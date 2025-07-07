"""
@경로 data/prompts/pmpt.py
LLaMA 3 기반 모델이
 tokenizer.apply_chat_template()로
  사용자-어시스턴트 메시지를 정리할 때 쓰는 ChatML 포맷
"""

PROMPT_TEMPLATE = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}\n"
        "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}\n"
        "{% endif %}{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n"
    )

