# # Tokenize 함수
# def tokenize(batch):
#     """
#     LaMA3.2 모델은 Supervised Fine-Tuning (SFT) 시 모델이 assistant 응답만 학습하도록 해야 한다고 한다.
#     labels 설정 시 user의 부분은 -100으로 마스킹 () 하기
#     """
#     results = {"input_ids": [], "attention_mask": [], "labels": []}
#     assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
#
#     for messages in batch["messages"]:
#         chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
#
#         try:
#             # 1. assistant 블록 위치 찾기
#             assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
#             if assistant_marker not in chat_text:
#                 continue
#             # 문맥 추출
#             start = chat_text.index(assistant_marker)
#             context = chat_text[max(0, start - 800):]
#
#             # 토큰화
#             enc = tokenizer(context, padding="max_length", max_length=MAX_LENGTH, truncation=True)
#             input_ids = enc["input_ids"]
#             attention_mask = enc["attention_mask"]
#             labels = input_ids.copy()
#
#             # 마스킹 처리
#             prefix_len = len(tokenizer(context[:context.index(assistant_marker)])["input_ids"])
#             labels[:prefix_len] = [-100] * prefix_len
#
#             if all(l == -100 for l in labels):
#                 continue
#
#             results["input_ids"].append(input_ids)
#             results["attention_mask"].append(attention_mask)
#             results["labels"].append(labels)
#
#         except Exception:
#             continue
#
#         return results if results["input_ids"] else None
#
# # 전처리 적용
# tokenized_train_dataset = train_dataset.map(
#     tokenize,
#     batched=True,
#     remove_columns=["messages"],
#     desc="Tokenizing train dataset"
# )
# tokenized_eval_dataset = eval_dataset.map(
#     tokenize,
#     batched=True,
#     remove_columns=["messages"],
#     desc="Tokenizing eval dataset"
# )