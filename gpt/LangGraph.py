from langgraph.graph import StateGraph, END
from typing import TypedDict
from IPython.display import display, Image
from PIL import Image # pillow
import io # 바이너리 데이터를 이미지로 변환할 때 필요

# Step 1. 워크플로우의 입력/출력 구조 정의
class WorkflowState(TypedDict):
    text: str  # 입력 데이터 타입을 명확히 정의

# Step 2. StateGraph에 스키마 적용
graph = StateGraph(WorkflowState)

# Step 3. 상태 노드 정의
graph.add_node("input", lambda state: {"text": state["text"]})  # 입력 그대로 전달
graph.add_node("search", lambda state: {"text": f"Searching: {state['text']}"})  # RAG 검색
graph.add_node("gpt", lambda state: {"text": f"GPT 응답: {state['text']}"})  # GPT 모델 실행

# Step 4. 노드 연결 (워크플로우 흐름 정의)
graph.add_edge("input", "search")
graph.add_edge("search", "gpt")
graph.add_edge("gpt", END)

# Step 5. 시작 지점(Entry Point) 설정
graph.set_entry_point("input")  # "input"을 START로 설정

# Step 6. 워크플로우 실행
workflow = graph.compile()  # `compile()`을 호출하여 실행 가능한 워크플로우 생성
result = workflow.invoke({"text": "LangGraph가 뭐야?"})  # 입력 데이터 전달

print(result)  # 최종 결과 출력

# Step 7. 시각화: Mermaid 기반 이미지로 그래프 확인
image_data = workflow.get_graph().draw_mermaid_png()  # 바이너리 데이터
image = Image.open(io.BytesIO(image_data))  # 바이너리를 이미지로 변환
image.show()  # 실제 이미지 창으로 표시