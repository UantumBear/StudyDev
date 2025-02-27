from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from datetime import datetime
import uvicorn
import os
from SampleFastapi.guide.api.first_backend_api import FIRST # APIRouter 가져오는 방식 1
from SampleFastapi.guide.api.second_backend_api import secondBackendApi  # APIRouter 가져오는 방식 2 (클래스 인스턴스를 통해 가져오기)

PROJECT_ROOT_DIR = os.path.abspath(__file__)  # 현재 파일 (run.py)의 절대경로

# FastAPI 앱 생성
app = FastAPI()
# 하위 라우터들 등록
app.include_router(FIRST) # APIRouter 등록하는 방식 1
app.include_router(secondBackendApi.router) # APIRouter 등록하는 방식 2


# run.py의 경로를 기준으로 템플릿 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "guide/templates")  # 템플릿 경로 설정
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# HTML 페이지 렌더링
@app.get("/render/index")
def render_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    #  uvicorn SampleFastapi.run:app --host 0.0.0.0 --port 8000 --reload