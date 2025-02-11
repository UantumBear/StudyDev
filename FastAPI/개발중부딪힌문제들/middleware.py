"""
nginx 파일을 직접 설정할 수 없는 (서버 파일을 건드리지 못하는) 환경 상
http->https 로 리디렉트 하는 과정에서
pagination 의 prams 가 제대로 들어오지 않았다.

add_pagination(app) 이 충돌하는 문제가 아니였고,

middleware 를 통해 응답을 리턴할 때 url에서 쿼리 파람이 빠졌던 문제였다.
"""
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from urllib.parse import urlencode
app = FastAPI()
@app.middleware("http")
async def redirect_middleware(request: Request, call_next):
    if request.headers.get("x-forwarded-proto", "") == "http":
        query_params = urlencode(request.query_params)
        https_url = request.url.replace(scheme="https")
        https_url = f"{https_url}?{query_params}" if query_params else str(https_url)
        # 이미 리디렉션 된 url은 처리하지 않음
        if https_url == str(request.url):
            return await call_next(request)
        return RedirectResponse(url=str(https_url), status_code=307) # POST, PUT 요청이 리디렉트 될 때 요청 방식(쿼리파람)을 유지한다.