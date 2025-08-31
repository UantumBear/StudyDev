# @경로: backend/routers/kakao/auth.py
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import RedirectResponse # 클라이언트를 다른 URL로 리다이렉트하는 응답 타입(기본 307).
from backend.utils.kakao.client import KakaoClient
from backend.utils.kakao.deps import get_kakao_client
from config import conf

KAKAOAUTH = APIRouter(prefix="/kakao/auth", tags=["@KAKAOAUTH"]) # tags: Swagger/Docs 그룹 표시
FRONT_SUCCESS = conf.KAKAO_FRONTEND_SUCCESS_REDIRECT

@KAKAOAUTH.get("/login")
async def kakao_login(kakao: KakaoClient = Depends(get_kakao_client)):
    url = kakao.build_authorize_url(state="xyz123") # state: CSRF 방지용 임의 문자열
    return RedirectResponse(url)

@KAKAOAUTH.get("/callback")
async def kakao_callback(
    code: str | None = None,
    error: str | None = None,
    kakao: KakaoClient = Depends(get_kakao_client),
):
    # code: 성공 시 카카오에서 ?code= 를 붙여서 보냄.
    if error:
        raise HTTPException(status_code=400, detail=f"Kakao error: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="missing code")

    token_json = await kakao.exchange_token(code)
    access_token = token_json.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="no access_token")

    user = await kakao.get_user_info(access_token)
    # 쿠키에 닉네임만 심고 프론트로 리다이렉트
    nickname = (
        user.get("kakao_account", {}).get("profile", {}).get("nickname")
        or user.get("properties", {}).get("nickname")
        or "Unknown"
    )
    resp = RedirectResponse(url=f"{FRONT_SUCCESS}?provider=kakao&login=ok")
    # TODO 운영 시엔 HttpOnly/Secure 쿠키 + JWT/세션 사용하도록 수정하기
    resp.set_cookie("nickname", nickname, httponly=False, samesite="Lax")
    return resp
