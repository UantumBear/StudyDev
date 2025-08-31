# @경로: backend/utils/kakao/client.py

import httpx
from urllib.parse import urlencode
from config import conf

class KakaoClient:
    """
    Kakao OAuth2 클라이언트
    - 로그인 URL 생성
    - 인증 코드 → 토큰 교환
    - 액세스 토큰으로 사용자 정보 조회
    """

    def __init__(self):
        # 환경변수 기반 설정
        self.client_id = conf.KAKAO_CLIENT_ID
        self.client_secret = getattr(conf, "KAKAO_CLIENT_SECRET", "")
        self.redirect_uri = conf.KAKAO_REDIRECT_URI
        self.auth_base = conf.KAKAO_AUTH_BASE
        self.api_base = conf.KAKAO_API_BASE

        if not self.client_id or not self.redirect_uri:
            raise ValueError("KAKAO_CLIENT_ID/KAKAO_REDIRECT_URI 환경변수가 필요합니다.")

    def build_authorize_url(self, state: str | None = None, scope: list[str] | None = None) -> str:
        """사용자를 카카오 로그인 페이지로 보낼 authorize URL 생성"""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
        }
        if state:
            params["state"] = state
        if scope:
            params["scope"] = " ".join(scope)
        return f"{self.auth_base}/oauth/authorize?{urlencode(params)}"

    async def exchange_token(self, code: str) -> dict:
        """인가 코드(code) → 액세스 토큰 교환"""
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "code": code,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret

        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{self.auth_base}/oauth/token",
                data=data,
                headers={"Accept": "application/json"},
            )
            r.raise_for_status()
            return r.json()

    async def get_user_info(self, access_token: str) -> dict:
        """액세스 토큰으로 사용자 정보 가져오기"""
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{self.api_base}/v2/user/me", headers=headers)
            r.raise_for_status()
            return r.json()
