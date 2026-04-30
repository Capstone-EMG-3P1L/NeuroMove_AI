import os

import httpx

from schemas.stream_schema import BackendInferenceSchema


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL")
BACKEND_INTENT_ENDPOINT = "/api/ai/intent"
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")


class BackendService:
    def __init__(
        self,
        base_url: str | None = BACKEND_BASE_URL,
        internal_api_key: str = INTERNAL_API_KEY,
    ):
        # import / 객체 생성 시점에는 환경변수가 없어도 앱 시작을 막지 않음
        self.base_url = base_url.rstrip("/") if base_url else None
        self.internal_api_key = internal_api_key

    def send_intent(
        self,
        payload: BackendInferenceSchema,
    ) -> bool:
        # 실제 백엔드 전송 시점에만 환경변수 검증
        if not self.base_url:
            raise ValueError("BACKEND_BASE_URL 환경변수가 설정되지 않았습니다")

        url = f"{self.base_url}{BACKEND_INTENT_ENDPOINT}"

        headers = {
            "Content-Type": "application/json",
        }

        if self.internal_api_key:
            headers["X-API-KEY"] = self.internal_api_key

        try:
            response = httpx.post(
                url,
                json=payload.model_dump(),
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            # 백엔드 응답 body 전체를 예외에 포함하지 않음
            raise ValueError(
                f"backend returned error: status={e.response.status_code}"
            )

        except httpx.RequestError as e:
            raise ValueError(f"failed to connect backend: {str(e)}")

        return response.status_code == 200