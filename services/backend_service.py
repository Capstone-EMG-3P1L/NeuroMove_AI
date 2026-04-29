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
        if not base_url:
            raise ValueError("BACKEND_BASE_URL environment variable is required")

        self.base_url = base_url.rstrip("/")
        self.internal_api_key = internal_api_key

    def send_intent(
        self,
        payload: BackendInferenceSchema,
    ) -> bool:
        # AI 추론 결과를 백엔드로 전송
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
            raise ValueError(
                f"backend returned error: "
                f"status={e.response.status_code}, body={e.response.text}"
            )

        except httpx.RequestError as e:
            raise ValueError(f"failed to connect backend: {str(e)}")

        return response.status_code == 200