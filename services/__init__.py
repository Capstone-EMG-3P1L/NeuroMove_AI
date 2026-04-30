"""
services 패키지 - 도메인 서비스 싱글톤 모음.

각 라우터는 여기서 동일 인스턴스를 import 해서 공유한다.
StreamService 가 SessionService / CalibrationService  에 의존하기 때문에
인스턴스를 한 곳에서 함께 만들어 주입해 두는 편이 안전하다.
"""

from storage import (
    session_store,
    calibration_store,
    device_mode_registry,
)
from services.session_service import SessionService
from services.calibration_service import CalibrationService
from services.stream_service import StreamService
from services.backend_service import backend_service


session_service = SessionService(
    session_store=session_store,
    device_mode_registry=device_mode_registry,
)

calibration_service = CalibrationService(
    calibration_store=calibration_store,
    device_mode_registry=device_mode_registry,
)



stream_service = StreamService(
    device_mode_registry=device_mode_registry,
    calibration_service=calibration_service,
    session_service=session_service,
    backend_service=backend_service,
)

__all__ = [
    "session_service",
    "calibration_service",
    "stream_service",
    "SessionService",
    "CalibrationService",
    "StreamService",
    "backend_service",
]