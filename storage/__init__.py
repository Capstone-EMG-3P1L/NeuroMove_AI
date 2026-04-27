"""
storage 패키지 - 프로세스 단위 싱글톤 store / registry 모음.

각 라우터는 여기서 동일 인스턴스를 import 해서 공유한다.
(stream_router, calibration_router, session_router 가 같은 store 를 봐야
calibration / session 시작·종료와 WebSocket 으로 들어오는 EMG 데이터가 일관되게 처리됨)
"""

from storage.calibration_store import CalibrationSessionStore
from storage.session_store import SessionStore
from storage.device_mode_registry import DeviceModeRegistry

calibration_store = CalibrationSessionStore()
session_store = SessionStore()
device_mode_registry = DeviceModeRegistry()

__all__ = [
    "calibration_store",
    "session_store",
    "device_mode_registry",
    "CalibrationSessionStore",
    "SessionStore",
    "DeviceModeRegistry",
]
