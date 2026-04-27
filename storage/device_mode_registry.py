import time
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel
from threading import Lock


class DeviceMode(str, Enum):
    IDLE = "IDLE"
    CALIBRATION = "CALIBRATION"
    SESSION = "SESSION"


class DeviceState(BaseModel):
    deviceId: str
    mode: DeviceMode = DeviceMode.IDLE
    # mode 가 CALIBRATION 이면 calibrationSessionId,
    # mode 가 SESSION 이면 sessionId 가 들어감.
    activeId: Optional[str] = None
    # 마지막 EMG 패킷 도착(또는 모드 전환) 시각, epoch ms.
    # IDLE 상태에서는 None. lock 누수 방어 sweeper 가 이 값으로 stale 판단.
    lastActiveAt: Optional[int] = None


class DeviceModeRegistry:
    """
    deviceId 단위로 현재 활성 모드(IDLE / CALIBRATION / SESSION) 와
    마지막 활동 시각을 추적.

    - ESP32 한 보드(=deviceId 한 개)는 동시에 calibration 과 session 을 진행할 수 없다.
    - WebSocket 으로 들어온 EMG 패킷은 이 registry 에서 해당 deviceId 의 mode 를
      확인한 뒤 해당 store 로 라우팅 된다.
    - mode 가 IDLE 이면 들어온 EMG 데이터는 무시(drop) 된다.

    Lock 누수 방어:
    - WebSocket 비정상 종료, ESP32 다운, 사용자가 /end 를 호출 안 한 채로 앱 닫음 등
      "정상 종료 호출이 빠지는" 모든 케이스에서 _states 가 영원히 점유되어 같은 device 로
      재시작 못 하는 문제를 막기 위함.
    - 매 EMG 패킷 도착 시 touch() 가 lastActiveAt 을 갱신.
    - 백그라운드 sweeper 가 주기적으로 sweep_stale() 을 호출해
      idle_threshold_ms 초과한 device 를 강제 해제.
    """

    def __init__(self):
        self._states: Dict[str, DeviceState] = {}
        self._lock = Lock()

    def get(self, device_id: str) -> DeviceState:
        with self._lock:
            return self._states.get(device_id) or DeviceState(deviceId=device_id)

    def is_idle(self, device_id: str) -> bool:
        with self._lock:
            state = self._states.get(device_id) or DeviceState(deviceId=device_id)
            return state.mode == DeviceMode.IDLE

    def set_calibration(
        self,
        device_id: str,
        calibration_session_id: str,
    ) -> DeviceState:
        
        with self._lock:
            current = self._states.get(device_id) or DeviceState(deviceId=device_id)

            if current.mode != DeviceMode.IDLE:
                raise ValueError(
                    f"device {device_id} is busy in {current.mode.value} mode "
                    f"(activeId={current.activeId})"
                )
            state = DeviceState(
                deviceId=device_id,
                mode=DeviceMode.CALIBRATION,
                activeId=calibration_session_id,
                lastActiveAt=_now_ms(),
            )
            self._states[device_id] = state
            return state

    def set_session(self, device_id: str, session_id: str) -> DeviceState:
        with self._lock:
            current = self._states.get(device_id) or DeviceState(deviceId=device_id)

            if current.mode != DeviceMode.IDLE:
                raise ValueError(
                    f"device {device_id} is busy in {current.mode.value} mode "
                    f"(activeId={current.activeId})"
                )
            state = DeviceState(
                deviceId=device_id,
                mode=DeviceMode.SESSION,
                activeId=session_id,
                lastActiveAt=_now_ms(),
            )
            self._states[device_id] = state
            return state

    def touch(self, device_id: str) -> None:
        """
        활성 device 의 lastActiveAt 갱신.
        StreamService 가 IDLE 이 아닌 모든 EMG 패킷에 대해 호출한다.
        IDLE 이라 _states 에 없는 device 는 no-op.
        """
        with self._lock:
            state = self._states.get(device_id)
            if state is not None:
                state.lastActiveAt = _now_ms()

    def clear(self, device_id: str) -> None:
        with self._lock:
            self._states.pop(device_id, None)

    def sweep_stale(self, idle_threshold_ms: int) -> List[str]:
        """
        lastActiveAt 이 idle_threshold_ms 보다 오래된 device 의 lock 을 강제 해제.
        반환값: 청소된 deviceId 목록 (로깅 용도).
        """
        now = _now_ms()
        
        
        with self._lock:
            stale_device_ids = [
                device_id
                for device_id, state in self._states.items()
                if state.lastActiveAt is not None
                and now - state.lastActiveAt > idle_threshold_ms
            ]

            for device_id in stale_device_ids:
                self._states.pop(device_id, None)

            return stale_device_ids


def _now_ms() -> int:
    return int(time.time() * 1000)
