from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel


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


class DeviceModeRegistry:
    """
    deviceId 단위로 현재 활성 모드(IDLE / CALIBRATION / SESSION)를 추적.

    - ESP32 한 보드(=deviceId 한 개)는 동시에 calibration 과 session 을 진행할 수 없다.
    - WebSocket 으로 들어온 EMG 패킷은 이 registry 에서 해당 deviceId 의 mode 를
      확인한 뒤 해당 store 로 라우팅 된다.
    - mode 가 IDLE 이면 들어온 EMG 데이터는 무시(drop) 된다.
    """

    def __init__(self):
        self._states: Dict[str, DeviceState] = {}

    def get(self, device_id: str) -> DeviceState:
        return self._states.get(device_id) or DeviceState(deviceId=device_id)

    def is_idle(self, device_id: str) -> bool:
        return self.get(device_id).mode == DeviceMode.IDLE

    def set_calibration(
        self,
        device_id: str,
        calibration_session_id: str,
    ) -> DeviceState:
        current = self.get(device_id)
        if current.mode != DeviceMode.IDLE:
            raise ValueError(
                f"device {device_id} is busy in {current.mode.value} mode "
                f"(activeId={current.activeId})"
            )
        state = DeviceState(
            deviceId=device_id,
            mode=DeviceMode.CALIBRATION,
            activeId=calibration_session_id,
        )
        self._states[device_id] = state
        return state

    def set_session(self, device_id: str, session_id: str) -> DeviceState:
        current = self.get(device_id)
        if current.mode != DeviceMode.IDLE:
            raise ValueError(
                f"device {device_id} is busy in {current.mode.value} mode "
                f"(activeId={current.activeId})"
            )
        state = DeviceState(
            deviceId=device_id,
            mode=DeviceMode.SESSION,
            activeId=session_id,
        )
        self._states[device_id] = state
        return state

    def clear(self, device_id: str) -> None:
        self._states.pop(device_id, None)
