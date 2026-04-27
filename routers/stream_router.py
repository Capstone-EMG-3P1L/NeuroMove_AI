from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from schemas.stream_schema import EmgWindowMessage, EmgWindowAck
from services import stream_service

router = APIRouter(
    prefix="/ai/stream",
    tags=["stream"],
)


@router.get("/ping")
def stream_ping():
    return {"message": "stream router connected"}


@router.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    """
    ESP32 ─ AI 서버 단일 EMG 스트림 채널.

    - ESP32 는 calibration / session 구분 없이 EMG window 만 계속 전송한다.
    - 서버는 deviceId 기준으로 DeviceModeRegistry 의 mode 를 보고
        IDLE         → drop
        CALIBRATION  → CalibrationService.append_calibration_data
        SESSION      → SessionService.append_window
      해서 처리한다.
    """
    await websocket.accept()
    print("EMG websocket connected")

    try:
        while True:
            try:
                raw = await websocket.receive_json()
                msg = EmgWindowMessage.model_validate(raw)
            except (ValidationError, ValueError, TypeError) as e:
                await websocket.send_json({
                    "type": "emg_window_ack",
                    "success": False,
                    "message": str(e),
                    "data": None,
                })
                continue

            try:
                ack: EmgWindowAck = stream_service.handle_emg_window(msg)
            except Exception as e:
                await websocket.send_json({
                    "type": "emg_window_ack",
                    "success": False,
                    "message": str(e),
                    "data": None,
                })
                continue

            await websocket.send_json(ack.model_dump(by_alias=True))

    except WebSocketDisconnect:
        print("WebSocket disconnected")
