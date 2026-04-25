from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from schemas.stream_schema import EmgWindowMessage, EmgWindowAck
from services.stream_service import StreamService
from storage import (
    calibration_store,
    session_store,
    device_mode_registry,
)

router = APIRouter(
    prefix="/ai/stream",
    tags=["stream"],
)

stream_service = StreamService(
    device_mode_registry=device_mode_registry,
    calibration_store=calibration_store,
    session_store=session_store,
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
        CALIBRATION  → calibration_store 로 라우팅
        SESSION      → session_store 로 라우팅
      해서 처리한다.
    """
    await websocket.accept()
    print("EMG websocket connected")

    try:
        while True:
            raw = await websocket.receive_json()

            try:
                msg = EmgWindowMessage(**raw)
            except ValidationError as e:
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

            await websocket.send_json(ack.model_dump())

    except WebSocketDisconnect:
        print("WebSocket disconnected")
