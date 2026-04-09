from fastapi import APIRouter,HTTPException,WebSocket,WebSocketDisconnect
from pydantic import ValidationError
from schemas.stream_schema import *;
from services.stream_service import StreamService

router = APIRouter(
    prefix="/ai/stream",
    tags=["stream"]
)

stream_service = StreamService()


@router.get("/ping")
def stream_ping():
    return {"message": "stream router connected"}

@router.websocket("/ws")
async def websocket_handler(websocket:WebSocket):
    await websocket.accept()
    print("EMG websocket connected")
    
    try:
        while True:
            raw = await websocket.receive_json()
            
            msg_type = raw.get("type")

            try:
                if msg_type == "calibration_window":
                    req = CalibrationWSRequest(**raw)
                    res = stream_service.handle_calibration(req)

                elif msg_type == "driving_window":
                    req = DrivingWSRequest(**raw)
                    res = stream_service.handle_driving(req)

                else:
                    raise ValueError("invalid type")

            except ValidationError as e:
                await websocket.send_json({
                    "type": msg_type,
                    "success": False,
                    "message": str(e),
                    "data": None
                })
                continue

            except Exception as e:
                await websocket.send_json({
                    "type": msg_type,
                    "success": False,
                    "message": str(e),
                    "data": None
                })
                continue

            await websocket.send_json(res.model_dump(by_alias=True))

    except WebSocketDisconnect:
        print("WebSocket disconnected")