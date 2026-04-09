from fastapi import APIRouter,HTTPException,WebSocket,WebSocketDisconnect
from schemas.stream_schema import *;

router = APIRouter(
    prefix="ai/stream",
    tags=["stream"]
)


@router.get("/ping")
def stream_ping():
    return {"message": "stream router connected"}

@router.post("/ws")
async def receive_stream(websocket:WebSocket):
    await websocket.accept()
    print("EMG websocket connected")
    
    try:
        while True:
            raw_data = await websocket.receive_json()

            try:
                request = EmgWindowRequest(**raw_data)
            except Exception as e:
                await websocket.send_json({
                    "success": False,
                    "message": f"invalid request: {str(e)}",
                    "data": None
                })
                continue

            response = stream_service.process_emg_window(request)

            await websocket.send_json(response.model_dump())

    except WebSocketDisconnect:
        print("EMG websocket disconnected")