from fastapi import APIRouter,HTTPException,WebSocket,WebSocketDisconnect
from schemas.stream_schema import *;
from services.signal_processing_service import SignalProcessingService
from services.feature_service import FeatureService
from services.inference_service import InferenceService

router = APIRouter(
    prefix="/ai/stream",
    tags=["stream"]
)

signal_processing_service = SignalProcessingService()
feature_service = FeatureService()
inference_service = InferenceService()


@router.get("/ping")
def stream_ping():
    return {"message": "stream router connected"}

@router.websocket("/ws")
async def receive_stream(websocket:WebSocket):
    await websocket.accept()
    print("EMG websocket connected")
    
    try:
        while True:
            raw_data = await websocket.receive_json()

            try:
                request = StreamRequestSchema(**raw_data)
            except Exception as e:
                await websocket.send_json({
                    "success": False,
                    "message": f"invalid request: {str(e)}",
                    "data": None
                })
                continue

            # 1. 전처리 -> 서비스명 향후 수정 예정
            processed_channels = signal_processing_service.process(request.channels)

            # 2. feature 추출 -> 서비스명 향후 수정 예정
            feature_vector = feature_service.extract_features(processed_channels)

            # 3. 추론 -> 서비스명 향후 수정 예정
            inference_result = inference_service.predict(feature_vector)

            buffered_window_count += 1

            response = StreamAckResponseSchema(
                success=True,
                message="실시간 EMG 데이터가 처리되었습니다.",
                data={
                    "sessionId": request.session_id,
                    "deviceId": "emg-esp32-A12F",
                    "acceptedSequenceNumber": request.sequence_number,
                    "bufferedWindowCount": buffered_window_count,
                    "inferenceTriggered": True
                }
            )

            await websocket.send_json(response.model_dump())

    except WebSocketDisconnect:
        print("EMG websocket disconnected")