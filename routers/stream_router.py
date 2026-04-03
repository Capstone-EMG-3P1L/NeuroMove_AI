from fastapi import APIRouter,HTTPException
from schemas.stream_schema import *;

router = APIRouter(
    prefix="ai/stream",
    tags=["stream"]
)


@router.get("/ping")
def stream_ping():
    return {"message": "stream router connected"}

@router.post("/emg", response_model=StreamAckResponseSchema)
def receive_stream(request: StreamRequestSchema):
    try:
        request.validate_window_matches_samples()

        return StreamAckResponseSchema(
            status="success",
            message="EMG stream received successfully"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))