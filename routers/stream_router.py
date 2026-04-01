from fastapi import APIRouter

router = APIRouter(
    prefix="/stream",
    tags=["stream"]
)


@router.get("/ping")
def stream_ping():
    return {"message": "stream router connected"}