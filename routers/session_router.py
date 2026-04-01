from fastapi import APIRouter

router = APIRouter(
    prefix="/sessions",
    tags=["sessions"]
)


@router.get("/ping")
def session_ping():
    return {"message": "session router connected"}