from fastapi import APIRouter, HTTPException
from schemas.session_schema import *
from services.session_service import *
from storage.session_store import *

router = APIRouter(
    prefix="/sessions",
    tags=["sessions"]
)

session_store = SessionStore()
session_service = SessionService(session_store)


@router.get("/ping")
def session_ping():
    return {"message": "session router connected"}


@router.post("/start", response_model=SessionStartResponse)
def start_session(request: SessionStartRequest):
    try:
        return session_service.start_session(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{sessionId}", response_model=SessionStatusResponse)
def get_session_status(sessionId: str):
    try:
        return session_service.get_session_status(sessionId)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/{sessionId}/end", response_model=SessionEndResponse)
def end_session(sessionId: str):
    try:
        request = SessionEndRequest(sessionId=sessionId)
        return session_service.end_session(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))