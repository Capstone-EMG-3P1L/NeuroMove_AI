from fastapi import APIRouter, HTTPException
from schemas.calibration_schema import *;
from services.calibration_service import *;
from storage.calibration_store import *;

router = APIRouter(
    prefix="/ai/calibration",
    tags=["calibration"]
)


calibration_store = CalibrationSessionStore()
calibration_service = CalibrationService(calibration_store)

@router.get("/ping")
def calibration_ping():
    return {"message": "calibration router connected"}

@router.post("/start",response_model=CalibrationStartResponse)
def start_calibration(request:CalibrationStartRequest):
    try:
        return calibration_service.start_calibration(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.get("/status/{calibrationSessionId}", response_model=CalibrationStatusResponse)
def get_calibration(calibrationSessionId:str):
    try:
        return calibration_service.get_calibration_status(calibrationSessionId)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.patch("/step", response_model=CalibrationStepUpdateResponse)
def update_calibration_step(request: CalibrationStepUpdateRequest):
    try:
        return calibration_service.update_calibration_step(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/data", response_model=CalibrationDataResponse)
def append_calibration_data(request: CalibrationDataRequest):
    try:
        return calibration_service.append_calibration_data(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/finish", response_model=CalibrationFinishResponse)
def finish_calibration(request: CalibrationFinishRequest):
    try:
        return calibration_service.finish_calibration(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))