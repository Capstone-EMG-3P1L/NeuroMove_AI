from fastapi import APIRouter

router = APIRouter(
    prefix="/calibration",
    tags=["calibration"]
)


@router.get("/ping")
def calibration_ping():
    return {"message": "calibration router connected"}