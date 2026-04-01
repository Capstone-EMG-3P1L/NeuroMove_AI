from fastapi import FastAPI
from routers.session_router import router as session_router
from routers.calibration_router import router as calibration_router
from routers.stream_router import router as stream_router

app = FastAPI(
    title="NeuroMove AI Server",
    description="EMG-based intent inference AI server",
    version="0.1.0"
)

app.include_router(session_router)
app.include_router(calibration_router)
app.include_router(stream_router)

@app.get("/")
def root():
    return {"message": "NeuroMove AI Server running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}