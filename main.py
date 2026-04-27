import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers.session_router import router as session_router
from routers.calibration_router import router as calibration_router
from routers.stream_router import router as stream_router
from storage import device_mode_registry


# -----------------------
# Device lock 누수 방어 sweeper 설정
# -----------------------
# EMG 스트림이 이 시간(ms) 이상 끊긴 device 의 lock 을 강제 해제.
# 너무 짧으면 잠깐 멈췄다가 재개할 때 lock 풀림 → 세션 끊김.
# 너무 길면 진짜 누수 회복까지 오래 걸림.
# 100Hz 이상 EMG 스트림 기준 30 초 비활성은 명백한 비정상.
DEVICE_IDLE_THRESHOLD_MS = 30_000

# sweeper 가 검사하는 주기(초).
DEVICE_SWEEP_INTERVAL_S = 5


async def _device_sweeper() -> None:
    """
    백그라운드에서 주기적으로 stale device lock 을 청소.
    FastAPI lifespan 으로 등록되어 앱 시작 시 자동 실행, 종료 시 cancel 된다.
    """
    while True:
        try:
            await asyncio.sleep(DEVICE_SWEEP_INTERVAL_S)
            cleared = device_mode_registry.sweep_stale(DEVICE_IDLE_THRESHOLD_MS)
            if cleared:
                print(f"[device-sweeper] cleared stale devices: {cleared}")
        except asyncio.CancelledError:
            # 정상 종료 신호
            raise
        except Exception as e:
            # sweeper 가 죽으면 lock 누수 방어가 사라지므로 로그만 남기고 계속 돈다.
            print(f"[device-sweeper] error (continuing): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    sweeper_task = asyncio.create_task(_device_sweeper())
    try:
        yield
    finally:
        sweeper_task.cancel()
        try:
            await sweeper_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="NeuroMove AI Server",
    description="EMG-based intent inference AI server",
    version="0.1.0",
    lifespan=lifespan,
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
