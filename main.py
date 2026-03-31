from fastapi import FastAPI

app = FastAPI(
    title="NeuroMove AI Server",
    description="EMG-based intent inference AI server",
    version="0.1.0"
)


@app.get("/health")
def health_check():
    return {"status": "ok"}