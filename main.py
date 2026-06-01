from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="Live Fact Checker"
)

app.include_router(router)