import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes.api import router


LOGGER = logging.getLogger(__name__)
app = FastAPI(
    title="UdyamPulse AI API",
    version="1.0.0",
    description="REST API for the existing UdyamPulse financial analytics and AI services.",
)

origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, error: Exception):
    LOGGER.exception("Unhandled API error for %s %s.", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )


app.include_router(router)
