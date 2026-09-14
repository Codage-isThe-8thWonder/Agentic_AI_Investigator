from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from backend.api.routes import router


app = FastAPI(
    title="Agentic RAG Investigation",
    description=(
        "Interactive Agentic RAG "
        "Investigation system using "
        "hybrid retrieval, Investigator "
        "and Fact Checker agents, "
        "and an evidence graph."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "*"
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# ROUTER
# ============================================================

app.include_router(
    router,
    prefix="/api",
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": (
            "Agentic RAG Investigation"
        ),
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "api_prefix": "/api",
    }