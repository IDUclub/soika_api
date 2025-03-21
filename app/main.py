from fastapi import FastAPI
from app.dependencies import setup_logger, include_routers
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.preprocessing.modules.models import models_initialization
from fastapi.responses import JSONResponse
from fastapi import Request
from starlette.responses import RedirectResponse
from loguru import logger
from iduconfig import Config

config = Config()
setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await models_initialization.init_models()
    yield

app = FastAPI(
    title="SOIKA API",
    description="Datamining, preprocessing and scoring of social risk and public outrage based on digital footprints",
    version="0.5",
    lifespan=lifespan
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

include_routers(app, config)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error occurred")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")