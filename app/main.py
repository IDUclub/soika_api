from fastapi import FastAPI
from app.dependencies import setup_logger, include_routers
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.preprocessing.modules.models import models_initialization
from app.common.api.effects_api_gateway import effects_api
from app.common.api.townsnet_api_gateway import townsnet_api
from app.common.api.urbandb_api_gateway import urban_db_api
from app.common.api.values_api_gateway import values_api
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

    await urban_db_api.init() 
    app.state.urban_db_api = urban_db_api

    await effects_api.init() 
    app.state.effects_api = effects_api

    await townsnet_api.init() 
    app.state.townsnet_api = townsnet_api

    await values_api.init() 
    app.state.values_api = values_api

    yield

    await app.state.urban_db_api.session.close()
    await app.state.effects_api.session.close()
    await app.state.townsnet_api.session.close()
    await app.state.values_api.session.close()

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

app.mount(
    "/gpu_access",
    StaticFiles(directory="app/gpu_access"), 
    name="gpu_access"
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error occurred")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")