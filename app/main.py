from fastapi import FastAPI
from app.dependencies import setup_logger
from app.dependencies import config
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.preprocessing.modules.models import models_initialization
from app.common.api.effects_api_gateway import effects_api
from app.common.api.townsnet_api_gateway import townsnet_api
from app.common.api.urbandb_api_gateway import urban_db_api
from app.common.api.values_api_gateway import values_api
from app.risk_calculation.social_risk_controller import calculation_router
from app.preprocessing.preprocessing_controller import (
    territories_router,
    groups_router,
    messages_router,
    named_objects_router,
    indicators_router,
    services_router
)
from app.common.system_controller import system_router
from fastapi.responses import JSONResponse
from fastapi import Request
from starlette.responses import RedirectResponse
from loguru import logger
from iduconfig import Config

setup_logger(config)

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

    await app.state.urban_db_api.close()
    await app.state.effects_api.close()
    await app.state.townsnet_api.close()
    await app.state.values_api.close()

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

def include_routers(app: FastAPI, config: Config) -> None:
    prefix = config.get("FASTAPI_PREFIX")
    app.include_router(system_router, prefix=prefix, tags=["System"])
    app.include_router(calculation_router, prefix=prefix, tags=["Analysis"])
    app.include_router(territories_router, prefix=prefix, tags=["Territories"])
    app.include_router(groups_router, prefix=prefix, tags=["Groups"])
    app.include_router(messages_router, prefix=prefix, tags=["Messages"])
    app.include_router(named_objects_router, prefix=prefix, tags=["Named objects"])
    app.include_router(indicators_router, prefix=prefix, tags=["Indicators"])
    app.include_router(services_router, prefix=prefix, tags=["Services"])

include_routers(app, config)

app.mount(
    "/gpu_access",
    StaticFiles(directory="app/gpu_access"), 
    name="gpu_access"
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error occurred")
    error_message = str(exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"error: {error_message}"}
    )

@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")