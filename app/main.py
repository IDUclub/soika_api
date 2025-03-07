import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.common.config import config
from app.risk_calculation.social_risk_controller import calculation_router
from app.risk_calculation.preprocessing_controller import territories_router, groups_router, messages_router, named_objects_router, indicators_router, services_router
from app.risk_calculation.logic.preprocessing_methods import preprocessing

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:MM-DD HH:mm}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True
)

app = FastAPI(
    title='SOIKA API',
    description='Datamining, preprocessing and scoring of social risk and public outrage based on digital footprints',
    version="0.5"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(calculation_router, prefix=config.get("FASTAPI_PREFIX"), tags=['Analysis'])
app.include_router(territories_router, prefix=config.get("FASTAPI_PREFIX"), tags=['Territories'])
app.include_router(groups_router, prefix=config.get("FASTAPI_PREFIX"), tags=['Groups'])
app.include_router(messages_router, prefix=config.get("FASTAPI_PREFIX"), tags=['Messages'])
app.include_router(named_objects_router, prefix=config.get("FASTAPI_PREFIX"), tags=['Named objects'])
app.include_router(indicators_router, prefix=config.get("FASTAPI_PREFIX"), tags=['Indicators'])
app.include_router(services_router, prefix=config.get("FASTAPI_PREFIX"), tags=['Services'])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error occurred")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.on_event("startup")
async def launch_models():
    await preprocessing.init_models()
