from fastapi import APIRouter, HTTPException
from starlette.responses import FileResponse
from iduconfig import Config


config = Config()
system_router = APIRouter()

@system_router.get("/logs")
async def get_logs():
    """
    Получить файл логов приложения
    """
    try:
        return FileResponse(
        f"{config.get('LOG_FILE')}.log",
        media_type='application/octet-stream',
        filename=f"{config.get('LOG_FILE')}.log",
        )
    except FileNotFoundError as e:
        raise HTTPException(
        status_code=404,
        msg="Log file not found",
        _input={"log_file_name": f"{config.get('LOG_FILE')}.log"},
        _detail={"error": e.str()}
        )
    except Exception as e:
        raise HTTPException(
        status_code=500,
        msg="Internal server error during reading logs",
        _input={"log_file_name": f"{config.get('LOG_FILE')}.log"},
        _detail={"error": e.str()}
        )