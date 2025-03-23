from fastapi import HTTPException

def http_exception(status_code: int, msg: str, input_data=None, detail=None) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"msg": msg, "input": input_data, "detail": detail})
