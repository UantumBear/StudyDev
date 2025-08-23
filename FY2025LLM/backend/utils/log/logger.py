# utils/logger.py
import logging
import sys
from datetime import datetime, timezone, timedelta

class KSTFormatter(logging.Formatter):
    """로그에 KST(+09:00) 타임존과 호출 위치(모듈, 함수)를 찍어주는 Formatter"""
    def formatTime(self, record, datefmt=None):
        tz = timezone(timedelta(hours=9))  # KST(+09:00)
        dt = datetime.fromtimestamp(record.created, tz)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            # ISO8601 스타일
            return dt.isoformat()

    def format(self, record):
        # 클래스/모듈명 + 함수명을 기록
        record.module_func = f"{record.module}.{record.funcName}"
        return super().format(record)


def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # 중복 핸들러 방지
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = KSTFormatter(
        fmt="[{asctime}] [{levelname:<8}] {module_func} : {message}",
        datefmt="%Y-%m-%d %H:%M:%S%z",
        style="{"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
