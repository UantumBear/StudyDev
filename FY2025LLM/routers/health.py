from fastapi import APIRouter

HEALTH = APIRouter(prefix="/health", tags=["HEALTH"])

@HEALTH.get("/hello")
async def hello_adm1000():
    return {"message": "Hello from HEALTH!"}