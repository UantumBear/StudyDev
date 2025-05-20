@asynccontextmanager
async def lifespan(app: FastAPI):
    await on_startup()
    yield
    await on_shutdown()

import fastapi
app = FastAPI(lifespan=lifespan)