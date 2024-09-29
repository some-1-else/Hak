from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_sqlalchemy import DBSessionMiddleware
from .routes import routes

app = FastAPI(
    title="QMonitoring",
    description="An API for AI, detecting video duplicates"
    # Отключаем нелокальную документацию
)

app.add_middleware(
    DBSessionMiddleware,
    db_url=str("postgresql://postgres:12345@localhost:5432/hakaton"),
    engine_args={"pool_pre_ping": True, "isolation_level": "AUTOCOMMIT"}
)

app.include_router(routes, prefix="/check")
