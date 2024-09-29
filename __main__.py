import uvicorn

from api.base import app

if __name__ == "__main__":
    uvicorn.run("api.base:app", reload=True, host='176.123.162.198')
