# main.py
from fastapi import FastAPI
from qRootdataAPI import router as qRootdataAPI_router
from qNewsAPI import router as qNewsAPI_router

app = FastAPI()

app.include_router(qRootdataAPI_router, prefix="/crypto ")
app.include_router(qNewsAPI_router, prefix="/cryoto")
