from summarize import Summarizer
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse


app = FastAPI(
    title="Minor Project",
    description="A simple API to summarize text",
    version="0.1.0",  
)

app.get("/", response_class=PlainTextResponse)
async def home():
    return "Welcome to the home page"

@app.post("/summarize", tags=["Summarize"])
async def summarize(text: str):
    summarizer = Summarizer()
    return summarizer.get_summary(text)

@app.get("/getFromMongo", tags=["GetFromMongo"])
async def getFromMongo():
    summarizer = Summarizer()
    return summarizer.getFromMongo()
