from summarize import Summarizer
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Minor Project",
    description="A simple API to summarize text",
    version="0.1.0",  
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.get("/", response_class=PlainTextResponse)
async def home():
    return "Welcome to the home page"

@app.post("/summarize", tags=["Summarize"])
async def summarize(text: str, type: str):
    summarizer = Summarizer()
    return summarizer.get_summary(text, type)

@app.get("/getFromMongo", tags=["GetFromMongo"])
async def getFromMongo():
    summarizer = Summarizer()
    return summarizer.getFromMongo()
