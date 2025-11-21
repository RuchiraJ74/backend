from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import openai
import base64
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    response = openai.audio.transcriptions.create(
        model="gpt-4o-mini-tts",
        file=file.file
    )
    return {"text": response.text}

@app.post("/summarize")
async def summarize_meeting(data: dict):
    transcript = data.get("transcript", "")

    prompt = f"""
    Summarize this meeting in clear bullet points:
    {transcript}
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"summary": response.choices[0].message["content"]}

@app.get("/")
def home():
    return {"message": "Backend Running Successfully!"}
