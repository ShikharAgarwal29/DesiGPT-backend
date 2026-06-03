from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="DesiGPT API", version="2.0.0")

# CORS — allows GitHub Pages to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are DesiGPT, a witty and intelligent AI assistant built by ShikharAgarwal29.
Your personality:
- You mix Hindi and English (Hinglish) naturally in every response
- You are funny, confident and very desi 🇮🇳
- You use emojis naturally
- You are actually intelligent and can answer ANY question accurately
- You add desi flavor to everything — chai, cricket, Bollywood references
- When asked who made you: "ShikharAgarwal29 ne banaya! Ekdum desi AI! 😎"
- Keep responses concise but informative
- Never break character"""

# Request/Response models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []

class ChatResponse(BaseModel):
    reply: str
    tokens_used: int

@app.get("/")
async def root():
    return {"status": "DesiGPT API is live 🚀", "version": "2.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Build conversation with history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Append conversation history (last 10 messages for context)
        for msg in request.history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Append current message
        messages.append({"role": "user", "content": request.message})

        # Call Groq API with Llama 3
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.8,
            max_tokens=500,
        )

        reply = completion.choices[0].message.content
        tokens_used = completion.usage.total_tokens

        return ChatResponse(reply=reply, tokens_used=tokens_used)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}