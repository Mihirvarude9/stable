from diffusers import BitsAndBytesConfig, StableDiffusion3Pipeline, SD3Transformer2DModel
import torch
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv
import asyncio

# Global placeholder
pipeline = None

# Load environment variables
load_dotenv()

app = FastAPI(title="Stable Diffusion API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# API key setup
API_KEY = os.getenv("API_KEY", "your-secret-key-here")
api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(request: Request, authorization: str = Header(None)):
    if request.method == "OPTIONS":
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key format")
    api_key = authorization.replace("Bearer ", "")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Data models
class GenerateRequest(BaseModel):
    prompt: str
    num_steps: Optional[int] = 28
    guidance_scale: Optional[float] = 7.0

class GenerateResponse(BaseModel):
    image: str
    status: str

# Image generation (async + thread-safe)
async def generate_image_async(prompt: str, num_steps: int = 28, guidance_scale: float = 7.0) -> Image.Image:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: pipeline(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=512
    ).images[0])

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Main API endpoint
@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        image = await generate_image_async(
            prompt=request.prompt,
            num_steps=request.num_steps,
            guidance_scale=request.guidance_scale
        )
        return GenerateResponse(
            image=image_to_base64(image),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}

# ✅ Model loading (once at startup)
@app.on_event("startup")
def load_model():
    global pipeline
    print("Loading model...")
    try:
        model_id = "stabilityai/stable-diffusion-3.5-large"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.float16
        )
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=model_nf4,
            torch_dtype=torch.float16
        )
        pipeline.enable_model_cpu_offload()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e
