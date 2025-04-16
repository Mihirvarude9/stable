from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from diffusers import BitsAndBytesConfig, StableDiffusion3Pipeline, SD3Transformer2DModel
from PIL import Image
from io import BytesIO
import base64
import torch
import asyncio
import os

# Load environment variables
load_dotenv()

# Globals
app = FastAPI(title="SD3.5 Concurrent API")
pipeline = None
semaphore = asyncio.Semaphore(15)  # Allow 15 simultaneous requests

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key
API_KEY = os.getenv("API_KEY", "your-secret-key")
api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(request: Request, authorization: str = Header(None)):
    if request.method == "OPTIONS":
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key format")
    if authorization.replace("Bearer ", "") != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return authorization

# Request & Response Schema
class GenerateRequest(BaseModel):
    prompt: str
    num_steps: Optional[int] = 28
    guidance_scale: Optional[float] = 7.0

class GenerateResponse(BaseModel):
    image: str
    status: str

# Convert image to base64
def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Run pipeline in separate thread (non-blocking)
def run_generation(prompt: str, steps: int, scale: float) -> Image.Image:
    with torch.inference_mode():
        output = pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=scale,
            max_sequence_length=512
        )
        return output.images[0]

# Async handler with semaphore for concurrency
async def generate_image(prompt: str, steps: int, scale: float):
    async with semaphore:
        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None, run_generation, prompt, steps, scale)
        return image

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        image = await generate_image(request.prompt, request.num_steps, request.guidance_scale)
        return GenerateResponse(
            image=image_to_base64(image),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/health")
async def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}

# Load model once at startup
@app.on_event("startup")
def load_model():
    global pipeline
    try:
        print("🚀 Loading SD3.5 Large model...")
        model_id = "stabilityai/stable-diffusion-3.5-large"

        transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            ),
            torch_dtype=torch.float16
        )

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=torch.float16
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()

        pipeline = pipe
        print("✅ Model loaded and ready.")
    except Exception as e:
        print(f"❌ Model failed to load: {e}")
        raise e
