import threading
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
from PIL import Image
from io import BytesIO
import base64
import torch
import asyncio
import os

# Load env
load_dotenv()
API_KEY = os.getenv("API_KEY", "your-secret-key")

# App setup
app = FastAPI(title="SD3.5 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class GenerateRequest(BaseModel):
    prompt: str
    num_steps: Optional[int] = 28
    guidance_scale: Optional[float] = 7.5

class GenerateResponse(BaseModel):
    image: str
    status: str

# Auth
api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key format")
    if authorization.replace("Bearer ", "") != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return authorization

# Shared resources
pipeline = None
semaphore = asyncio.Semaphore(15)  # Limit concurrency
pipeline_lock = threading.Lock()   # Protect shared pipeline object

# Convert image to base64
def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Inference in thread-safe block
def run_inference(prompt: str, steps: int, scale: float) -> Image.Image:
    with pipeline_lock:
        with torch.inference_mode():
            result = pipeline(prompt=prompt, num_inference_steps=steps, guidance_scale=scale)
            return result.images[0]

# Async inference
async def generate_image(prompt: str, steps: int, scale: float):
    async with semaphore:
        return await asyncio.to_thread(run_inference, prompt, steps, scale)

@app.get("/api/health")
async def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        steps = min(req.num_steps or 28, 28)
        if req.num_steps and req.num_steps > 28:
            print(f"⚠️ Clamped num_steps from {req.num_steps} to 28")

        image = await generate_image(req.prompt, steps, req.guidance_scale)
        return GenerateResponse(image=image_to_base64(image), status="success")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

# Load model once
@app.on_event("startup")
def load_model():
    global pipeline
    print("🚀 Loading Stable Diffusion 3.5 Large...")

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

    print("✅ SD3.5 Model loaded and ready.")
