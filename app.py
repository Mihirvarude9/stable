from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from queue import Queue

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY", "your-secret-key")

# CORS setup for Vercel frontend
origins = [
    "https://sd-deploy-wgx123.vercel.app/",  # Replace with your actual deployed domain
    "http://localhost:3000"
]

# Initialize FastAPI
app = FastAPI(title="SD3.5 Parallel Pool")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
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

# API Key Middleware
api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key format")
    if authorization.replace("Bearer ", "") != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return authorization

# Base64 Image Conversion
def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Global pipeline queue
pipeline_queue: Queue = Queue()

# Inference logic
def run_inference(prompt: str, steps: int, scale: float) -> Image.Image:
    pipe = pipeline_queue.get()
    try:
        with torch.inference_mode():
            image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=scale).images[0]
        return image
    finally:
        pipeline_queue.put(pipe)

async def generate_image(prompt: str, steps: int, scale: float):
    return await asyncio.to_thread(run_inference, prompt, steps, scale)

# Preflight OPTIONS handler
@app.options("/api/generate")
async def preflight_handler():
    return JSONResponse(status_code=200, content={"message": "CORS preflight successful"})

# Health check
@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "pipeline_pool_size": pipeline_queue.qsize()
    }

# Main inference endpoint
@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        steps = min(req.num_steps or 28, 28)
        image = await generate_image(req.prompt, steps, req.guidance_scale)
        return GenerateResponse(image=image_to_base64(image), status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Load SD3.5 pipelines on startup
@app.on_event("startup")
def load_multiple_pipelines():
    global pipeline_queue
    num_pipelines = 1  # You can increase if VRAM allows

    print(f"🚀 Loading {num_pipelines} instances of SD3.5 for concurrent processing...")

    model_id = "stabilityai/stable-diffusion-3.5-large"

    for i in range(num_pipelines):
        print(f"🔧 Loading pipeline {i + 1}/{num_pipelines}...")

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
        pipeline_queue.put(pipe)

    print("✅ All SD3.5 pipelines are ready.")