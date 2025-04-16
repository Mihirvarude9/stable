from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
import torch
from PIL import Image
from io import BytesIO
import base64
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

API_KEY = os.getenv("API_KEY", "your-secret-key")

# Globals
app = FastAPI(title="SD3.5 Concurrent API")
pipeline = None
executor = ThreadPoolExecutor(max_workers=10)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Auth
api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key format")
    if authorization.replace("Bearer ", "") != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return authorization

# Request & Response Models
class GenerateRequest(BaseModel):
    prompt: str
    num_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5

class GenerateResponse(BaseModel):
    image: str
    status: str

# Convert image to base64
def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Synchronous generation
def generate_image(prompt: str, steps: int, scale: float) -> Image.Image:
    with torch.inference_mode():
        result = pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=scale
        )
        return result.images[0]

# Async generation endpoint
@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    loop = asyncio.get_running_loop()
    image = await loop.run_in_executor(executor, generate_image, req.prompt, req.num_steps, req.guidance_scale)
    return GenerateResponse(image=image_to_base64(image), status="success")

@app.get("/api/health")
async def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}

# Load model once
@app.on_event("startup")
def load_model():
    global pipeline
    print("🚀 Loading SD3.5 Large model...")
    model_id = "stabilityai/stable-diffusion-3.5-large"

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    transformer = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
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
    print("✅ SD3.5 model loaded.")
