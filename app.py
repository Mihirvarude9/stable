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

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY", "your-secret-key")

# FastAPI app
app = FastAPI(title="Stable Diffusion 3.5 Large API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Header
api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key format")
    if authorization.replace("Bearer ", "") != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return authorization

# Pydantic request/response models
class GenerateRequest(BaseModel):
    prompt: str
    num_steps: Optional[int] = 28  # MAX SUPPORTED
    guidance_scale: Optional[float] = 7.5

class GenerateResponse(BaseModel):
    image: str
    status: str

# Shared pipeline and lock
pipeline = None
semaphore = asyncio.Semaphore(15)  # Allow up to 15 parallel inferences

# Convert image to base64
def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Run inference on a separate thread
def run_inference(prompt: str, steps: int, scale: float) -> Image.Image:
    with torch.inference_mode():
        result = pipeline(prompt=prompt, num_inference_steps=steps, guidance_scale=scale)
        return result.images[0]

# Async wrapper for image generation
async def generate_image(prompt: str, steps: int, scale: float) -> Image.Image:
    async with semaphore:
        return await asyncio.to_thread(run_inference, prompt, steps, scale)

# POST endpoint
@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        # Clamp num_steps to 28 max
        steps = min(req.num_steps or 28, 28)
        if req.num_steps and req.num_steps > 28:
            print(f"⚠️ Clamped num_steps from {req.num_steps} to 28 to avoid scheduler crash.")

        image = await generate_image(req.prompt, steps, req.guidance_scale)
        return GenerateResponse(image=image_to_base64(image), status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

# Health check
@app.get("/api/health")
async def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}

# Load model once at startup
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
    print("✅ SD3.5 model loaded and ready.")
