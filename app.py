from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from diffusers import BitsAndBytesConfig, StableDiffusion3Pipeline, SD3Transformer2DModel
import torch
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from functools import lru_cache
import os
import asyncio

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY", "your-secret-key-here")

# Initialize FastAPI app
app = FastAPI(title="Stable Diffusion API")

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# API key security
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

# Model configuration
model_id = "stabilityai/stable-diffusion-3.5-large"
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load pipeline once per worker
@lru_cache()
def get_pipeline():
    print("Loading model...")
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
    return pipeline

# Request and response schemas
class GenerateRequest(BaseModel):
    prompt: str
    num_steps: Optional[int] = 60
    guidance_scale: Optional[float] = 7.0

class GenerateResponse(BaseModel):
    image: str
    status: str

# Image generation logic
def generate_image(prompt: str, num_steps: int = 60, guidance_scale: float = 7.0) -> Image.Image:
    pipe = get_pipeline()
    print(f"Generating image with prompt: {prompt}")
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=512,
    ).images[0]
    print("Image generated successfully!")
    return image

async def async_generate_image(prompt: str, num_steps: int = 60, guidance_scale: float = 7.0) -> Image.Image:
    return await asyncio.to_thread(generate_image, prompt, num_steps, guidance_scale)

# Convert image to base64
def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# API Endpoint to generate image
@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        image = await async_generate_image(
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

# CORS Preflight
@app.options("/api/generate")
async def generate_options():
    return {"message": "OK"}

# Health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# Uvicorn server entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7861, workers=4, loop="uvloop")
