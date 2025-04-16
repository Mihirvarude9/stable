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
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# === Global Definitions === #
app = FastAPI(title="Stable Diffusion 3.5 API")
pipeline = None
executor = ThreadPoolExecutor(max_workers=15)  # <-- Allow up to 15 threads (tune as needed)

# === CORS Middleware === #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === API Key === #
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

# === Request & Response Schema === #
class GenerateRequest(BaseModel):
    prompt: str
    num_steps: Optional[int] = 28
    guidance_scale: Optional[float] = 7.0

class GenerateResponse(BaseModel):
    image: str
    status: str

# === Utility Functions === #
def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_image_sync(prompt: str, num_steps: int, guidance_scale: float) -> Image.Image:
    with torch.inference_mode():
        result = pipeline(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512
        )
        return result.images[0]

async def generate_image_async(prompt: str, num_steps: int, guidance_scale: float) -> Image.Image:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, generate_image_sync, prompt, num_steps, guidance_scale
    )

# === Routes === #
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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}

# === Load Model on Startup === #
@app.on_event("startup")
def load_model():
    global pipeline
    try:
        print("🚀 Loading Stable Diffusion 3.5 Large model...")
        model_id = "stabilityai/stable-diffusion-3.5-large"

        # Load quantized transformer
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

        # Load pipeline
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
        print(f"❌ Failed to load model: {str(e)}")
        raise e
