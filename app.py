from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from diffusers import BitsAndBytesConfig, StableDiffusion3Pipeline, SD3Transformer2DModel
import torch
from PIL import Image
from io import BytesIO
import base64
import asyncio
import os

load_dotenv()
app = FastAPI(title="Stable Diffusion 3.5 Concurrent API")
base_pipeline = None
API_KEY = os.getenv("API_KEY", "your-secret-key")
api_key_header = APIKeyHeader(name="Authorization")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class GenerateRequest(BaseModel):
    prompt: str
    num_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5

class GenerateResponse(BaseModel):
    image: str
    status: str

# API Key Verification
async def verify_api_key(request: Request, authorization: str = Header(None)):
    if request.method == "OPTIONS":
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key format")
    if authorization.replace("Bearer ", "") != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return authorization

# Convert PIL image to base64
def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# Clone pipeline per request to avoid blocking
def generate_image(prompt: str, steps: int, scale: float):
    local_pipe = StableDiffusion3Pipeline(
        vae=base_pipeline.vae,
        text_encoder=base_pipeline.text_encoder,
        tokenizer=base_pipeline.tokenizer,
        unet=base_pipeline.unet,
        scheduler=base_pipeline.scheduler,
        feature_extractor=base_pipeline.feature_extractor,
        safety_checker=None,
        requires_safety_checker=False
    )
    local_pipe.to("cuda")
    with torch.inference_mode():
        image = local_pipe(prompt, num_inference_steps=steps, guidance_scale=scale).images[0]
        return image

# Async API endpoint
@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(
            None, generate_image, request.prompt, request.num_steps, request.guidance_scale
        )
        return GenerateResponse(image=image_to_base64(image), status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.get("/api/health")
async def health():
    return {"status": "healthy", "model_loaded": base_pipeline is not None}

# Load model on startup
@app.on_event("startup")
def load_model():
    global base_pipeline
    print("🚀 Loading base model...")

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

    base_pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch.float16
    )
    base_pipeline.to("cuda")
    base_pipeline.enable_model_cpu_offload()

    print("✅ Base pipeline loaded successfully.")
