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

# Load environment
load_dotenv()
API_KEY = os.getenv("API_KEY", "your-secret-key")

app = FastAPI(title="SD3.5 Parallel Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema
class GenerateRequest(BaseModel):
    prompt: str
    num_steps: Optional[int] = 28
    guidance_scale: Optional[float] = 7.5

class GenerateResponse(BaseModel):
    image: str
    status: str

# API key auth
api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key format")
    if authorization.replace("Bearer ", "") != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return authorization

# Shared base components
base_components = {}

def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Clone pipeline for each request
def run_inference(prompt: str, steps: int, scale: float) -> Image.Image:
    pipe = StableDiffusion3Pipeline(
        vae=base_components["vae"],
        text_encoder=base_components["text_encoder"],
        tokenizer=base_components["tokenizer"],
        unet=base_components["unet"],
        scheduler=base_components["scheduler"],
        feature_extractor=base_components["feature_extractor"],
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.to("cuda")

    with torch.inference_mode():
        result = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=scale)
        return result.images[0]

# Async wrapper to trigger multiple generations in parallel
async def generate_image(prompt: str, steps: int, scale: float):
    return await asyncio.to_thread(run_inference, prompt, steps, scale)

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        steps = min(req.num_steps or 28, 28)
        image = await generate_image(req.prompt, steps, req.guidance_scale)
        return GenerateResponse(image=image_to_base64(image), status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/health")
async def health():
    return {"status": "healthy", "components_loaded": bool(base_components)}

@app.on_event("startup")
def preload_components():
    global base_components
    print("🚀 Preloading SD3.5 components for cloning...")

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

    base_components = {
        "vae": pipe.vae,
        "text_encoder": pipe.text_encoder,
        "tokenizer": pipe.tokenizer,
        "unet": pipe.unet,
        "scheduler": pipe.scheduler,
        "feature_extractor": pipe.feature_extractor
    }

    print("✅ SD3.5 components loaded for parallel use.")
