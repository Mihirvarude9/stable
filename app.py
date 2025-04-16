from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, BitsAndBytesConfig
from PIL import Image
from io import BytesIO
import base64
import torch
import asyncio
import os

# Load env
load_dotenv()
API_KEY = os.getenv("API_KEY", "your-secret-key")
model_id = "stabilityai/stable-diffusion-3.5-large"

# Globals
app = FastAPI(title="SD3.5 Concurrent API")
base_components = {}
semaphore = asyncio.Semaphore(15)  # limit 15 concurrent jobs

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth
api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(request: Request, authorization: str = Header(None)):
    if request.method == "OPTIONS":
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key format")
    if authorization.replace("Bearer ", "") != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return authorization

# Models
class GenerateRequest(BaseModel):
    prompt: str
    num_steps: Optional[int] = 28
    guidance_scale: Optional[float] = 7.0

class GenerateResponse(BaseModel):
    image: str
    status: str

# Utilities
def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def generate_image(prompt: str, steps: int, scale: float) -> Image.Image:
    pipe = StableDiffusion3Pipeline(
        vae=base_components["vae"],
        text_encoder=base_components["text_encoder"],
        tokenizer=base_components["tokenizer"],
        unet=base_components["unet"],
        scheduler=base_components["scheduler"],
        feature_extractor=base_components["feature_extractor"],
        requires_safety_checker=False,
        safety_checker=None
    )
    pipe.to("cuda", torch_dtype=torch.float16)
    with torch.inference_mode():
        output = pipe(prompt, num_inference_steps=steps, guidance_scale=scale)
    return output.images[0]

async def async_generate_image(prompt: str, steps: int, scale: float) -> Image.Image:
    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, generate_image, prompt, steps, scale)

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        image = await async_generate_image(
            prompt=request.prompt,
            steps=request.num_steps,
            scale=request.guidance_scale
        )
        return GenerateResponse(image=image_to_base64(image), status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.get("/api/health")
async def health():
    return {"status": "healthy", "model_loaded": bool(base_components)}

# Model loading
@app.on_event("startup")
def load_model():
    global base_components
    print("🚀 Loading Stable Diffusion 3.5 Large...")

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

    # Store components
    base_components = {
        "vae": pipe.vae,
        "text_encoder": pipe.text_encoder,
        "tokenizer": pipe.tokenizer,
        "unet": pipe.unet,
        "scheduler": pipe.scheduler,
        "feature_extractor": pipe.feature_extractor
    }
    print("✅ Model components loaded and ready.")
