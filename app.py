from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from PIL import Image
from io import BytesIO
import base64
import torch
import asyncio
import os

# ✅ Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY", "your-secret-key")

# ✅ Initialize FastAPI
app = FastAPI(title="SD3.5 L40s Image Generator")

# ✅ CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sd-deploy-wgx123.vercel.app",  # Deployed frontend
        "http://localhost:3000",                # Local frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request & Response Schemas
class GenerateRequest(BaseModel):
    prompt: str
    num_steps: Optional[int] = 28
    guidance_scale: Optional[float] = 7.5

class GenerateResponse(BaseModel):
    image: str
    status: str

# ✅ API Key verification
api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(request: Request, authorization: str = Header(None)):
    if request.method == "OPTIONS":
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key format")
    if authorization.replace("Bearer ", "") != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return authorization

# ✅ Utility: Convert image to base64
def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ✅ Global pool for concurrency
pipeline_pool: List[StableDiffusion3Pipeline] = []
pipeline_semaphore: asyncio.Semaphore = None

# ✅ Sync inference logic
def run_inference_with_pipeline(pipe, prompt, steps, scale):
    with torch.inference_mode():
        return pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=scale).images[0]

# ✅ Async inference wrapper
async def generate_image(prompt: str, steps: int, scale: float):
    await pipeline_semaphore.acquire()
    try:
        pipe = pipeline_pool.pop()
        image = await asyncio.to_thread(run_inference_with_pipeline, pipe, prompt, steps, scale)
        return image
    finally:
        pipeline_pool.append(pipe)
        pipeline_semaphore.release()

# ✅ Preflight support
@app.options("/api/generate")
async def preflight_handler():
    return JSONResponse(status_code=200, content={"message": "Preflight OK"})

# ✅ Health check endpoint
@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "available_slots": pipeline_semaphore._value,
        "total_pipelines": len(pipeline_pool)
    }

# ✅ Main image generation endpoint
@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        steps = min(req.num_steps or 28, 28)
        image = await generate_image(req.prompt, steps, req.guidance_scale)
        return GenerateResponse(image=image_to_base64(image), status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ✅ Pipeline loader on startup
@app.on_event("startup")
def load_multiple_pipelines():
    global pipeline_pool, pipeline_semaphore

    num_pipelines = 2  # Adjust based on L40s VRAM
    pipeline_pool = []
    pipeline_semaphore = asyncio.Semaphore(num_pipelines)

    print(f"🚀 Loading {num_pipelines} SD3.5 pipelines...")
    model_id = "stabilityai/stable-diffusion-3.5-large"

    for i in range(num_pipelines):
        print(f"🔧 Initializing pipeline {i+1}/{num_pipelines}...")

        # ✅ Load transformer without quantization
        transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.float16
        )

        # ✅ Load main SD3.5 pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=torch.float16
        )

        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipeline_pool.append(pipe)

    print("✅ All pipelines loaded successfully.")
