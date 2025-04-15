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

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Stable Diffusion API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins temporarily for testing
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    max_age=3600,  # Cache preflight requests for 1 hour
)

# API Key security
API_KEY = os.getenv("API_KEY", "your-secret-key-here")
api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(request: Request, authorization: str = Header(None)):
    # Skip validation for OPTIONS requests
    if request.method == "OPTIONS":
        return None
        
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key format")
    api_key = authorization.replace("Bearer ", "")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Model loading
model_id = "stabilityai/stable-diffusion-3.5-large"

# Ensure quantization uses the correct data type
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model...")
try:
    # Load the Transformer model with quantization
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.float16
    )

    # Load Stable Diffusion pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id, 
        transformer=model_nf4,
        torch_dtype=torch.float16
    )
    pipeline.enable_model_cpu_offload()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e

# Pydantic models for request/response
class GenerateRequest(BaseModel):
    prompt: str
    num_steps: Optional[int] = 60
    guidance_scale: Optional[float] = 7.0

class GenerateResponse(BaseModel):
    image: str  # Base64 encoded image
    status: str

def generate_image(prompt: str, num_steps: int = 60, guidance_scale: float = 7.0) -> Image.Image:
    try:
        print(f"Generating image with prompt: {prompt}")
        # Generate image
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
        ).images[0]
        print("Image generated successfully!")
        return image
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        image = generate_image(
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

@app.options("/api/generate")
async def generate_options():
    return {"message": "OK"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861)
