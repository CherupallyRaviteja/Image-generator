from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load Stable Diffusion model from Hugging Face
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
def generate_image():
    prompt = input("prompt: ")
    if prompt.lower() in ["exit", "stop"]:
        print("Goodbye 👋")
        return
    image = pipe(prompt).images[0]
    image.save("generated_image.png")
    display(image)
while True:
    generate_image()