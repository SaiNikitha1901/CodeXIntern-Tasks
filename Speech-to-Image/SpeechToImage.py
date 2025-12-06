import speech_recognition as sr
from translate import Translator
import requests
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import os

# from monsterapi import client
# api_key = "your_api_key_here"
# monster_client = client.MonsterClient(api_key)

recognizer = sr.Recognizer()
translator = Translator(from_lang="te", to_lang="en")

with sr.Microphone() as source:
    print("Say something in Telugu...")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="te-IN")
        print("You said (in Telugu): " + text)

        translated_text = translator.translate(text)
        print("Translated to English: " + translated_text)

    except:
        translated_text = "a simple drawing"
        print("Error understanding audio. Using fallback prompt.")

input_data = {
    'prompt': translated_text,
    'negative_prompt': 'lowres, bad anatomy, error body, error hands, error fingers, error legs, error feet, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, ugly, duplicate',
    'samples': 1,
    'steps': 50,
    'aspect_ratio': '16:9',
    'guidance_scale': 7.5,
    'seed': 2414
}

# MonsterAPI Generation Attempt (Server was down) - YOUR ORIGINAL CODE INTACT
# model = 'txt2img'
# print("Generating image using MonsterAPI...")
# result = monster_client.generate(model=model, input_data=input_data)
# img_urls = result['output'][0]
# file_name = "generated_image.png"
# response = requests.get(img_urls)
# with open(file_name, 'wb') as f:
#     f.write(response.content)
# img = Image.open(file_name)
# img.show()

print("Generating image using Hugging Face Diffusers (FREE)...")

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("Using GPU acceleration")
else:
    print("Using CPU (slower)")

# Generate image using your input_data parameters
image = pipe(
    prompt=input_data["prompt"],
    negative_prompt=input_data["negative_prompt"],
    num_inference_steps=input_data["steps"],
    guidance_scale=input_data["guidance_scale"],
    width=1024,  # Matches your 16:9 aspect ratio
    height=576,
    generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(input_data["seed"])
).images[0]

file_name = "generated_image.png"
image.save(file_name)
image.show()
print(f"Image saved as {file_name}")
