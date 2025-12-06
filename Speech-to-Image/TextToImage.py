from monsterapi import client
import requests
from PIL import Image


api_key="your_api_key_here"
monster_client = client.MonsterClient(api_key)

model='txt2img'
input_data={
    'prompt':'detailed sketch of lion by greg rutkowski',
    'negative_prompt':'lowres, bad anatomy, error body, error hands, error fingers, error legs, error feet, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, ugly, duplicate',
    'samples':1,
    'steps':50,
    'aspect_ratio':'16:9',
    'guidance_scale':7.5,
    'seed':2414
}

print("Generating image...")
result=monster_client.generate(model=model, input_data=input_data)
img_urls = result['output'][0]

file_name='generated_image.png'
response = requests.get(img_urls)
if response.status_code == 200:
    with open(file_name, 'wb') as f:
        f.write(response.content)
    print(f"Image saved as {file_name}")
    img = Image.open(file_name)
    img.show()
else:
    print("Failed to retrieve image from URL.")