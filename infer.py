import torch
from controlnet_flux import FluxControlNetModel
from pipeline_flux_controlnet import FluxControlNetPipeline

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import re
import os

def contains_chinese(text):
    if re.search(r'[\u4e00-\u9fff]', text):
        return True
    return False

def canny(img):
    low_threshold = 50
    high_threshold = 100
    img = cv2.Canny(img, low_threshold, high_threshold)
    img = img[:, :, None]
    img = 255 - np.concatenate([img, img, img], axis=2)
    return img


if __name__ == "__main__":
    
    base_model = "black-forest-labs/FLUX.1-dev"
    controlnet_model = "Shakker-Labs/RepText"

    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
    pipe = FluxControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
    ).to("cuda")

    ## set resolution
    width, height = 1024, 1024

    ## set font
    font_path = "./assets/Arial_Unicode.ttf" # use your own font
    font_size = 80 # it is recommended to use a font size >= 60
    font = ImageFont.truetype(font_path, font_size)

    ## set text content, position, color
    text_list = ["哩布哩布"]
    text_position_list = [(370, 200)]
    text_color_list = [(255, 255, 255)]

    # text_list = ["Shakker Labs"]
    # text_position_list = [(270, 300)]
    # text_color_list = [(255, 255, 255)]

    # text_list = ["Lovart AI", "Always Day 1"]
    # text_position_list = [(470, 300), (470, 400)]
    # text_color_list = [(255, 255, 255), (255, 255, 255)]

    # text_list = ["以往不谏", "来者可追"]
    # text_position_list = [(200, 200), (200, 300)]
    # text_color_list = [(255, 255, 255), (255, 255, 255)]

    # text_list = ["Shakker Labs", "RepText"]
    # text_position_list = [(200, 200), (200, 300)]
    # text_color_list = [(255, 255, 255), (255, 255, 255)]

    ## set controlnet conditions
    control_image_list = [] # canny list
    control_position_list = [] # position list
    control_mask_list = [] # regional mask list
    control_glyph_all = np.zeros([height, width, 3], dtype=np.uint8) # all glyphs
    
    ## handle each line of text
    for text, text_position, text_color in zip(text_list, text_position_list, text_color_list):

        ### glyph image, render text to black background
        control_image_glyph = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(control_image_glyph)
        draw.text(text_position, text, font=font, fill=text_color)

        ### get bbox
        bbox = draw.textbbox(text_position, text, font=font)

        ### position condition
        control_position = np.zeros([height, width], dtype=np.uint8)
        control_position[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
        control_position = Image.fromarray(control_position.astype(np.uint8))
        control_position_list.append(control_position)

        ### regional mask
        control_mask_np = np.zeros([height, width], dtype=np.uint8)
        control_mask_np[bbox[1]-5:bbox[3]+5, bbox[0]-5:bbox[2]+5] = 255
        control_mask = Image.fromarray(control_mask_np.astype(np.uint8))
        control_mask_list.append(control_mask)

        ### accumulate glyph
        control_glyph = np.array(control_image_glyph)
        control_glyph_all += control_glyph

        ### canny condition
        control_image = canny(cv2.cvtColor(np.array(control_image_glyph), cv2.COLOR_RGB2BGR))
        control_image = Image.fromarray(cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB))
        control_image_list.append(control_image)
        
    control_glyph_all = Image.fromarray(control_glyph_all.astype(np.uint8))
    control_glyph_all = control_glyph_all.convert("RGB")
    # control_glyph_all.save("./results/control_glyph.jpg")

    # it is recommended to use words such 'sign', 'billboard', 'banner' in your prompt
    # for Englith text, it helps if you add the text to the prompt
    prompt = "a street sign in city"
    for text in text_list:
        if not contains_chinese(text):
            prompt += f", '{text}'"
    prompt += ", filmfotos, film grain, reversal film photography" # optional
    print(prompt)

    generator = torch.Generator(device="cuda").manual_seed(42)

    image = pipe(
        prompt,
        control_image=control_image_list, # canny
        control_position=control_position_list, # position
        control_mask=control_mask_list, # regional mask
        control_glyph=control_glyph_all, # as init latent, optional, set to None if not used
        controlnet_conditioning_scale=1.0,
        controlnet_conditioning_step=30,
        width=width,
        height=height,
        num_inference_steps=30,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]

    if not os.path.exists("./results"):
        os.makedirs("./results")
    image.save(f"./results/result.jpg")