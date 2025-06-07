import torch
from diffusers.utils import load_image
from controlnet_flux import FluxControlNetModel
from pipeline_flux_controlnet_inpaint import FluxControlNetPipeline

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

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def extract_dwpose(img, include_body=True, include_hand=True, include_face=True):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    detected_map = dwprocessor(img, include_body=include_body, include_hand=include_hand, include_face=include_face)
    detected_map = Image.fromarray(detected_map)   
    return detected_map

if __name__ == "__main__":
        
    base_model = "black-forest-labs/FLUX.1-dev"
    controlnet_model = "Shakker-Labs/RepText"
    
    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
    controlnet_inpaint = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)

    pipe = FluxControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, controlnet_inpaint=controlnet_inpaint, torch_dtype=torch.bfloat16
    ).to("cuda")

    control_image_inpaint = load_image("assets/adam-jang-8pOTAtyd_Mc-unsplash.jpg")
    control_image_inpaint = resize_img(control_image_inpaint)
    width, height = control_image_inpaint.size

    ## set font
    font_path = "./assets/Arial_Unicode.ttf" # use your own font
    font_size = 70 # it is recommended to use a font size >= 60
    font = ImageFont.truetype(font_path, font_size)

    ## set text content, position, color
    text_list = ["哩布哩布"]
    text_position_list = [(585, 375)]
    text_color_list = [(0, 255, 0)]

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
        control_position[bbox[1]-5:bbox[3]+5, bbox[0]-5:bbox[2]+5] = 255
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

    # it is recommended to use words such 'sign', 'billboard', 'banner' in your prompt
    # for Englith text, it helps if you add the text to the prompt
    prompt = "a street photo, wall"
    for text in text_list:
        if not contains_chinese(text):
            prompt += f", '{text}'"
    prompt += ", filmfotos, film grain, reversal film photography" # optional
    print(prompt)

    generator = torch.Generator(device="cuda").manual_seed(42)

    image = pipe(
        prompt,
        true_guidance_scale=3.5, # set 1.0 to disable negative guidance
        # for text rendering
        control_image=control_image_list, # canny
        control_position=control_position_list, # position
        control_mask=control_mask_list, # regional mask
        control_glyph=control_glyph_all, # as init latent, optional, set to None if not used
        controlnet_conditioning_scale=1.0,
        controlnet_conditioning_step=30,
        # for inpainting
        control_image_inpaint=control_image_inpaint,
        control_mask_inpaint=control_mask,
        controlnet_conditioning_scale_inpaint=1.0, 
        width=width,
        height=height,
        num_inference_steps=30,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]

    if not os.path.exists("./results"):
        os.makedirs("./results")
    image.save(f"./results/result_inpaint.jpg")