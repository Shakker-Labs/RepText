<div align="center">
<h1>RepText: Rendering Visual Text via Replicating </h1>

<div>
    <a href='https://haofanwang.github.io/'>Haofan Wang<sup>†</sup></a>, 
    Yujia Xu, 
    Yimeng Li, 
    Junchen Li, 
    Chaowei Zhang, 
    Jing Wang, 
    Kejia Yang, 
    Zhibo Chen
</div>
<div>
    Shakker Labs, Liblib AI<br>
   <p><sup>†</sup>Corresponding author</p>
</div>

<a href='https://reptext.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2504.19724'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Shakker-Labs/RepText)

</div>

We present RepText, which aims to empower pre-trained monolingual text-to-image generation models with the ability to accurately render, or more precisely, replicate, multilingual visual text in user-specified fonts, without the need to really understand them. Specifically, we adopt the setting from ControlNet and additionally integrate language agnostic glyph and position of rendered text to enable generating harmonized visual text, allowing users to customize text content, font and position on their needs. To improve accuracy, a text perceptual loss is employed along with the diffusion loss. Furthermore, to stabilize rendering process, at the inference phase, we directly initialize with noisy glyph latent instead of random initialization, and adopt region masks to restrict the feature injection to only the text region to avoid distortion of the background. We conducted extensive experiments to verify the effectiveness of our RepText relative to existing works, our approach outperforms existing open-source methods and achieves comparable results to native multi-language closed-source models.

<div align="center">
<img src='assets/example1.png' width=1024>
</div>

## ⭐ Update
- [2025/06/07] [Model Weights](https://huggingface.co/Shakker-Labs/RepText) and inference code released!
- [2025/04/28] [Technical Report](https://arxiv.org/abs/2504.19724) released!

## Method

<div align="center">
<img src='assets/train.png' width=1024>
</div>

<div align="center">
<img src='assets/infer.png' width=1024>
</div>

## Usage
```python
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
```

For inpainting demo, 

```python
python infer_inpaint.py
```



## Compatibility to Other Works
- [FLUX.1-dev-ControlNet-Union-Pro-2.0](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0).
<div align="center">
<img src='assets/union.png' width=1024>
</div>

- [FLUX.1-dev-Controlnet-Inpainting-Beta](https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta).
<div align="center">
<img src='assets/inpaint.png' width=1024>
</div>

- [FLUX.1-dev-IP-Adapter](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter).
<div align="center">
<img src='assets/ipa.png' width=1024>
</div>

## Generated Samples

<div align="center">
<img src='assets/example2.png' width=1024>
<img src='assets/example3.png' width=1024>
<img src='assets/example4.png' width=1024>
<img src='assets/example5.png' width=1024>
</div>

## 📑 Citation
If you find RepText useful for your research and applications, please cite us using this BibTeX:
```bibtex
@article{wang2025reptext,
  title={RepText: Rendering Visual Text via Replicating},
  author={Wang, Haofan and Xu, Yujia and Li, Yimeng and Li, Junchen and Zhang, Chaowei and Wang, Jing and Yang, Kejia and Chen, Zhibo},
  journal={arXiv preprint arXiv:2504.19724},
  year={2025}
}
```

## 📧 Contact
If you have any questions, please feel free to reach us at `haofanwang.ai@gmail.com`. 
