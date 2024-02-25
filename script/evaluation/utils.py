import torchvision.transforms as transforms
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import Union

def base_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
def path_to_base64(input_image : Path):
    with open(input_image, 'rb') as img:
        img_bytes = img.read()
    return base64.b64encode(img_bytes).decode("utf-8")
    
def open_image_path(input_image : Path) -> Image:
    return Image.open(input_image).convert("RGB")

def open_image_bytes(img_bytes : bytes) -> Image:
    return Image.open(BytesIO(img_bytes)).convert("RGB")

def get_normalized_image_path(input_image : Path):
        pil_img = open_image_path(input_image)
        normalized_img = base_transform()(pil_img)
        return normalized_img

def get_normalized_image_base64( input_image : str ):
    img_bytes = base64.b64decode(input_image)
    pil_img = open_image_bytes(img_bytes)
    normalized_img = base_transform()(pil_img)
    return normalized_img

def get_normalized_image(input_image : Union[Path, str], is_base64 : bool = False):
    if (is_base64):
        return get_normalized_image_base64(input_image)
    return get_normalized_image_path(input_image)