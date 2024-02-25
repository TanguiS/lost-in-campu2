
import torch
from typing import Tuple, Union
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness: float = 0., contrast: float = 0., saturation: float = 0., hue: float = 0.):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images


class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size: Union[int, Tuple[int, int]], scale: float):
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(size=size, scale=scale)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        random_resized_crop = super(DeviceAgnosticRandomResizedCrop, self).forward
        augmented_images = [random_resized_crop(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images


def path_to_tensor(image_path):
    # Open image using PIL library
    image = Image.open(image_path)
    # Define transformation to be applied to the image
    transform = T.Compose([T.ToTensor()])
    # Apply the transformation and return the tensor
    return transform(image)

def tensor_to_image(tensor):
    """
    Takes a tensor and returns an image
    """
    image = T.functionnal.to_pil_image(tensor)
    return image

def augmented_suffix(image_path : Path, method : str) -> Path:
    name = image_path.stem
    parts = name.split('@')
    parts[-2] = "augmented"
    parts[-1] = method
    new_name = image_path.with_name(''.join(parts))
    return new_name

if __name__ == "__main__":
    tester = Path("../../dataset/processed/test/database/@691383.35@5454712.35@10@S@-0.37194@49.21528@IMG_20230216_134916_0@@-1@@@@@@.jpg")
    print(augmented_suffix(tester, "rotate"))
    #augmentation = DeviceAgnosticColorJitter(0.7, 0.7, 0.7, 0.5)
    
    
