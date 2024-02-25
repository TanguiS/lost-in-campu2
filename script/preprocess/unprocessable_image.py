from pathlib import Path

def change_suffix(image_path : Path) -> None:
    tmp = image_path.with_suffix(image_path.suffix + ".UNPROCESSABLE")
    image_path.rename(tmp)
    
def delete_image( image_path : Path ) -> None:
    image_path.unlink()
    
