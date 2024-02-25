from multiprocessing import cpu_count
from pathlib import Path
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from PIL import Image
import exifread
import shutil
import utm
import util.image_manager as im
import preprocess.unprocessable_image as uni
from typing import Dict, Callable
from preprocess.verify import VerifyRawDataset
from concurrent.futures import as_completed, ProcessPoolExecutor

class RawPreProcesser():
    def __init__(self, raw_input_path : Path, 
                 recognition_polygon : Polygon,
                 processed_output_path : Path = None, 
                 verifier : VerifyRawDataset = None, 
                 reset : bool = False, 
                 ) -> None:
        self.__raw_input_path = raw_input_path
        self.__processed_output_path = processed_output_path
        self.__polygon = recognition_polygon
        if ( verifier is not None ):
            self.verifier = verifier
        else:
            self.verifier = VerifyRawDataset(self.__raw_input_path, self.__polygon)
        self.__verify_args()
        if (reset):
            self.__reset_process()
    
    def __verify_args(self):
        if ( not self.__raw_input_path.exists() ):
            raise FileNotFoundError(f"Folder : {self.__raw_input_path} does not exists")
        if ( not self.__raw_input_path.is_dir() ):
            raise NotADirectoryError(f"Path : {self.__raw_input_path} is not a Folder")
        if ( self.__processed_output_path is None ):
            default_path = self.__raw_input_path.joinpath("processed")
            if (not default_path.exists()):
                default_path.mkdir()
            self.__processed_output_path = default_path
        elif( not self.__processed_output_path.exists() ):
            raise FileNotFoundError(f"Folder : {self.__processed_output_path} does not exists")
        elif( not self.__processed_output_path.is_dir() ):
            raise NotADirectoryError(f"Path : {self.__processed_output_path} is not a Folder")
        
        
    def __is_already_processed(self, image_to_process : Path) -> bool:
        return image_to_process.stem in [im.read_datas(image)["IMG_name"][0:-2] for image in self.__processed_output_path.rglob("*.jpg*")]
    
    def __call_rezize__(self, input_image : Path, width : int, height : int, unprocessable_method : Callable[[Path], None] = uni.change_suffix) -> None:
        if (self.verifier.is_invalid_images(input_image)):
            unprocessable_method(input_image)
            return None
        if (self.__is_already_processed(input_image)):
            return None
        with open(input_image, 'rb') as img:
            tags = exifread.process_file(img, details=False)
        index = 0
        is_rotate = self.verifier.is_to_rotate(input_image)
        has_heading = not self.verifier.has_no_heading(input_image)
        for cropped_image in crop_image(tags, input_image, is_rotate, width, height):
            heading = -1
            if ( has_heading ):
                heading = im.extract_heading_datas(tags)
            save_image_for_dataset(cropped_image, input_image.with_name(f"{im.get_image_name(input_image)}_{index}"), self.__processed_output_path, im.extract_geo_datas(tags),heading)
            index += 1
        img.close()
       
    def run(self, number_of_workers : int = cpu_count(), unprocessable_method : Callable[[Path], None] = uni.change_suffix, width : int = 512, height : int = 512 ):
        self.verifier.verify(number_of_workers)
        images_to_crop = [file for file in self.__raw_input_path.iterdir() if file.is_file() and (file.suffix).lower() in im.get_image_ext()]
        with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
            futures = [executor.submit(self.__call_rezize__, input_image, width=width, height=height, unprocessable_method=unprocessable_method) for input_image in images_to_crop]
            for _ in tqdm(as_completed(futures), total=len(images_to_crop), miniters=1, desc="Cropping", unit="image"):
                pass
        print(f"==> Number of images generated : {len([img for img in self.__processed_output_path.glob('*.jpg')])}")
        
    def __reset_process(self):
        clear_processed_output_path(self.__processed_output_path)

def rename_dataset_image( image_path : Path, longitude : int, latitude : int ):
    name = im.get_image_name(image_path)
    utm_east, utm_north, _, _ = utm.from_latlon(latitude, longitude)
    tmp = image_path.with_name(f"@{round(utm_east, 2)}@{round(utm_north, 2)}@{longitude}@{latitude}@{name}@.jpg")
    image_path.rename(tmp)

def save_image_for_dataset(image, image_path : Path, output_path : Path, geo_loc : Point, heading : int = -1) -> bool:
    name = im.get_image_name(image_path)
    utm_east, utm_north, _, _ = utm.from_latlon(geo_loc.y, geo_loc.x)
    image.save(f"{output_path}/@{round(utm_east, 2)}@{round(utm_north, 2)}@10@S@{geo_loc.x}@{geo_loc.y}@{name}@@{heading}@@@@@@.jpg")
    
def get_boxes( image_tags : Dict[str, any]) -> tuple:
    size = im.extract_size_datas(image_tags)
    x = size.x
    y = size.y
    if(y > x):
        return ((0, 0, x, x), (0, y-x, x, y))
    return ((0, 0, y, y), (x-y, 0, x, y))

def crop_image(image_tags : Dict[str, any], image_to_crop_path : Path, rotate : bool, width : int, height : int) -> Image:
    boxes = get_boxes(image_tags)
    for i in range(len(boxes)):
        croppedImage = Image.open(image_to_crop_path).crop(boxes[i])
        reducedImage = croppedImage.resize((width, height))
        if rotate:
            reducedImage = reducedImage.rotate(-90)
        yield reducedImage
        croppedImage.close()

def clear_processed_output_path(processed_output_path : Path) -> None:
    if processed_output_path.exists():
        shutil.rmtree(processed_output_path)
    processed_output_path.mkdir()
    