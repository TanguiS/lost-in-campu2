from shapely.geometry import Polygon
from pathlib import Path
from tqdm import tqdm
from util.polygon_manager import is_in_polygon
from util.image_manager import extract_geo_datas, get_image_ext
import multiprocessing as mp
import exifread
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Dict, Union

class VerifyRawDataset():
    def __init__(self, dataset_folder : Path, recognition_polygon : Polygon) -> None:
        self.__dataset_folder = dataset_folder
        self.__polygon = recognition_polygon
        self.__invalid_images = mp.Manager().list()
        self.__needs_to_be_rotated = mp.Manager().list()
        self.__has_no_heading = mp.Manager().list()
        self.__verify_args()

    def __verify_args(self):
        if ( not self.__dataset_folder.exists() ):
            raise FileNotFoundError(f"Folder : {self.__dataset_folder} does not exists")
        if ( not self.__dataset_folder.is_dir() ):
            raise NotADirectoryError(f"Path : {self.__dataset_folder} is not a Folder")
        

    def __verify_image__(self, image_path : Path):
        if ( not is_image(image_path) ):
            return
        try:
            image_tags = try_read_metadatas(image_path)
        except Exception as e:
            self.__invalid_images.append(image_path)
            return
        if (image_tags is None):
            self.__invalid_images.append(image_path)
            return
        if (not has_size_metadata(image_tags)):
            self.__invalid_images.append(image_path)
            return
        if (not has_geolocation_metadata(image_tags)):
            self.__invalid_images.append(image_path)
            return
        if (not is_in_polygon(extract_geo_datas(image_tags), self.__polygon)):
            self.__invalid_images.append(image_path)
            return
        if (not has_heading_metadata(image_tags)):
            self.__has_no_heading.append(image_path)
        if (is_rotate(image_tags)):
            self.__needs_to_be_rotated.append(image_path)
            
    def verify(self, number_of_workers : int = mp.cpu_count()):    
        files_to_verify = [file for file in self.__dataset_folder.iterdir()]
        with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
            futures = [executor.submit(self.__verify_image__, to_verify) for to_verify in files_to_verify]
            for _ in tqdm(as_completed(futures), total=len(futures), miniters=1, desc="Verification", unit="image"):
                pass
        self.__invalid_images = set(self.__invalid_images)
        self.__needs_to_be_rotated = set(self.__needs_to_be_rotated)
        self.__has_no_heading = set(self.__has_no_heading)
        print(f" ==> {len(self.__invalid_images)} incorect images were found.")
        print(f" ==> {len(self.__needs_to_be_rotated)} images need to be rotated.")
        print(f" ==> {len(self.__has_no_heading)} images have no heading.")
        
    def is_invalid_images(self, input_image : Path) -> bool:
        return input_image in self.__invalid_images

    def is_to_rotate(self, input_image : Path) -> bool:
        return input_image in self.__needs_to_be_rotated
    
    def has_no_heading(self, input_image : Path) -> bool:
        return input_image in self.__has_no_heading

def is_image(file_path : Path) -> bool:
    if ( not file_path.is_file() ):
        return False
    image_extensions = get_image_ext()
    if ( (file_path.suffix).lower() in image_extensions ):
        return True
    return False

def try_read_metadatas(image_path : Path) -> Union[Dict[str, any], None]:
    try:
        with open(image_path, 'rb') as img:
            tags = exifread.process_file(img, details=False)
            img.close()
        if ( tags is None ):
            return None
    except Exception as e:
        raise Exception(e)
    return tags

def has_geolocation_metadata(image_tags: Dict[str, any]) -> bool:
    return 'GPS GPSLatitude' in image_tags and 'GPS GPSLongitude' in image_tags and 'GPS GPSLatitudeRef' in image_tags and 'GPS GPSLongitudeRef' in image_tags

def has_heading_metadata(image_tags: Dict[str, any]) -> bool:
    return 'GPS GPSImgDirection' in image_tags and 'GPS GPSImgDirectionRef' in image_tags

def has_size_metadata(image_tags: Dict[str, any]) -> bool:
    return 'EXIF ExifImageWidth' in image_tags and 'EXIF ExifImageLength' in image_tags

def is_rotate(image_tags : Dict[str, any]) -> bool:
    if ( 'Image Orientation' in image_tags ):
        return "Rotated" in str(image_tags['Image Orientation'])
    return False