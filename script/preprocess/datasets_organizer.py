from pathlib import Path
from shapely.geometry import Polygon, Point
from util.image_manager import is_facing_north, read_datas
from util.polygon_manager import section_polygon_grid
from tqdm import tqdm
from time import time
from typing import Set
import shutil
import random
import utm
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import as_completed, ProcessPoolExecutor


def mv(input_file : Path, output_folder : Path):
    if ( input_file.exists() ):
        shutil.move(input_file, output_folder / input_file.name)

class DatasetsOrganizer():
    def __init__(self, 
                 input_output_processed_folder : Path, 
                 recognition_polygon : Polygon,
                 grid_square_size : int = 25,
                 train_sub_folder : Path = Path("train"), 
                 test_sub_folder : Path = Path("test"), 
                 val_sub_folder : Path = Path("val"),
                 is_training : bool = True,
                 ) -> None:
        self.__input_folder = input_output_processed_folder
        self.__is_training = is_training
        self.__train_folder = train_sub_folder
        self.__test_folder = test_sub_folder
        self.__val_folder = val_sub_folder
        self.__database = "database"
        self.__queries = "queries"
        self.__verify_args(recognition_polygon)
        self.__parse_args(recognition_polygon, grid_square_size)
        
    def __verify_args(self, polygon : Polygon):
        if ( not self.__input_folder.exists() ):
            raise FileNotFoundError(f"Path : {self.__input_folder} does not exist.")
        if ( not self.__input_folder.is_dir() ):
            raise NotADirectoryError(f"Path : {self.__input_folder} is not a folder.")
        self.__img_number = len(list(self.__input_folder.rglob("*.jpg")))
        if (self.__img_number == 0):
            raise Exception(f"Folder : {self.__input_folder} has not images with jpg extenssion.")
        if ( not polygon.is_valid ):
            raise ValueError(f"Polygon : {self.__polygon} is not valid.")
        
    def __parse_args(self, polygon : Polygon, grid_square_size : int):
        self.__grid = section_polygon_grid(polygon, grid_square_size)
        
    def __clean(self):
        remove_empty_dirs(self.__input_folder)
        
    def __create_section_directory(self):
        [self.__input_folder.joinpath(self.__test_folder).joinpath(self.__queries).joinpath(f"section_id_{i}").mkdir(exist_ok=True) for i in range(len(self.__grid))]
        
    def __create_directories(self):
        if ( self.__is_training ):
            self.__input_folder.joinpath(self.__train_folder).mkdir(exist_ok=True)
        self.__input_folder.joinpath(self.__test_folder).mkdir(exist_ok=True)
        self.__input_folder.joinpath(self.__test_folder).joinpath(self.__database).mkdir(exist_ok=True)
        self.__input_folder.joinpath(self.__test_folder).joinpath(self.__queries).mkdir(exist_ok=True)
        self.__input_folder.joinpath(self.__val_folder).mkdir(exist_ok=True)
        self.__input_folder.joinpath(self.__val_folder).joinpath(self.__database).mkdir(exist_ok=True)
        self.__input_folder.joinpath(self.__val_folder).joinpath(self.__queries).mkdir(exist_ok=True)
        self.__create_section_directory()
        
    def __deal_training(self, in_image : Path, p_training : float) -> bool:
        if ( not is_facing_north(in_image) ):
            return False
        if ( self.__rand.random() > p_training  ):
            return False
        mv(in_image, self.__input_folder.joinpath(self.__train_folder))
        return True
        
    def __deal_val(self, in_image : Path, p_val : float, p_database : float) -> bool:
        if ( self.__rand.random() > p_val ):
            return False
        if ( self.__rand.random() <= p_database ):
            mv(in_image, self.__input_folder.joinpath(self.__val_folder).joinpath(self.__database))
            return True
        mv(in_image, self.__input_folder.joinpath(self.__val_folder).joinpath(self.__queries))
        
    def __deal_test(self, in_image : Path, p_database : float):
        if ( self.__rand.random() <= p_database ):
            mv(in_image, self.__input_folder.joinpath(self.__test_folder).joinpath(self.__database))
            return True
        tags = read_datas(in_image)
        utm_loc = Point(tags["UTM_east"], tags["UTM_north"])
        for index, section in enumerate(self.__grid):
            if ( section.contains(utm_loc) ):
                mv(in_image, self.__input_folder.joinpath(self.__test_folder).joinpath(self.__queries).joinpath(f"section_id_{index}"))
                return True
        return False
            
    def __deal_error(self, in_image : Path):
        mv( in_image, self.__input_folder.joinpath(self.__val_folder).joinpath(self.__queries) )
            
    def __organize_image__(self, in_image : Path, p_training : float, p_val : float, p_database : float):
        self.__rand = random.Random(time())
        if ( self.__is_training and self.__deal_training(in_image, p_training) ):
            return
        if ( self.__deal_val(in_image, p_val, p_database) ):
            return
        if (self.__deal_test(in_image, p_database)):
            return
        self.__deal_error(in_image)
        
    def __display_results(self):
        print(f"-- Oragnized Tree : --\n.\n└── {self.__input_folder.stem}")
        start = "   "
        sp = "   "
        strbuilder = ""
        if ( self.__is_training ):
            start = sp
            strbuilder += start + f"└── {self.__train_folder}  : {len([img for img in self.__input_folder.joinpath(self.__train_folder).glob('*.jpg')])}\n"
        strbuilder += start + f"└── {self.__val_folder}  : {len([img for img in self.__input_folder.joinpath(self.__val_folder).rglob('*.jpg')])}\n"
        strbuilder += start + sp + f"├── {self.__database}  : {len([img for img in self.__input_folder.joinpath(self.__val_folder).joinpath(self.__database).glob('*.jpg')])}\n"
        strbuilder += start + sp + f"└── {self.__queries}  : **/ {len([img for img in self.__input_folder.joinpath(self.__val_folder).joinpath(self.__queries).rglob('*.jpg')])}\n"
  
        strbuilder += start + f"└── {self.__test_folder}  : {len([img for img in self.__input_folder.joinpath(self.__test_folder).rglob('*.jpg')])}\n"
        strbuilder += start + sp + f"├── {self.__database}  : {len([img for img in self.__input_folder.joinpath(self.__test_folder).joinpath(self.__database).glob('*.jpg')])}\n"
        strbuilder += start + sp + f"└── {self.__queries}  : **/ {len([img for img in self.__input_folder.joinpath(self.__test_folder).joinpath(self.__queries).rglob('*.jpg')])}\n"
        print(strbuilder)
        
    def organize(self, p_training : int = 0.85, p_val : int = 0.15, p_database : int = 0.8, num_workers : int = mp.cpu_count()):
        self.__clean()
        self.__create_directories()
        images_to_organize = [img for img in self.__input_folder.rglob("*.jpg")]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.__organize_image__, img, p_training, p_val, p_database) for img in images_to_organize]
            for _ in tqdm(as_completed(futures), total=len(futures), miniters=1, desc="Organizing images", unit="image"):
                pass
        self.__clean()
        self.__display_results()
    
    def __number_images_from_id_section(self, index : int) -> int:
        None
        
    def __xy_images_from_id_section(self, index : int)->Set:
        coords = set()
        if ( not self.__input_folder.joinpath(self.__test_folder).joinpath(self.__queries).joinpath(f"section_id_{index}").exists() ):
            return None
        for img in self.__input_folder.joinpath(self.__test_folder).joinpath(self.__queries).joinpath(f"section_id_{index}").iterdir():
            tags = read_datas(img)
            coords.add((tags["UTM_east"], tags["UTM_north"]))
        return coords
    
    def __xy_images_from_training(self)->Set:
        coords = set()
        for img in self.__input_folder.joinpath(self.__train_folder).glob("*.jpg"):
            tags = read_datas(img)
            coords.add((tags["UTM_east"], tags["UTM_north"]))
        return coords
    
    def display_section_distribution(self, GPS_polygon : Polygon, grid_quare_size : int):
        plt.figure("Dataset Section Distribution")
        plt.title(f"Dataset section distribution w\\{grid_quare_size} meters grid")
        plt.xlabel("UTM east")
        plt.ylabel("UTM north")
        x, y = Polygon([(utm.from_latlon(lat, lon)[:2]) for lon, lat in list(GPS_polygon.exterior.coords)]).exterior.xy
        plt.plot(x, y, 'k-o', markersize=5, alpha=0.7, label='Recognition Polygon : Campus 2')
        for index, poly in enumerate(self.__grid):
            if ( self.__xy_images_from_id_section(index) is None ):
                continue
            for x, y in self.__xy_images_from_id_section(index):
                plt.plot(x, y, 'o', markersize=2, color="#00FF00")
            x, y = poly.exterior.xy
            plt.plot(x, y, linestyle='dotted', linewidth=1.5, markevery=3)
        plt.legend()
        plt.show()
        
    def display_training_distribution(self, GPS_polygon : Polygon, grid_quare_size : int):
        plt.figure("Dataset Training Distribution")
        plt.title(f"Dataset training distribution w\\{grid_quare_size} meters grid")
        plt.xlabel("UTM east")
        plt.ylabel("UTM north")
        grid = section_polygon_grid(GPS_polygon, grid_quare_size)
        x, y = Polygon([(utm.from_latlon(lat, lon)[:2]) for lon, lat in list(GPS_polygon.exterior.coords)]).exterior.xy
        plt.plot(x, y, 'k-o', markersize=5, alpha=0.7, label='Recognition Polygon : Campus 2')
        for poly in grid:
            x, y = poly.exterior.xy
            plt.plot(x, y, linestyle='dotted', linewidth=1.5, markevery=3)
        for x, y in self.__xy_images_from_training():
            plt.plot(x, y, 'o', markersize=2, color="#00FF00")
        plt.legend()
        plt.show()


def remove_empty_dirs(root : Path):
    for path in root.iterdir():
        if ( path.is_dir() ):
            remove_empty_dirs(path)
            try:
                path.rmdir()
            except OSError as e:
                pass
           
        