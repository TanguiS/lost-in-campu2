import faiss
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import DataLoader
from pathlib import Path
from evaluation.database_loader import DatabaseLoaderPIL, DatabaseLoaderPaths
from util.image_manager import read_datas
from evaluation.utils import get_normalized_image
from shapely.geometry import Point

RECALL_VALUES = [1, 5, 10, 20]

class Evaluator():
    def __init__(
                    self, eval_ds       : DatabaseLoaderPIL, 
                    model               : torch.nn.Module, 
                    infer_batch_size    : int = 32, 
                    num_workers         : int = 8, 
                    device              : str = "cuda", 
                    fc_output_dim       : int = 2048
                ):
        
        self.model = model.eval()
        self.eval_ds = eval_ds
        self.device = device
        database_descriptors = self.__load_database_descriptors(num_workers, infer_batch_size, fc_output_dim)
        self.faiss_index = faiss.IndexFlatL2(fc_output_dim)
        self.faiss_index.add(database_descriptors)
        del database_descriptors
        self.eval_ds.__class__ = DatabaseLoaderPaths
        
    def __load_database_descriptors(self, num_workers : int, infer_batch_size : int, fc_output_dim : int):
        with torch.no_grad():
            database_dataloader = DataLoader(dataset=self.eval_ds, num_workers=num_workers,
                                            batch_size=infer_batch_size, pin_memory=(self.device == "cuda"))
            database_descriptors = np.empty((len(self.eval_ds), fc_output_dim), dtype="float32")
            for images, indices in tqdm(database_dataloader, ncols=100, desc="Extracting database descriptors", total=len(database_dataloader), miniters=1, unit="descriptor"):
                descriptors = self.model(images.to(self.device))
                descriptors = descriptors.cpu().numpy()
                database_descriptors[indices.numpy(), :] = descriptors
        return database_descriptors
        
    def __input_image_descriptors(self, input_image : Path, is_base64 : bool):
        normalized_img = get_normalized_image(input_image, is_base64).unsqueeze(0)
        with torch.no_grad():
            descriptors = self.model(normalized_img.to(self.device))
            descriptors = descriptors.cpu().numpy()
        return descriptors
    
    def __geoloc_prediction(self, indexes : List[int] ) -> Dict[str, Point]:
        GPS_prediction = dict()
        GPS_prediction["GPS_lonlat"] = list()
        GPS_prediction["GPS_utm"] = list()
        for index in indexes:
            datas = read_datas(self.eval_ds[index][0])
            GPS_prediction["GPS_lonlat"].append(Point(datas["GPS_longitude"], datas["GPS_latitude"]))
            GPS_prediction["GPS_utm"].append(Point(datas["UTM_east"], datas["UTM_north"]))
            
        GPS_prediction["GPS_lonlat"] = Point(
            sum(p.x for p in GPS_prediction["GPS_lonlat"]) / len(GPS_prediction["GPS_lonlat"]),
            sum(p.y for p in GPS_prediction["GPS_lonlat"]) / len(GPS_prediction["GPS_lonlat"])
        )
        GPS_prediction["GPS_utm"] = Point(
            sum(p.x for p in GPS_prediction["GPS_utm"]) / len(GPS_prediction["GPS_utm"]),
            sum(p.y for p in GPS_prediction["GPS_utm"]) / len(GPS_prediction["GPS_utm"])
        )
        return GPS_prediction
    
    def evaluate(self, input_image : Path, is_base64 : bool = False) -> Dict[str, Point]:
        recall = RECALL_VALUES[0]
        descriptors = self.__input_image_descriptors(input_image, is_base64)
        _, index_prediction_matrix = self.faiss_index.search(descriptors, recall)
        closest_geoloc = self.__geoloc_prediction(index_prediction_matrix[0][:recall])
        return closest_geoloc
        