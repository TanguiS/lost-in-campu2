import torch.utils.data as data
from pathlib import Path
from evaluation.utils import get_normalized_image

class DatabaseLoaderPIL(data.Dataset):
    def __init__(self, database_folder : Path = Path("processed")):
        super().__init__()
        self.database_folder = database_folder
        self.__verify_args()
        self.database_paths = sorted( self.database_folder.rglob("*.jpg") )

    def __verify_args(self):
        if ( not self.database_folder.exists() ):
            raise FileNotFoundError(f"Folder : {self.database_folder} does not exists")
        if ( not self.database_folder.is_dir() ):
            raise NotADirectoryError(f"Path : {self.database_folder} is not a Folder")

    def __getitem__(self, index):
        image_path = self.database_paths[index]
        return get_normalized_image(image_path), index
    
    def __len__(self):
        return len(self.database_paths)
    
    def __repr__(self):
        return f"< {self.database_folder.stem} - #db: {self.database_paths} >"
        
    
class DatabaseLoaderPaths(DatabaseLoaderPIL):
    def __init__(self, database_folder : Path = Path("processed")):
        super.__init__(database_folder, database_folder)
    
    def __getitem__(self, index):
        return self.database_paths[index], index
