from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def arg_parser(default_path : Path):
    parser = argparse.ArgumentParser(
        prog="unrename-image",
        description="remove the .UNPROCESSABLE suffix / tag from images"
    )
    parser.add_argument(
        "-i", "--input_path",
        type=Path,
        default=default_path,
        help="Path to the image to be processed.")
    return parser.parse_args();

def rename_file(path : Path) -> None:
    if path.suffix == '.UNPROCESSABLE':
        new_path = path.with_suffix('')
        path.rename(new_path)

def process_folder(folder_path : Path) -> None:
    with ThreadPoolExecutor() as executor:
        paths = list(folder_path.glob('*.*'))
        tasks = [executor.submit(rename_file, path) for path in paths]
        for task in tqdm(as_completed(tasks), total=len(tasks), miniters=1, desc="Parsing images", unit="image"):
            task.result()

def main() -> None:
    images_path = arg_parser(
        Path("/media", "tangui", "My Passport", "storage-photos", "lost-in-campus-2", "dataset")
        ).input_path
    folder_path = Path(images_path)
    process_folder(folder_path) 

if __name__ == '__main__':
    main()
