import argparse
from pathlib import Path
from typing import Dict

def args_parser() -> Dict[str, any]:
    parser = argparse.ArgumentParser(
        prog="argparse, args_parser.py",
        description="Args parser for analysis scripts",
    )
    parser.add_argument(
        "-i", "--input_folder",
        type=Path,
        default=Path("./dataset/processed"),
        help="Input folder of the processed images. (recurcif)"
    )
    parser.add_argument(
        "-map", "--map_displayed",
        action="store_true",
        help="The map will be displayed in the background"
    )
    parser.add_argument(
        "-smooth", "--smooth_display",
        action="store_true",
        help="Shows data in a smooth way"
    )
    parser.add_argument(
        "-dx", "--unit_size",
        type=int,
        default=5,
        help="Size in meter of each square for a non smooth display"
    )
    parser.add_argument(
        "-max", "--density_max",
        type=int,
        default=12,
        help="Number of datas required per square of size X size"
    )
    parser.add_argument(
        "-p", "--csv_polygon",
        type=Path,
        default=Path("./script/polygon.csv"),
        help="Path to csv polygon file."
    )
    
    return vars(parser.parse_args())