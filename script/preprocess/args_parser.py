import argparse
from pathlib import Path
from typing import *

def args_parser() -> Dict[str, any]:
    parser = argparse.ArgumentParser(
        prog="argparse, args_parser.py",
        description="Args parser for preprocess scripts",
    )
    parser.add_argument(
        "-i", "--input_folder",
        type=Path,
        default=Path("./dataset"),
        help="Input folder of raw images."
    )
    parser.add_argument(
        "-w", "--num_workers",
        type=int,
        default=8,
        help="Number of workers for scripts."
    )
    parser.add_argument(
        "-p", "--csv_polygon",
        type=Path,
        default=Path("./script/polygon.csv"),
        help="Path to csv polygon file."
    )
    parser.add_argument(
        "-o", "--ouput_folder",
        type=Path,
        default=Path("./dataset/processed"),
        help="Output folder of the processed images."
    )
    parser.add_argument(
        "-r", "--reset",
        action="store_true",
        help="Reset the preprocessing, (restart from scratch)"
    )
    parser.add_argument(
        "-u", "--unprocessable_method",
        type=str,
        default="change_suffix",
        choices=["change_suffix", "delete_image"],
        help="Method used from preprocess/unprocessable_image package to deal with unprocessable images."
    )
    parser.add_argument(
        "-g", "--grid_size",
        type=int,
        default=25,
        help="The size of the grid in meters for the test/queries" 
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of preprocessed images."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of preprocessed images."
    )
    parser.add_argument(
        "--is_training",
        type=bool,
        default=True,
        help="Dataset will provide a training folder, less images for the other folders."
    )
    parser.add_argument(
        "--p_training",
        type=float,
        default=0.85,
        help="Percentage of images facing north for training folder."
    )
    parser.add_argument(
        "--p_val",
        type=float,
        default=0.15,
        help="Percentage of images for evaluation folder, the rest is for test."
    )
    parser.add_argument(
        "--p_database",
        type=float,
        default=0.80,
        help="Percentage of images for database sub-folder of test and evaluation folder, the rest is for queries."
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=False,
        help="Shuffle the organized processed folder, ONLY shuffle will be executed"
    )
    return vars(parser.parse_args())
