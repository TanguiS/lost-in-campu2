import argparse
from pathlib import Path
from typing import *

def args_parser() -> Dict[str, any]:
    parser = argparse.ArgumentParser(
        prog="argparse, section_args_parser.py",
        description="Args parser for section evaluator scripts",
    )
    parser.add_argument(
        "-p", "--csv_polygon",
        type=Path,
        default=Path("./script/polygon.csv"),
        help="Path to csv polygon file."
    )
    parser.add_argument(
        "-i", "--input_folder",
        type=Path,
        default=Path("./dataset/processed/test"),
        help="Folder of the processed images where the sections are."
    )
    parser.add_argument(
        "-s", "--sections",
        type=int,
        nargs="+",
        default=None,
        help="List of sections to evaluate, if None use --random_sections_folders to find 5 random sections to evaluate."
    )
    parser.add_argument(
        "-r", "--random_sections_number",
        type=int,
        default=5,
        help="Number of random sections to evaluate if --sections is None, 5 random sections if --random_sections_number is not specified"
    )
    parser.add_argument(
        "-A", "--all_sections",
        action="store_true",
        help="Evaluate all the sections"
    )
    parser.add_argument(
        "--input_database_folder",
        type=Path,
        default=Path("./dataset/processed"),
        help="Input folder of processed images for database recognition (recurcive=True)."
    )
    parser.add_argument(
        "-w", "--num_workers",
        type=int,
        default=8,
        help="Number of workers for scripts."
    )
    parser.add_argument(
        "-b", "--infer_batch_size", 
        type=int, 
        default=16,
        help="Batch size for inference (validating and testing)"
    )
    parser.add_argument(
        "--resume_model", 
        type=Path, 
        default=None,
        help="path to model to resume, e.g. logs/.../best_model.pth -- If None, Auto DL from CosPlace using --backbone & --fc_output_dim args"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu"], 
        help="Device to use for calculation augmentation"
    )
    parser.add_argument(
        "--backbone", 
        type=str, 
        default="ResNet50",
        choices=["VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152"], 
        help="Choice of Backbone from CosPlace"
    )
    parser.add_argument(
        "--fc_output_dim", 
        type=int, 
        default=2048,
        help="Output dimension of final fully connected layer"
    )
    parser.add_argument(
        "-d", "--display_results_only",
        action="store_true",
        help="Display only a pie chart and the average time on the section, to know the proportion of good predictions"
    )
    parser.add_argument(
        "-m", "--error_margin",
        type=float,
        default=10.0,
        help="Margin of error, to define the proportion of good predictions"
    )
    parser.add_argument(
        "--section_size",
        type=int,
        default=25,
        help="Size (width) of a section"
    )
    return vars(parser.parse_args())
