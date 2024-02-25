import argparse
from pathlib import Path
from typing import *

def args_parser() -> Dict[str, any]:
    parser = argparse.ArgumentParser(
        prog="argparse, args_parser.py",
        description="Args parser for preprocess scripts",
    )
    parser.add_argument(
        "-i", "--input_database_folder",
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
        "-m", "--resume_model", 
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
        "--image_to_evaluate",
        type=Path,
        default=None,
        help="Path to the image to evaluate, if None use --random_queries_folder to find a random image to evaluate."
    )
    parser.add_argument(
        "--random_queries_folder",
        type=Path,
        default=None,
        help="Path to queries folder to find a random images to evaluate if --image_to_evaluate is None, if both are None -> Error"
    )
    parser.add_argument(
        "--use_base64",
        action="store_true",
        help="Transform Path of the image to evaluate to a base64, simulation of server request."
    )
    return vars(parser.parse_args())