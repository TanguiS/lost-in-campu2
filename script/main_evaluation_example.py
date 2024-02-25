from evaluation.database_loader import DatabaseLoaderPIL
from evaluation.evaluator import Evaluator
from evaluation.utils import path_to_base64
from util.image_manager import read_datas
import evaluation.args_parser as args_parser

import torch
import random
from evaluation.CosPlace_src import network
from pathlib import Path

torch.backends.cudnn.benchmark = True  # Provides a speedup

if __name__ == "__main__":
    args = args_parser.args_parser()
    print(f"Arguments: {args}")
    if ( args['resume_model'] == None ):
        print(f"Using hub to download model : {args['backbone']} -- {args['fc_output_dim']}")
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone=args["backbone"], fc_output_dim=args["fc_output_dim"], trust_repo=True)
    else:
        print(f"Using local model : {args['resume_model']} -- {args['backbone']} -- {args['fc_output_dim']}")
        model = network.GeoLocalizationNet(args["backbone"], args["fc_output_dim"])
        model_state_dict = torch.load(args["resume_model"], map_location=str(args["device"]))
        model.load_state_dict(model_state_dict)
    model = model.to(args["device"])
    print("Model is loaded")
    
    if ( args['image_to_evaluate'] is None and args['random_queries_folder'] is None ):
        raise IOError
    if ( args["image_to_evaluate"] is None ):
        print(f"Using a random images from {args['random_queries_folder']}")
        images = [img for img in args['random_queries_folder'].rglob("*.jpg")]
        args["image_to_evaluate"] = images[random.randint(0, len(images) - 1)]
    print(f"Image to evaluate : {args['image_to_evaluate']}")
    args["image"] = args["image_to_evaluate"]
    if ( args["use_base64"] ):
        print(f"Using base64 to evaluate, simulation of a server request. {args['use_base64']}")
        args["image"] = path_to_base64(args["image_to_evaluate"])
    
    print(f"Creating Dataset from database folder : {args['input_database_folder']}")
    database = DatabaseLoaderPIL(args["input_database_folder"])
    print(f"Creating Evaluator from previous Dataset")
    evaluator = Evaluator(database, model, num_workers=args['num_workers'], infer_batch_size=args['infer_batch_size'], device=args['device'])
    
    print(" -- Evaluation -- ")
    prediction = evaluator.evaluate(args["image"], args["use_base64"])
    truth = read_datas(args["image_to_evaluate"])
    print(f"UTM_prediction = {prediction['GPS_utm']}\nUTM_truth = {truth['UTM_east']}, {truth['UTM_north']}\n - Diff UTM_east : {abs(prediction['GPS_utm'].x - truth['UTM_east'])} meters\n - Diff UTM_north : {abs(prediction['GPS_utm'].y - truth['UTM_north'])} meters")
    
    