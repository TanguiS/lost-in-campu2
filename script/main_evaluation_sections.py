from section_evaluation.section_evaluator import SectionEvaluator
import section_evaluation.section_args_parser as args_parser

from evaluation.database_loader import DatabaseLoaderPIL
from evaluation.evaluator import Evaluator
from evaluation.CosPlace_src import network

from pathlib import Path
import random
import torch
import util.polygon_manager as pm

if __name__ == "__main__":
    args = args_parser.args_parser()
    print(f"Arguments: {args}")
    if ( args['resume_model'] == None ):
        print(f"Using hub to download model : {args['backbone']} -- {args['fc_output_dim']}")
        model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                               backbone=args["backbone"],
                               fc_output_dim=args["fc_output_dim"],
                               trust_repo=True)
    else:
        print(f"Using local model : {args['resume_model']} -- {args['backbone']} -- {args['fc_output_dim']}")
        model = network.GeoLocalizationNet(args["backbone"], args["fc_output_dim"])
        model_state_dict = torch.load(args["resume_model"], map_location=str(args["device"]))
        model.load_state_dict(model_state_dict, strict=False)
    model = model.to(args["device"])
    print("Model is loaded")
    
    print(f"Creating Dataset from database folder : {args['input_database_folder']}")
    database = DatabaseLoaderPIL(args["input_database_folder"])
    print(f"Creating Evaluator from previous Dataset")
    evaluator = Evaluator(database, model,
                          num_workers=args['num_workers'],
                          infer_batch_size=args['infer_batch_size'],
                          device=args['device']
                          )
    
    folder_path = args['input_folder']
    
    if ( args["sections"] is None and args["all_sections"] is False and args['random_sections_number'] > 1 ):
        print(f"Using {args['random_sections_number']} random sections.")
        available_sections = [str(section.stem.split("_")[-1]) for section in folder_path.joinpath("queries").iterdir()]
        sections = []
        for i in range (args['random_sections_number']):
            sections.append( available_sections.pop(random.randint(0, len(available_sections) - 1)) )
        args["sections"] = sections
        print("The {args['random_sections_number']} random sections selected are", args['sections'])
    
    polygon = pm.list_to_polygon( pm.read_polygon_csv(args['csv_polygon']), True )
    section_size = args['section_size']
    
    section_evaluator = SectionEvaluator(polygon=polygon,
                                         section_size=section_size,
                                         test_input_path= folder_path,
                                         evaluator=evaluator,
                                         display_results_only=args['display_results_only'],
                                         error_margin=args['error_margin'],
                                         sections_id=args['sections'],
                                         all_sections=args['all_sections']
                                         )
    section_evaluator.display()