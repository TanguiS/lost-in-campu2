import multiprocessing as mp
import preprocess.args_parser as ap
import util.polygon_manager as pm
import preprocess.unprocessable_image as uni
from preprocess.datasets_organizer import DatasetsOrganizer
from preprocess.preprocesser import RawPreProcesser

if __name__ == '__main__':
    mp.freeze_support()
    
    args_dico = ap.args_parser()
    polygon = pm.list_to_polygon(pm.read_polygon_csv(args_dico["csv_polygon"]))
    
    print(args_dico)
    
    if ( not args_dico["shuffle"] ):
        preprocesser = RawPreProcesser(
            raw_input_path=args_dico["input_folder"],
            recognition_polygon=polygon,
            processed_output_path=args_dico["ouput_folder"],
            reset=args_dico["reset"]
        )
        
        if (args_dico["unprocessable_method"] == "change_suffix"):
            unprocessable_method = uni.change_suffix
        else:
            unprocessable_method = uni.delete_image
            
        preprocesser.run(
            number_of_workers=args_dico["num_workers"],
            unprocessable_method=unprocessable_method,
            width=args_dico["width"],
            height=args_dico["height"]
        )   

    organizer = DatasetsOrganizer(
        input_output_processed_folder=args_dico["ouput_folder"],
        recognition_polygon=polygon,
        grid_square_size=args_dico["grid_size"],
        is_training=args_dico["is_training"]
    )
    
    organizer.organize(
        p_training=args_dico["p_training"],
        p_val=args_dico["p_val"],
        p_database=args_dico["p_database"],
        num_workers=args_dico["num_workers"]
    )
    organizer.display_training_distribution(polygon, 5)
    organizer.display_section_distribution(polygon, args_dico["grid_size"])

    