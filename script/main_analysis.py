from analysis import args_parser as ap
from analysis.data_analyzer import DataAnalysis
from pathlib import Path

if __name__ == "__main__":
    args_dico = ap.args_parser()
    print(f"Argument: {args_dico}")
    analyzis = DataAnalysis(input_output_processed_folder=args_dico["input_folder"], 
                            polygon_csv=args_dico["csv_polygon"],
                            dx=args_dico["unit_size"],
                            max_density=args_dico["density_max"],
                            map_in_background=args_dico["map_displayed"]
                            )

    #folder = folder = args_dico["val_folder"]
    if args_dico["smooth_display"]:
        analyzis.show_heatmap_smooth()
    else:
        analyzis.show_heatmap_concrete()