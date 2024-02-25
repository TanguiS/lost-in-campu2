# libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import time
import utm
import math as m
from pathlib import Path
from shapely.geometry import Polygon, Point
from evaluation.evaluator import Evaluator
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.feature import ShapelyFeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class SectionEvaluator:
    def __init__(self,
                 polygon: Polygon,
                 section_size: int,
                 test_input_path: Path,
                 evaluator: Evaluator,
                 display_results_only: bool,
                 error_margin: float,
                 all_sections: bool,
                 sections_id: list = []):
        plt.rcParams["figure.figsize"] = [32, 32]
        plt.rcParams["figure.autolayout"] = True
        self.__distance_limit = 20
        self.__cmap = mcolors.LinearSegmentedColormap.from_list('distancescmap', ['#87d37c', '#f7dc6f', '#ec7063'])
        self.__all_sections = all_sections
        self.__evaluator = evaluator
        self.__display_results_only = display_results_only
        self.__error_margin = error_margin
        self.__zone_number = 30 # UTM zone for Caen
        self.__zone_letter = "U" # Northern hemisphere
        self.__N = 20
        self.__polygon = polygon
        self.__section_size = section_size
        
        if (self.__all_sections):
            self.__sections = [section for section in test_input_path.joinpath("queries").iterdir()]
        else:
            unique_sections_id = self.__remove_duplicates_of_lists( list=sections_id )
            self.__sections = []
            for section_id in unique_sections_id:
                self.__sections.append([section for section in test_input_path.joinpath("queries").glob("*_" + str(section_id))])
            self.__sections = [section for sublist in self.__sections for section in sublist]

    def __remove_duplicates_of_lists(self,
                                       list: list) -> list:
        list_of_unique_elements = []
        for element in list:
            if element not in list_of_unique_elements:
                list_of_unique_elements.append(element)
        return list_of_unique_elements
    
    def __get_coordinates_from_image(self,
                                       image_path: Path) -> list:
        coordinates = image_path.stem.split("@")
        return [ float(coordinates[1]), float(coordinates[2]) ]
         
    def __compute_distance(self,
                             marker_truth,
                             marker_prediction):
        first_factor = marker_truth[0] - marker_prediction[0]
        second_factor = marker_truth[1] - marker_prediction[1]
        return m.sqrt( first_factor * first_factor + second_factor * second_factor )

    def __compute_distance_with_coord(self,
                                        marker_truth,
                                        point_x,
                                        point_y):
        first_factor = marker_truth[0] - point_x
        second_factor = marker_truth[1] - point_y
        return m.sqrt( first_factor * first_factor + second_factor * second_factor )

    def __direction_from_truth_to_prediction( self,
                                               marker_truth,
                                               marker_prediction,
                                               distance ):
        abscissa = marker_prediction[0] - marker_truth[0]
        ordinate = marker_prediction[1] - marker_truth[1]
        if (distance != 0):
            abscissa = ( abscissa / distance )
            ordinate = ( ordinate / distance )
        return abscissa, ordinate
        
    def __point_for_line_from_truth_to_prediction( self,
                                                    marker_truth,
                                                    abscissa_increment,
                                                    ordinate_increment,
                                                    index ):
        return marker_truth[0] + index * abscissa_increment, marker_truth[1] + index * ordinate_increment

    def __get_distance_line(self,
                            marker_truth,
                            marker_prediction):
        distance = self.__compute_distance(marker_truth, marker_prediction)
        points_x = [ marker_truth[0] ]
        points_y = [ marker_truth[1] ]
        distances_points_from_truth = [ self.__compute_distance_with_coord(marker_truth, points_x[0], points_y[0]) ]
        direction_abscissa, direction_ordinate = self.__direction_from_truth_to_prediction(marker_truth, marker_prediction, distance)
        for i in range (1, self.__N + 1):
            tmp_point_x, tmp_point_y = self.__point_for_line_from_truth_to_prediction(marker_truth, 
                                                                            direction_abscissa * (distance / self.__N),
                                                                            direction_ordinate * (distance / self.__N),
                                                                            i)
            points_x.append(tmp_point_x)
            points_y.append(tmp_point_y)
            distances_points_from_truth.append( self.__compute_distance_with_coord(marker_truth, points_x[i], points_y[i]) )
        distances_array = np.array(distances_points_from_truth)
        colors = []
        distances_norm = distances_array / self.__distance_limit
        distances_norm[distances_norm > 1] = 1
        for i in range(self.__N + 1):
            colors.append(self.__cmap(distances_norm[i]))
        return points_x, points_y, colors

    def __plot_markers(self,
                       marker_truth,
                       marker_prediction):
        plt.plot(marker_truth[0], marker_truth[1], 'g', marker='v')
        plt.plot(marker_prediction[0], marker_prediction[1], 'b', marker='v')
    
    def __plot_distance_line(self,
                             points_x,
                             points_y,
                             colors ):
        plt.scatter(points_x, points_y, c=colors, s=10, edgecolors='none')
        for i in range (self.__N):
            plt.plot([points_x[i], points_x[i+1]], [points_y[i], points_y[i+1]], color=colors[i], linewidth=3)

    def __display_distance_box(self,
                          marker_truth,
                          marker_prediction,
                          distance,
                          colors):
        position_distance_box_x = (marker_truth[0] + marker_prediction[0]) / 2
        position_distance_box_y = (marker_truth[1] + marker_prediction[1]) / 2
        plt.text(position_distance_box_x, position_distance_box_y, str(round(distance, 2)) + "m", size=12,
                ha="center", va="center",
                bbox=dict(facecolor=colors[-1],
                          boxstyle="round"
                          )
                )

    def __display_time_box(self,
                          position,
                          time):
        position_time_box_x = position[0]
        position_time_box_y = position[1] - 12
        plt.text(position_time_box_x, position_time_box_y, str(round(time, 3)) + "s", size=12,
                ha="center", va="center",
                bbox=dict(facecolor='w',
                          boxstyle="round")
                )

    def __display_polygon(self,
                            polygon,
                            map,
                            proj_utm):      
        points = [(point[0], point[1]) for point in polygon.exterior.coords]
        margin = 50
        request = cimgt.OSM()
        extent = [min(polygon.exterior.xy[0]) - margin, max(polygon.exterior.xy[0]) + margin, min(polygon.exterior.xy[1]) - margin, max(polygon.exterior.xy[1]) + margin]
        map.set_extent(extent, crs=proj_utm)
        map.add_image(request, 19)
        poly_feature = ShapelyFeature([polygon], crs=proj_utm, facecolor="none", edgecolor="blue", linewidth=2)
        map.add_feature(poly_feature)
        for point in points:
            plt.plot(point[0], point[1], "-ro", transform=proj_utm)
    
    def __display_pie_chart(self,
                            proportions,
                            position,
                            plot) -> any:
        pie_axis = inset_axes(plot, width=0.7, height=0.7, loc=10, bbox_to_anchor=(position[0], position[1]), bbox_transform=plot.transData)
        size = 0.5
        pie_axis.pie(proportions, radius=1, startangle=90, colors=['g', 'r'], wedgeprops=dict(width=size))

    def __display_markers_and_distance_line(self,
                                              point_truth,
                                              point_prediction,
                                              points_x,
                                              points_y,
                                              distance,
                                              colors):
        self.__plot_distance_line(points_x, points_y, colors)
        self.__plot_markers(point_truth, point_prediction)
        self.__display_distance_box(point_truth, point_prediction, distance, colors)
    
    def __compute_average(self,
                          list):
        average = 0
        for value in list:
            average += value
        return round(average / len(list), 2)
    
    def __get_section_id_from_path(self,
                                   path: Path) -> str:
        return path.name.split("_")[-1]
    
    def __generate_dict_of_sections_positions(self,
                                    polygon: Polygon) -> dict:
        sections_dict = dict()
        current_id = 1
        UTM_exterior = polygon.exterior.xy
        UTM_min_east, UTM_max_east, UTM_min_north, UTM_max_north = (min(UTM_exterior[0]), max(UTM_exterior[0]), min(UTM_exterior[1]), max(UTM_exterior[1]))
        for x in range(int(UTM_min_east), int(UTM_max_east), self.__section_size):
            for y in range(int(UTM_min_north), int(UTM_max_north), self.__section_size):
                x_right = x + self.__section_size
                y_top = y + self.__section_size
                if polygon.contains(Point(x, y)) or polygon.contains(Point(x_right, y)) or polygon.contains(Point(x_right, y_top)) or polygon.contains(Point(x, y_top)):
                    sections_dict[str(current_id)] = (x + 7 , y + 15)
                    current_id += 1
        return sections_dict
    
    def __generate_pie_chart_and_time_legends(self,
                                              plot):
        legends = [
            Patch(facecolor='g', label="Good prediction"), 
            Patch(facecolor='r', label="Bad prediction"), 
            Patch(edgecolor='black', label="Average time", fill=False) 
        ]
        
        plot.legend(handles=legends, prop={'size':40})
        
    def __generate_markers_legends(self,
                                   plot):
        truth_legend = mlines.Line2D([], [], color='g', marker='v', linestyle='None',
                          markersize=10, label='Truth')
        prediction_legend = mlines.Line2D([], [], color='b', marker='v', linestyle='None',
                          markersize=10, label='Prediction')
        plot.legend(handles=[truth_legend, prediction_legend], prop={'size':40})
    
    def __evaluate_section(self,
                           fig,
                           plot,
                           proportions_dict,
                           all_distances_list,
                           all_times_list,
                           position_from_id,
                           section,
                           good_prediction_counter,
                           times_for_a_section,
                           distances):
        for image in section.iterdir():
            point_truth = self.__get_coordinates_from_image(image)
            
            # Prediction work
            pre_prediction = time.time()
            prediction = self.__evaluator.evaluate(image)
            post_prediction = time.time()
            point_prediction = [ prediction['GPS_utm'].x, prediction['GPS_utm'].y ]
            # End of prediction work
            
            times_for_a_section.append(post_prediction - pre_prediction)
            distance = self.__compute_distance(point_truth, point_prediction)
            distances.append( distance )
            
            if ( self.__display_results_only is False ):
                points_x, points_y, colors = self.__get_distance_line(point_truth, point_prediction)
                self.__display_markers_and_distance_line(point_truth=point_truth,
                                                        point_prediction=point_prediction,
                                                        points_x=points_x,
                                                        points_y=points_y,
                                                        distance=distance,
                                                        colors=colors)
        for distance in distances:
            if distance < self.__error_margin:
                good_prediction_counter = good_prediction_counter + 1
            all_distances_list.append( distance )
        average_time = self.__compute_average(list=times_for_a_section)
        all_times_list.append( average_time )
        if ( self.__display_results_only ):
            proportions_dict[self.__get_section_id_from_path(path=section)] = [ good_prediction_counter, len(distances) - good_prediction_counter ]
            self.__display_time_box(position=position_from_id, 
                                    time=average_time)
       
    def display(self) -> None:
        sections_paths = [section for section in self.__sections]
        sections_positions_from_id = self.__generate_dict_of_sections_positions(polygon=self.__polygon)
        
        proj_utm = ccrs.UTM(self.__zone_number, southern_hemisphere=(self.__zone_letter < "N"))
        fig, plot = plt.subplots(subplot_kw={'projection': proj_utm})
        
        proportions_dict = dict()
        all_distances_list = []
        all_times_list = []
        
        self.__display_polygon(polygon=self.__polygon, map=plot, proj_utm=proj_utm)
        
        with ThreadPoolExecutor() as executor:
            tasks = [executor.submit(self.__evaluate_section, fig, plot, proportions_dict, all_distances_list, all_times_list, sections_positions_from_id[self.__get_section_id_from_path(path=section)], section, 0, [], []) for section in sections_paths]
            for task in tqdm(as_completed(tasks), total=len(tasks), miniters=1, desc="Evaluated sections", unit="sections", leave=True):
                task.done()
        for section_id in list(proportions_dict.keys()):
            self.__display_pie_chart(proportions=proportions_dict[section_id],
                            position=sections_positions_from_id[section_id],
                            plot=plot)
        if ( self.__display_results_only ):
            self.__generate_pie_chart_and_time_legends(plot=plot)
        else:
            self.__generate_markers_legends(plot=plot)
        plt.show()
        print("")
        print("Average of all distances:", self.__compute_average( all_distances_list ), "m")
        print("Average of all times:", self.__compute_average( all_times_list ), "sec")