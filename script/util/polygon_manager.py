from shapely.geometry import Point, Polygon
from pathlib import Path
from array import array as ar
from typing import List
import csv
import utm

def is_in_polygon(gps_coords : Point, polygon : Polygon) -> bool:
    return polygon.contains(gps_coords)

def read_polygon_csv(polygon_csv : Path) -> list:
    with open(polygon_csv, 'r') as csv_file:
        return list(csv.reader(csv_file))[1:]
    
def list_to_polygon(polygon_list : list, isUtm : bool = False):
    if (isUtm):
        return Polygon([(utm.from_latlon(float(lat), float(lon))[:2]) for lon, lat in polygon_list])
    return Polygon([(float(lon), float(lat)) for lon, lat in polygon_list])

def divide_polygon(polygon : Polygon, n : int) -> List[Polygon]:
    exterior = polygon.exterior.xy
    center = polygon.representative_point()
    sub_polygon = list()
    for index in range (1, n):
        print(index)
        x = ar('d', [((ext - center.x) * (index / n))+center.x for ext in exterior[0]])
        y = ar('d', [((ext - center.y) * (index / n))+center.y for ext in exterior[1]])
        sub_polygon.append(Polygon(list(zip(x, y))))
    sub_polygon.append(polygon)
    return sub_polygon

def section_polygon_grid( GPS_polygon : Polygon, grid_size_meters : int = 25 ) -> List[Polygon]:
    UTM_polygon = Polygon([(utm.from_latlon(lat, lon)[:2]) for lon, lat in list(GPS_polygon.exterior.coords)])
    UTM_exterior = UTM_polygon.exterior.xy
    UTM_min_east, UTM_max_east, UTM_min_north, UTM_max_north = (min(UTM_exterior[0]), max(UTM_exterior[0]), min(UTM_exterior[1]), max(UTM_exterior[1]))
    grid = list()
    x = UTM_min_east
    for x in range(int(UTM_min_east), int(UTM_max_east), grid_size_meters):
        for y in range(int(UTM_min_north), int(UTM_max_north), grid_size_meters):
            x_right = x + grid_size_meters
            y_top = y + grid_size_meters
            if UTM_polygon.contains(Point(x, y)) or UTM_polygon.contains(Point(x_right, y)) or UTM_polygon.contains(Point(x_right, y_top)) or UTM_polygon.contains(Point(x, y_top)):
                grid.append(Polygon([(x, y), (x_right, y), (x_right, y_top), (x, y_top), (x, y)]))
    return grid