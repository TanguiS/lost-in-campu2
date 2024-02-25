from pathlib import Path
from shapely.geometry import Point
from typing import Dict, Union
        
def get_image_name(image_path : Path) -> str:
    return image_path.stem

def extract_geo_datas(image_tags : Dict[str, any]) -> Point:
    lat = convert_location(image_tags["GPS GPSLatitude"])
    lon = convert_location(image_tags["GPS GPSLongitude"])
    if ( str(image_tags["GPS GPSLatitudeRef"]) != "N" ):
        lat = -lat
    if ( str(image_tags["GPS GPSLongitudeRef"]) != "E" ):
        lon = -lon
    return Point(round(lon, 5), round(lat, 5))

def convert_location(value : any ) -> float:
    d = value.values[0].num / value.values[0].den
    m = value.values[1].num / value.values[1].den
    s = value.values[2].num / value.values[2].den
    return d + (m / 60.0) + (s / 3600.0)

def extract_heading_datas(image_tags : Dict[str, any], deg_precision : float = 10.0) -> int:
    deg_to_north = eval(str(image_tags['GPS GPSImgDirection']))
    if deg_to_north <= deg_precision or deg_to_north >= 360.0 - deg_precision:
        return 0
    return int(deg_to_north)

def extract_size_datas(image_tags : Dict[str, any]) -> Point:
    x = int(str(image_tags['EXIF ExifImageWidth']))
    y = int(str(image_tags['EXIF ExifImageLength']))
    return Point(x, y)

def get_image_ext():
    return {'.jpeg', '.jpg', '.png', '.gif', '.bmp', '.svg'}


"""_summary_
For ProProcessed Images ONLY
"""

def read_datas(image_processed_path : Path) -> Dict[str, Union[float, int]]:
    list_datas = get_image_name( image_processed_path ).split("@")
    dict_datas = dict()
    dict_datas["UTM_east"] = float(list_datas[1])
    dict_datas["UTM_north"] = float(list_datas[2])
    dict_datas["GPS_longitude"] = float(list_datas[5])
    dict_datas["GPS_latitude"] = float(list_datas[6])
    dict_datas["GPS_heading"] = int(list_datas[9])
    dict_datas["IMG_name"] = str(list_datas[7])
    return dict_datas

def read_raw_datas( image_processed_path : Path) -> list:
    return get_image_name( image_processed_path ).split("@")[1:-1]

def is_facing_north(processed_image_path : Path) ->  bool:
    return read_datas(processed_image_path)['GPS_heading'] == 0