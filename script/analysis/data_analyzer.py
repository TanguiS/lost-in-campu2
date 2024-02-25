from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import List
from pathlib import Path
import os
import numpy as np
from util.image_manager import read_datas
import seaborn as sns
import pandas as pd
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.feature import ShapelyFeature

import util.polygon_manager as pm
import matplotlib.pyplot as plt

class DataAnalysis:
    def __init__(self,
                 input_output_processed_folder: Path = Path("dataset/processed"),
                 polygon_csv: Path = Path("polygon.csv"),
                 dx: int = 5,
                 max_density: int = 12,
                 map_in_background:bool = True
                 ) -> None:
        self.__input_folder = input_output_processed_folder
        self.__polygon_csv = polygon_csv
        self.__square_size = dx
        self.__max_density = max_density
        self.__map_in_background = map_in_background
        
    def __add_map_background(self)->None:
        print("Fetch Map")
        polygon = pm.list_to_polygon(pm.read_polygon_csv(self.__polygon_csv), True)
        points = [(point[0], point[1]) for point in polygon.exterior.coords]
        margin = 50

        extent = [min(polygon.exterior.xy[0]) - margin, max(polygon.exterior.xy[0]) + margin, min(polygon.exterior.xy[1]) - margin, max(polygon.exterior.xy[1]) + margin]
        self.__extent = extent
        
        proj_utm = ccrs.UTM(30,southern_hemisphere=("U"<"N"))
        request = cimgt.OSM()

        fig, ax = plt.subplots(subplot_kw={'projection': proj_utm})
        ax.set_extent(extent, crs=proj_utm)

        ax.add_image(request, 19, zorder=1)

        poly_feature = ShapelyFeature([polygon], crs=proj_utm, facecolor="none", edgecolor="blue", linewidth=2)
        ax.add_feature(poly_feature, zorder=3)

        for point in points:
            plt.plot(point[0], point[1], "-ro", transform=proj_utm, zorder=4)
            
        return fig, ax

    def __load_datas(self):
        UTM_east = []
        UTM_north = []
        
        for img in self.__input_folder.rglob("*.jpg"):
            img_datas = read_datas(img)
            UTM_east.append(img_datas["UTM_east"])
            UTM_north.append(img_datas["UTM_north"])
        
        self.__image_df = pd.DataFrame({
            'UTM_Easting': UTM_east,
            'UTM_Northing': UTM_north,
        })
        
        bins_east = pd.cut(self.__image_df['UTM_Easting'], np.arange(self.__image_df['UTM_Easting'].min(), self.__image_df['UTM_Easting'].max() + self.__square_size, self.__square_size))
        bins_north = pd.cut(self.__image_df['UTM_Northing'], np.arange(self.__image_df['UTM_Northing'].min(), self.__image_df['UTM_Northing'].max() + self.__square_size, self.__square_size))
        image_counts = pd.pivot_table(self.__image_df, values='UTM_Easting', index=bins_north, columns=bins_east, aggfunc='count')
        self.__image_counts = image_counts.fillna(0)

    def show_heatmap_smooth(self):
        if self.__map_in_background:
            fig, ax = self.__add_map_background()
        else:
            fig, ax = plt.subplots()

        ax.set_title('heatmap')
        
        self.__load_datas()
        
        sns.set_style("white")

        hmax = sns.kdeplot(
            data=self.__image_df,
            x='UTM_Easting',
            y='UTM_Northing',
            cmap='Blues',
            fill=True,
            bw_adjust=.5,
            alpha=0.6,
            levels=50,
            ax=ax,
            cbar=True,
            legend=True,
            cbar_kws={"label": "2D Density"},
            zorder=2
        )
        
        hmax.collections[0].set_alpha(0)  
        
        plt.title('HeatMap', fontweight ="bold")

        plt.show()

    def show_heatmap_concrete(self):
        if self.__map_in_background:
            fig, ax = self.__add_map_background()
        else:
            fig, ax = plt.subplots()
            
        self.__load_datas()
        sns.set_style("white")

        hmax = sns.heatmap(
            data=self.__image_counts.iloc[::-1],
            annot=False, 
            cmap='Blues',
            cbar=True,
            cbar_kws={"label": "Number of Images"},
            vmin=0,
            vmax=self.__max_density,
            alpha=0.6,
            ax=ax
        )

        ax.set_yticklabels([])
        ax.set_xticklabels([])
                
        plt.title('HeatMap', fontweight ="bold")

        plt.show()
        