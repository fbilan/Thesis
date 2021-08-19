import flickrapi
import geopandas
import pandas as pd
from os import path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
import urllib
import contextily as ctx
import libpysal as ps
import esda

import matplotlib.patheffects as pe
from A3_Data_Handling import df_gdf_conversion
from A6_Geometric_Functions import kde
from shapely.geometry import Polygon
from splot.esda import moran_scatterplot
from splot.esda import lisa_cluster
from sklearn.neighbors import KernelDensity
from sklearn.datasets._species_distributions import construct_grids
#from A6_Geometric_Functions import kde

# This function plots a dictionary as a histogram (usually months and integer)
def plot_dict(dictionary, title, output_name):
    plt.grid(True, linewidth=0.5, linestyle=':')
    plt.bar(dictionary.keys(), dictionary.values(), color='grey')
    plt.title(title)
    plt.xticks(rotation=30)
    plt.savefig('Plots/' + output_name + '.png', dpi = 400, bbox_inches="tight")
    plt.show()

def base_plot(ax, output_name):
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.ticklabel_format(style='plain')
    plt.tight_layout()
    ctx.add_basemap(ax = ax, crs='EPSG:2056',
                    source=ctx.providers.Stamen.TerrainBackground,
                    # 'https://tiles.wmflabs.org/hillshading/{z}/{x}/{y}.png',
                    attribution_size=5,
                    attribution="Source: flickr.com (2020),\n"
                                "Swisstopo - TLM_Strasse (2020),\n"
                                "Swisstopo - swissBOUNDARIES3D (2015)\n" +
                                "Background: Stamen.TerrainBackground")
    plt.savefig('Plots/' + output_name + '.png', dpi=500, bbox_inches="tight")
    #plt.show()

# This function plots the distance_to_line column in a histogram
def min_distance_plot(df, output_name, mode):
    if mode == 'cummulative':
        x = df
        binwidth = 1
        counts, bins = np.histogram(x, bins=range(0, 400 + binwidth, binwidth))
        cdf = np.cumsum(counts) / np.sum(counts) *100
        plt.subplots(figsize=(8, 5))
        plt.plot(
            np.vstack((bins, np.roll(bins, -1))).T.flatten()[:-2],
            np.vstack((cdf, cdf)).T.flatten()
        )
        plt.grid(axis='y', alpha=0.5)
        plt.xticks(np.arange(min(x), 400, 20), rotation=30)
        plt.yticks(np.arange(80, 102, 2))
        plt.xlabel('Distance [m]')
        plt.ylabel('Cummulative Share (%)')
        plt.savefig('Plots/' + output_name + '.png', dpi = 500)
        plt.show()

    else:
        dist = df
        binwidth = 40  # steps of 20m
        plt.hist(dist, bins=range(0, 400 + binwidth, binwidth), edgecolor='grey', color='grey')
        plt.grid(axis='y', alpha=0.5)
        plt.xticks(np.arange(min(dist), 400, 20), rotation=30)
        plt.yticks(np.arange(0, 650, 25))
        #plt.title('Histogram of Distances between Flickr Pictures and nearest Hiking Trail')
        plt.xlabel('Distance [m]')
        plt.ylabel('Count')
        plt.savefig('Plots/' + output_name + '.png', dpi = 500)
        plt.show()

def hist_automatic():
    values = [0.14, 1.7, 0.56, 6.94, 0, 0, 0, 0, 0.2, 2.44, 0.14, 2.27, 0.11, 1.85, 0, 0, 0, 0, 0.15, 2.44, 1.18, 0.77, 0.78, 0.51, 1.0, 0.65, 0, 0, 0.9, 0.59, 0.18, 0.45, 0.44, 1.11, 1.5, 3.75, 0, 0, 0.29, 0.73, 2.14, 0.67, 3.44, 1.08, 3.0, 0.94, 0, 0, 2.2, 0.69, 3.45, 0.92, 4.11, 1.09, 3.5, 0.93, 0, 0, 3.17, 0.84,0.18, 0.73, 0.44, 1.78, 4.5, 18.0, 0, 0, 0.41, 1.66,0.18, 0.53, 0.22, 0.65, 0, 0, 0, 0, 0.15, 0.43,0.73, 0.89, 0, 0, 0, 0, 0, 0, 0.44, 0.54,0.23, 0.41, 1.0, 1.79, 2.0, 3.57, 0, 0, 0.46, 0.83,0.14, 0.45, 0.22, 0.74, 0, 0, 0, 0, 0.15, 0.49,1.09, 0.77, 1.22, 0.86, 1.0, 0.7, 0.5, 0.35, 1, 0.7,0.55, 0.77, 1.0, 1.41, 2.5, 3.52, 0, 0, 0.68, 0.96,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0.05, 0.65, 0.22, 3.17, 0, 0, 0, 0, 0.07, 1.05,0.05, 0.57, 0.22, 2.78, 0, 0, 0, 0, 0.07, 0.91,3.09, 1.14, 8.33, 3.09, 5.5, 2.04, 1.5, 0.56, 4.05, 1.5,0.09, 0.53, 0, 0, 0, 0, 0, 0, 0.07, 0.43,0.86, 0.79, 0.89, 0.82, 0, 0, 1.0, 0.92, 0.83, 0.76,0.55, 0.94, 1.44, 2.49, 0, 0, 1.0, 1.72, 0.71, 1.22]
    binwidth = 0.1

    fig, ax = plt.subplots(figsize=(8,4))

    plt.ylim(0,100)
    alpha = 1
    #s = plt.Rectangle((-0.5, 0), 0.51, 75, fc='#E8F5E9', alpha=alpha)
    m = plt.Rectangle((0.01, 0), 0.67, 100, fc='#81C784', alpha=alpha)
    l = plt.Rectangle((0.68, 0), 0.4075, 100, fc='#43A047', alpha=alpha)
    xl = plt.Rectangle((1.0875, 0), 17, 100, fc='#1B5E20', alpha=alpha)
    #ax.add_patch(s)
    ax.add_patch(m)
    ax.add_patch(l)
    ax.add_patch(xl)
    #binwidth = 1
    counts, bins = np.histogram(values, bins=np.arange(0, max(values) + 0.01, step=binwidth))
    cdf = np.cumsum(counts) / np.sum(counts) *100
    plt.plot(
        np.vstack((bins, np.roll(bins, -1))).T.flatten()[:-2],
        np.vstack((cdf, cdf)).T.flatten(), color = 'black'
    )
    #plt.hist(values, bins=np.arange(0, max(values) + 0.01, step=binwidth), color='black', alpha = 0.7)
    plt.xlabel('%LF')
    plt.ylabel('Cummulative Share (%)')
    plt.show()

def kde_ces_plots(flickr_annotations_lv, text_ann_lv_clear, entlebuch_lv):
    ces_subtypes = ["Identity", "Information Board", "Information Office", "Local History", "Tradition",
                    "Traditional Architecture", "Recreational Facilities", "Signpost", "Viewpoint", "Camping", "People",
                    "Restaurant / Accommodation", "Dawn / Sunset", "Healing Powers", "Place Attachment", "Church",
                    "Summit Cross"]

    picture_annotations_lv = flickr_annotations_lv.drop_duplicates(subset=['PICTURE_ID', 'SUBTYPE'])
    picture_annotations_lv = picture_annotations_lv[['LF/CES', 'SUBTYPE', 'latitude', 'longitude', 'geometry']]
    ces_pic_annotations = picture_annotations_lv[picture_annotations_lv['LF/CES'] == 'CES']

    text_annotations_lv = text_ann_lv_clear[['LF/CES', 'SUBTYPE', 'latitude', 'longitude', 'geometry']]
    ces_text_annotations = text_annotations_lv[text_annotations_lv['LF/CES'] == 'CES']
    ces_subtype_annotations = ces_text_annotations.append(ces_pic_annotations).reset_index(drop=True, inplace=False)
    for ces in ces_subtypes:
        print('CES ' + str(ces) + " in production")
        print(len(ces_subtype_annotations))
        ces_subtypes = ces_subtype_annotations[ces_subtype_annotations['SUBTYPE'] == ces]
        print(len(ces_subtypes))
        x_mesh, y_mesh, intensity = kde(ces_subtypes, entlebuch_lv, 100, 1000)
        fig, ax = plt.subplots(figsize=(9, 9))
        #municipalities_lv.plot(ax=ax, facecolor='none', edgecolor='grey')
        entlebuch_lv.plot(facecolor='none', edgecolor='black', ax=ax)
        plt.title(str(ces) + ' (n = ' + str(len(ces_subtypes)) + ')', fontsize=14)
        #municipalities_lv.apply(
        #    lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=8, color='black',
        #                          path_effects=[pe.withStroke(linewidth=1.2, foreground="white")]), axis=1)
        cax = ax.pcolormesh(x_mesh, y_mesh, intensity, cmap='hot_r', alpha=0.6)
        cbar = plt.colorbar(cax, ticks=[np.min(intensity) + 0.1 * np.max(intensity),
                                        np.max(intensity) - np.max(intensity) * 0.1], shrink=0.75)
        cbar.ax.set_yticklabels(['Low', 'High'])
        ces = str(ces).replace(' / ', ' ')
        base_plot(ax=ax, output_name='kde_' + str(ces))
        # plt.show()

