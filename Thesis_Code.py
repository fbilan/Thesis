import matplotlib.patheffects as pe
from os import listdir
from os import path
from os.path import isfile, join
import gpxpy
import gpxpy.gpx
import contextily as ctx
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from A1_Data_Extraction import *
from A2_Visualisation import base_plot
from A2_Visualisation import min_distance_plot
from A3_Data_Handling import *
from A4_Picture_Handling import *
from A6_Geometric_Functions import *
from A7_Jaccard_Index import *



# Flickr Credentials
api_key = "6bb1751b44ffb015e3e46103bdd8a7e0"
secret_api_key = "26f97277c260faae"
flickr = flickrapi.FlickrAPI(api_key, secret_api_key)
csv_file = 'Data/' + 'flickr_points.csv'

### Load Data
# Load Base Data
entlebuch_lv = geopandas.read_file('Data/TLM_Entlebuch.shp')
entlebuch_lv = entlebuch_lv.to_crs(epsg=2056)
entlebuch_wgs = entlebuch_lv.to_crs(epsg=4326)

towns_lv = geopandas.read_file('Data/Entlebuch_Towns.shp')
towns_lv = towns_lv.to_crs(epsg=2056)
towns_wgs = towns_lv.to_crs(epsg=4326)

municipalities_lv = geopandas.read_file('Data/Entlebuch_Municipalities.shp')
municipalities_lv = municipalities_lv.to_crs(epsg=2056)
municipalities_wgs = municipalities_lv.to_crs(epsg=4326)

switzerland_lv = geopandas.read_file('Data/Switzerland_Boundaries.shp')
switzerland_lv = switzerland_lv.to_crs(epsg=2056)
switzerland_wgs = switzerland_lv.to_crs(epsg=4326)

# Hiking Trails
trails_lv = geopandas.read_file('Data/Hiking_Trails/TLM_Hiking.shp')

# Original Data Set
flickr_wgs = geopandas.read_file('Data/flickr_points.shp')
flickr_wgs = flickr_wgs.set_crs(epsg=4326)
flickr_lv = flickr_wgs.to_crs(epsg=2056)

# Filtered Data Set
flickr_40_wgs = geopandas.read_file('Data/flickr_40_summer.shp')
flickr_40_wgs = flickr_40_wgs.set_crs(epsg=4326)
flickr_40_lv = flickr_40_wgs.to_crs(epsg=2056)

# Only annotated pictures
flickr_ann_wgs = geopandas.read_file('Data/annotated_pictures.shp')
flickr_ann_wgs = flickr_ann_wgs.set_crs(epsg=4326)
flickr_ann_lv = flickr_ann_wgs.to_crs(epsg=2056)


# Text Annotations
text_ann_wgs = geopandas.read_file('Data/annotated_text.shp')
text_ann_wgs = text_ann_wgs.set_crs(epsg=4326)
text_ann_lv = text_ann_wgs.to_crs(epsg=2056)
text_ann_lv_clear = text_ann_lv[text_ann_lv['PRECISION'] == 'clear']
text_ann_lv_clear = text_ann_lv_clear.reset_index(drop=True, inplace = False)

# Text Annotations CES
text_ann_lv_clear_ces = pd.DataFrame(text_ann_lv_clear[text_ann_lv_clear['LF/CES'] == 'CES'])
text_ann_lv_clear_ces = text_ann_lv_clear_ces.reset_index(drop=True, inplace = False)
text_ann_lv_clear_ces = df_gdf_conversion(text_ann_lv_clear_ces, 'gdf')
text_ann_lv_clear_ces = text_ann_lv_clear_ces.set_crs(epsg=4326)
text_ann_lv_clear_ces = text_ann_lv_clear_ces.to_crs(epsg=2056)


# Picture Annotations with coords
flickr_annotations_wgs = geopandas.read_file('Data/Flickr_Annotations.shp')
flickr_annotations_wgs = flickr_annotations_wgs.set_crs(epsg=4326)
flickr_annotations_lv = flickr_annotations_wgs.to_crs(epsg=2056)
flickr_ces_lv = flickr_annotations_lv[flickr_annotations_lv['LF/CES'] == 'CES']
flickr_ces_lv = flickr_ces_lv.reset_index(drop=True, inplace = False)


# Combined Annotations for Manual
combined_manual_lv = pd.DataFrame(flickr_ces_lv).append(pd.DataFrame(text_ann_lv_clear_ces), ignore_index = True)
combined_manual_lv = combined_manual_lv.reset_index(drop = True, inplace = False)
id = list(range(len(combined_manual_lv)))
combined_manual_lv['id'] = id
combined_manual_lv = df_gdf_conversion(combined_manual_lv, 'gdf')
combined_manual_lv = combined_manual_lv.set_crs(epsg=4326)
combined_manual_lv = combined_manual_lv.to_crs(epsg=2056)


# Combined Annotations for Hotspot
combined_automatic_lv = pd.DataFrame(flickr_ann_lv[['owner', 'latitude', 'longitude']])
text_ann_temp = pd.DataFrame(text_ann_lv_clear[['FILE', 'latitude', 'longitude']])
text_ann_temp = text_ann_temp.rename(columns={"FILE": "owner"})
combined_automatic_lv = combined_automatic_lv.append(text_ann_temp, ignore_index = True)
combined_automatic_lv = combined_automatic_lv.reset_index(drop = True, inplace = False)
id = list(range(len(combined_automatic_lv)))
combined_automatic_lv['id'] = id
combined_automatic_lv = df_gdf_conversion(combined_automatic_lv, 'gdf')
combined_automatic_lv = combined_automatic_lv.set_crs(epsg=4326)
combined_automatic_lv = combined_automatic_lv.to_crs(epsg=2056)




# Flickr Hotspots
hotspots_flickr_lv = geopandas.read_file('Data/hotspots_dissolved_Flickr Pictures.shp')
hotspots_flickr_lv = hotspots_flickr_lv.set_crs(epsg=2056)
hotspots_flickr_wgs = hotspots_flickr_lv.to_crs(epsg=4326)

# Text Hotspots
hotspots_text_lv = geopandas.read_file('Data/hotspots_dissolved_Text Data.shp')
hotspots_text_lv = hotspots_text_lv.set_crs(epsg=2056)
hotspots_text_wgs = hotspots_text_lv.to_crs(epsg=4326)

# Combined Hotspots
hotspots_combined_lv = geopandas.read_file('Data/hotspots_dissolved_Combined.shp')
hotspots_combined_lv = hotspots_combined_lv.set_crs(epsg=2056)
hotspots_combined_wgs = hotspots_combined_lv.to_crs(epsg=4326)

hotspots_union = hotspots_text_lv.append(hotspots_flickr_lv)
hotspots_union = hotspots_union.dissolve()

bbox = bbox_coords(entlebuch_wgs)
photo_list = flickr.photos.search(api_key=api_key, bbox=bbox, format='parsed-json', has_geo=1,
                                  min_taken_date='1000-01-01', extras=',geo,tags,date_taken,url_c')
photos = photo_list['photos']


def create_flickr_data():
    df = create_photos_df(api_key, secret_api_key, photos, bbox)
    if path.exists(csv_file):
        df2 = pd.read_csv(csv_file, delimiter=',')
        df = df.append(df2, ignore_index=True)
        df = df.astype('str') # https://stackoverflow.com/questions/46489695/drop-duplicates-not-working-in-pandas Wizhi -> remove of duplicates works
        df.drop_duplicates(subset = 'id', inplace=True)
        gdf = df_gdf_conversion(df, 'gdf')
        gdf = clipping(gdf, entlebuch_lv)
        df = df_gdf_conversion(gdf, 'df')
        if df.shape[0] > df2.shape[0]: # if old file + new file is larger than the old file store the old + new file
            print(df.shape, ' is the new File and overwrote the old file, ', df2.shape)
            save(df, 'flickr_points')
        else:
            print('Old File is larger', df2.shape)
    else:
        df.drop_duplicates(inplace=True)
        gdf = df_gdf_conversion(df, 'gdf')
        gdf = clipping(gdf, entlebuch_lv)
        df = df_gdf_conversion(gdf, 'df')
        save(df, 'flickr_points')
        print("New file has been created with the first retrieval")

    flickr_lv2 = add_times(flickr_lv)
    flickr_lv2 = min_distance(flickr_lv2, trails_lv)
    min_distance_plot(flickr_lv2, 'Flickr_Distance', mode = 'cummulative')

    #Filter for May - October and within 40m to hiking trail
    flickr_lv_summer = flickr_lv2[(flickr_lv2['month'] >= 5) & (flickr_lv2['month'] < 11) & (flickr_lv2['year'] >= 1999) & (flickr_lv2['distance_to_line'] < 40)]

    save(flickr_lv_summer, 'flickr_40_summer')

    download_pictures(flickr_40_lv)

def flickr_data_analysis():
    # Overview Plot with Municipalities
    color = {0: 'gold', 1: 'red', 2: 'blue'}
    ht_type = ['Municipal Boundary', 'Hiking Trail', 'Mountain Hiking Trail', 'Alpine Hiking Trail']
    ax = municipalities_lv.plot(facecolor='none', edgecolor='black', figsize=(8, 8))
    trails_lv.plot(ax = ax,color = trails_lv['WANDERWEGE'].map(color), alpha = 0.5)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    #plt.title('Municipalities and Hiking Trails\nin the UNESCO Biosphere Entlebuch', fontsize = 13)
    plt.tight_layout()
    municipalities_lv.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize = 10, color = 'black', path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]), axis=1)

    custom_legend = [plt.Line2D([0], [0], color='black', linewidth=1, linestyle='-'),#plt.Line2D([0], [0], marker='o', markersize = 3, linestyle = 'None', color='black', label='Flickr Picture'),
                     plt.Line2D([0], [0], color=color[0], linewidth=1, linestyle='-'),
                     plt.Line2D([0], [0], color=color[1], linewidth=1, linestyle='-'),
                     plt.Line2D([0], [0], color=color[2], linewidth=1, linestyle='-')
                    ]
    ax.legend(custom_legend, ht_type, loc='upper left', fontsize = 8)



    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.ticklabel_format(style='plain')
    ctx.add_basemap(ax = ax, crs='EPSG:2056',
                    source=ctx.providers.Stamen.TerrainBackground,
                    # 'https://tiles.wmflabs.org/hillshading/{z}/{x}/{y}.png',
                    attribution_size=5,
                    attribution="Source: Swisstopo - TLM_Strasse (2020),\n"
                                "Swisstopo - swissBOUNDARIES3D (2015)\n" +
                                "Background: Stamen.TerrainBackground")


    # Inset Map
    ax2 = inset_axes(ax, width = '32%', height = '30%', loc = 'lower right', borderpad = 0.5)
    switzerland_lv.boundary.plot(ax = ax2, color = 'black', linewidth=0.5)
    entlebuch_lv.plot(ax = ax2, facecolor = 'None', edgecolor = 'black', linewidth = 0.5)
    ax2.xaxis.set_major_locator(ticker.NullLocator())
    ax2.yaxis.set_major_locator(ticker.NullLocator())
    ax2.set_xlabel('Location in Switzerland', fontsize = 7, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
    ax2.xaxis.set_label_position('top')
    ctx.add_basemap(ax2, crs=trails_lv.crs.to_string(),
                    source=ctx.providers.Stamen.TerrainBackground,
                    attribution_size=5,
                    attribution= False)
    plt.savefig('Plots/' + 'ube_overview' + '.png', dpi=800)
    plt.show()


    # All Flickr Pictures Overview Plot
    ht_type = ['Flickr Picture', 'Hiking Trail', 'Mountain Hiking Trail', 'Alpine Hiking Trail']
    ax = entlebuch_lv.plot(facecolor='none', edgecolor='black', figsize=(8, 10))
    trails_lv.plot(ax=ax, color=trails_lv['WANDERWEGE'].map(color), alpha=0.5)
    flickr_lv.plot(ax=ax, color='black', markersize=5, marker='o')
    towns_lv.apply(lambda x: ax.annotate(text=x.town_name, xy=x.loc['geometry'].coords[0], ha='center',
                                         path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]), axis=1)
    custom_legend = [
        plt.Line2D([0], [0], marker='.', markersize=3, linestyle='None', color='black', label='Flickr Picture'),
        plt.Line2D([0], [0], color=color[0], linewidth=1, linestyle='-'),
        plt.Line2D([0], [0], color=color[1], linewidth=1, linestyle='-'),
        plt.Line2D([0], [0], color=color[2], linewidth=1, linestyle='-')
        ]
    ax.legend(custom_legend, ht_type, loc='lower right', fontsize=8)
    base_plot(ax, 'Flickr_Hiking')



    # Flickr Plot with relevant pictures
    ax = entlebuch_lv.plot(facecolor='none', edgecolor='black', figsize=(8, 10))
    trails_lv.plot(ax=ax, color=trails_lv['WANDERWEGE'].map(color), alpha=0.5)
    flickr_40_lv.plot(ax=ax, color='black', markersize=5, marker='o')
    towns_lv.apply(lambda x: ax.annotate(text=x.town_name, xy=x.loc['geometry'].coords[0], ha='center',
                                         path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]), axis=1)
    custom_legend = [
        plt.Line2D([0], [0], marker='.', markersize=3, linestyle='None', color='black', label='Flickr Picture'),
        plt.Line2D([0], [0], color=color[0], linewidth=1, linestyle='-'),
        plt.Line2D([0], [0], color=color[1], linewidth=1, linestyle='-'),
        plt.Line2D([0], [0], color=color[2], linewidth=1, linestyle='-')
        ]
    ax.legend(custom_legend, ht_type, loc='lower right', fontsize=8)
    base_plot(ax, 'Flickr_Hiking_40_summer')

    # Calculate PUD
    pud = pud_per_month(flickr_40_lv)

    # Calculate Number of Pictures per Month
    months = group_pictures_count(flickr_40_lv, 'month')
    months_name = {}
    for key in sorted(months):
        month = month_to_string(key)
        months_name[month] = months[key]

    # Calculate Number of Pictures per Year
    pic_year = group_pictures_count(flickr_40_lv, 'year')
    years = {}
    for key in sorted(pic_year):
        if key > 1999:
            years[key] = pic_year[key]

    # # plots
    # plt.xticks(range(2000, 2021))
    # plot_dict(years, 'Number of Pictures per Year from 2000 (May - Oct)', 'Pictures_per_Year_40_summer')
    # #plt.ylim((0, 1500))
    # plot_dict(months_name, 'Absolute Number of Pictures per Month within 40m', 'Pictures_per_Month_40_summer')
    # #plt.ylim((0, 50))
    # plot_dict(pud, 'Number of Photo-User-Days (PUD) per Month within 40m', 'PUD_40_summer')

    get_contributors_dict(flickr_40_lv)

def create_hotspots(point_file_lv, boundary, grid_size, count, data):
    x = point_file_lv['geometry'].x
    y = point_file_lv['geometry'].y
    if data == 'Flickr Pictures':
        contributor = point_file_lv['owner']
    elif data == 'Text Data':
        contributor = point_file_lv['FILE']
    elif data == 'Combined':
        contributor = point_file_lv['owner']
    else:
        print('Wrong data')
    id = point_file_lv['id']

    bbox = boundary.total_bounds

    x_min = bbox[0]-grid_size
    x_max = bbox[2]+grid_size
    y_min = bbox[1]-grid_size
    y_max = bbox[3]+grid_size

    # CONSTRUCT GRID
    x_grid = np.arange(x_min, x_max, grid_size)
    y_grid = np.arange(y_min, y_max, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    # PROCESSING
    # Fill grid with two lists
    count_list = []
    for j in range(len(y_grid)):
        count_row = [[[], []] for a in range(len(x_grid))]
        count_list.append(count_row)

    for i in range(len(x)):
        x_row = round((x[i] - x_min) / grid_size)
        y_row = round((y[i] - y_min) / grid_size)
        #print(x_grid[x_row], y_grid[y_row])
        count_list[y_row][x_row][0].append(id[i])
        if contributor[i] in count_list[y_row][x_row][1]:
            pass
        else:
            count_list[y_row][x_row][1].append(contributor[i])

    # HEATMAP OUTPUT DEPENDING ON INPUT
    output_list = []
    if count == 'Contributors':
        for j in range(len(y_grid)):
            count_row = [[0] for a in range(len(x_grid))]
            output_list.append(count_row)
            for i in range(len(x_grid)):
                output_list[j][i] = len(count_list[j][i][1])
    elif count == 'Pictures':
        for j in range(len(y_grid)):
            count_row = [[0] for a in range(len(x_grid))]
            output_list.append(count_row)
            for i in range(len(x_grid)):
                output_list[j][i] = len(count_list[j][i][0])

    count_array = np.array(output_list)
    count_array_plot = np.ma.masked_array(count_array, count_array < 1)


    raster_polygon_plot = convert_raster_to_polygon(x_mesh, y_mesh, count_array_plot, grid_size)

    raster_polygon = convert_raster_to_polygon(x_mesh, y_mesh, count_array, grid_size)
    hh_gdf = moran_local_hh(raster_polygon) # extract hotspot

    # Convert geoseries to gdf
    hh_gdf = geopandas.GeoDataFrame(geometry=geopandas.GeoSeries(hh_gdf))
    test = 0
    if count == 'Contributors':
        if data == 'Flickr Pictures':
            # Assign cluster number based on cell position
            cluster = [1,1,1,1,1,1,1,1,2,1,1,1,1,1,2,2,2,1,1,1,1,2,2,1,1,3,2,3,4,5,5,5,5,5,6,6,6,6,7,8]
            try:
                hh_gdf['CLUSTER'] = cluster
                hh_gdf = hh_gdf.dissolve(by='CLUSTER')
                hh_gdf['ID'] = range(1, 9)
                hh_gdf.to_file('Data/{0}_{1}.shp'.format('hotspots_dissolved',data))
                test = 1
            except:
                pass
        elif data == 'Text Data':
            # Assign cluster number based on cell position
            cluster = [1,2,1,1,1,3,3,1,3,3,3,3,3,3,3,4,4,6,6,5,6,7,8,7,8,8]
            try:
                hh_gdf['CLUSTER'] = cluster
                hh_gdf = hh_gdf.dissolve(by='CLUSTER')
                hh_gdf['ID'] = range(1, 9)
                hh_gdf.to_file('Data/{0}_{1}.shp'.format('hotspots_dissolved',data))
                test = 1
            except:
                pass
        elif data == 'Combined':
            # Assign cluster number based on cell position
            cluster = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3,
                       3, 4, 5, 5, 6, 7, 7, 7, 7]

            try:
                hh_gdf['CLUSTER'] = cluster
                hh_gdf = hh_gdf.dissolve(by='CLUSTER')
                hh_gdf['ID'] = range(1, 8)
                hh_gdf.to_file('Data/{0}_{1}.shp'.format('hotspots_dissolved',data))
                test = 1
            except:
                try:
                    cluster = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                               3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8]
                    hh_gdf['CLUSTER'] = cluster
                    hh_gdf = hh_gdf.dissolve(by='CLUSTER')
                    hh_gdf['ID'] = range(1, 9)
                    hh_gdf.to_file('Data/{0}_{1}.shp'.format('hotspots_dissolved', data))
                    test = 1
                except:
                    try:
                        cluster = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2,
                                   2, 2, 2, 3, 3, 4, 5, 6, 6, 7, 8, 8, 8]
                        hh_gdf['CLUSTER'] = cluster
                        hh_gdf = hh_gdf.dissolve(by='CLUSTER')
                        hh_gdf['ID'] = range(1, 9)
                        hh_gdf.to_file('Data/{0}_{1}.shp'.format('hotspots_dissolved', data))
                        test = 1
                    except:
                        try:
                            cluster = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2,
                                       2, 2, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8]
                            hh_gdf['CLUSTER'] = cluster
                            hh_gdf = hh_gdf.dissolve(by='CLUSTER')
                            hh_gdf['ID'] = range(1, 9)
                            hh_gdf.to_file('Data/{0}_{1}.shp'.format('hotspots_dissolved', data))
                            test = 1
                        except:
                            pass

        else:
            print('Wrong data')

    # Plot count per cell
    ax = municipalities_lv.plot(facecolor='none', edgecolor='grey', figsize=(9, 9))
    entlebuch_lv.plot(facecolor='none', edgecolor='black', ax = ax)
    municipalities_lv.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize = 8, color = 'black', path_effects=[pe.withStroke(linewidth=1.2, foreground="white")]), axis=1)
    raster_polygon_plot.plot(ax = ax, column='value', cmap = 'hot_r', edgecolor = 'black', legend = True, alpha = 0.6)
    base_plot(ax = ax, output_name=data + '_total_number_' + str(grid_size) + '_' + count)
    plt.show()

    # Plot HH clusters
    ax = municipalities_lv.plot(facecolor='none', edgecolor='grey', figsize=(9, 9))
    entlebuch_lv.plot(facecolor='none', edgecolor='black', ax = ax)
    municipalities_lv.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize = 8, color = 'black', path_effects=[pe.withStroke(linewidth=1.2, foreground="white")]), axis=1)
    hh_gdf.plot(ax=ax, facecolor='red', edgecolor='black', alpha = 0.5)
    hotspot = ['Hotspot']
    #custom_legend = [mpatches.Patch(facecolor='red', edgecolor='black', alpha = 0.5, linewidth=1, linestyle='-')]
    #ax.legend(custom_legend, hotspot, loc='lower right', facecolor = 'white', fontsize=12)
    if count == 'Contributors':
        try:
            hh_gdf.apply(lambda x: ax.annotate(text=x.ID, xy=x.geometry.centroid.coords[0], ha='center', fontsize = 15, color = 'black', path_effects=[pe.withStroke(linewidth=1.2, foreground="white")]), axis=1)
        except:
            pass
    base_plot(ax=ax, output_name=data + '_extracted_hotspots_queen_' + str(grid_size) + '_' + count)
    plt.show()

    # KDE Plot
    x_mesh, y_mesh, intensity = kde(point_file_lv, boundary, 100, 1000)
    fig, ax = plt.subplots(figsize=(9, 9))
    municipalities_lv.plot(ax=ax, facecolor='none', edgecolor='grey')
    entlebuch_lv.plot(facecolor='none', edgecolor='black', ax=ax)
    municipalities_lv.apply(
        lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=8, color='black',
                              path_effects=[pe.withStroke(linewidth=1.2, foreground="white")]), axis=1)
    cax = ax.pcolormesh(x_mesh, y_mesh, intensity, cmap='hot_r', alpha=0.6)
    cbar = plt.colorbar(cax, ticks=[np.min(intensity) + 0.1 * np.max(intensity),
                                    np.max(intensity) - np.max(intensity) * 0.1], shrink=0.75)
    cbar.ax.set_yticklabels(['Low', 'High'])
    base_plot(ax=ax, output_name='kde_' + data)
    plt.show()
    return test

def kde_gpx_files(municipalities_lv, entlebuch_lv):
    folder = 'Data/Base Data/Tours/'
    gpx_list = [f for f in listdir(folder) if isfile(join(folder, f))]
    gpx_points = pd.DataFrame()
    lat = []
    lon = []
    contr = []
    for file in gpx_list:
        if file.endswith('.gpx'):
            file = folder + str(file)
            gpx_file = open(file, 'r')
            gpx = gpxpy.parse(gpx_file)
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        lat.append(point.latitude)
                        lon.append(point.longitude)
                        contr.append(file)
            #print(str(file) + ' added')

    gpx_points['latitude'] = lat
    gpx_points['longitude'] = lon

    gpx_gdf = df_gdf_conversion(gpx_points, 'gdf')
    gpx_gdf = gpx_gdf.set_crs(epsg=4326)
    gpx_gdf = gpx_gdf.to_crs(epsg=2056)
    gpx_gdf = geopandas.clip(gpx_gdf, entlebuch_lv)
    print(len(gpx_gdf))
    x_mesh, y_mesh, intensity = kde(gpx_gdf, entlebuch_lv, 50, 100)
    fig, ax = plt.subplots(figsize=(9, 9))
    municipalities_lv.plot(ax=ax, facecolor='none', edgecolor='grey')
    entlebuch_lv.plot(facecolor='none', edgecolor='black', ax=ax)
    municipalities_lv.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=8, color='black',
                              path_effects=[pe.withStroke(linewidth=1.2, foreground="white")]), axis=1)
    cax = ax.pcolormesh(x_mesh, y_mesh, intensity, cmap='hot_r', alpha=0.6)
    cbar = plt.colorbar(cax, ticks=[np.min(intensity) + 0.1 * np.max(intensity),
                                        np.max(intensity) - np.max(intensity) * 0.1], shrink=0.75)
    cbar.ax.set_yticklabels(['Low', 'High'])

    base_plot(ax=ax, output_name='kde_gpx_files')
    plt.show()

# This function calculates the annotation frequency for three activities
def boxplot_annotation_freq():
    w = {"W_Bramboden-Napf": 7,
         "W_Brienzer Rothorn - Eisee - Schoenebode": 6.1,
         "W_Emmenuferweg Schuepfheim-Wolhusen": 16.6,
         "W_Emmenuferweg Soerenberg-Schuepfheim": 22.1,
         "W_Entlebuch - Rengg - Ebnet": 12.1,
         "W_Escholzmatt - Turner": 10.4,
         "W_Fluehli - Wasserfall Chessiloch": 3.8,
         "W_Fuerstein - Wanderung zwischen Waldemme und Sarnersee": 8.8,
         "W_Fuerstein Langis": 12,
         "W_Gastronomische Rundwanderung Soerenberg": 8.9,
         "W_Glaubenbielen - Sattelpass - Seewen - Staeldili - Fluehli": 20.2,
         "W_Glaubenbielen - Soerenberg": 9.4,
         "W_Hirsegg - Schrattenfluh - Rossweid - Soerenberg": 18,
         "W_Hoehenroute Langis-Pilatus": 22.2,
         "W_Hoehenweg Entlebuch Emmental - Etappe 1 Doppleschwand - Obstaldenegg": 10,
         "W_Holzwegen - Napf": 5.1,
         "W_Im Wanderschuh zum Aelplerrendez-vous - Etappe 1": 10.8,
         "W_Im Wanderschuh zum Aelplerrendez-vous - Etappe 2": 8.3,
         "W_Marbachegg - Bumbach - Marbach": 14.3,
         "W_Marbachegg - Huernli - Marbach": 14.5,
         "W_Marbachegg - Kemmeriboden": 8.1,
         "W_Marbacher Genusstour": 3.4,
         "W_Menzberg - Napf - Huebeli b Hergiswil": 17.1,
         "W_Moorlandschaftspfad Habkern - Soerenberg": 16.8,
         "W_Moorlandschaftspfad Hilferenpass": 16.3,
         "W_Moorlandschaftspfad Klein Entlen": 21.3,
         "W_Moorlandschaftspfad Moorlandschaft Glaubenberg 1": 13.9,
         "W_Moorlandschaftspfad Moorlandschaft Glaubenberg 2": 11.3,
         "W_Moor-Rundweg Rossweid-Salwiden": 10,
         "W_Rossweid - Blattenegg": 5.3,
         "W_Rossweid - Kemmeribodenbad": 9.2,
         "W_Rundwanderung Gfellen-Schimbrig": 14.2,
         "W_Rundwanderung Heiligkreuz-First": 7.6,
         "W_Rundwanderung Schuepfheim- Farnere": 12.7,
         "W_Rundweg Marbachegg": 1.8,
         "W_Rundweg zur Kneippanlage in Fluehli": 4.3,
         "W_Salwideli - Arniberg - Tannigsbode - Salwideli": 16.1,
         "W_Soerenberg - Satz - Haglere - Mittlist Gfael - Soerenberg": 9.3,
         "W_Steinbock-Trek Brienzer Rothorn 1 Etappe": 7.5,
         "W_Steinbock-Trek Brienzer Rothorn 2 Etappe": 14.5,
         "W_Ueber den Schlierengrat": 10.4,
         "W_Wanderplausch und Biergenuss in Marbach": 3,
         "W_Wiggen – Marbach": 6,
         "W_Wiggen - Wachthubel - Marbach": 14.9,
         "W_Wiss Emme Weg - Escholzmatt-Schuepfheim": 9.1}
    tw = {"TW_Abenteuerpfad Marbach - Sagenhaftes Gezwitscher": 3.8,
          "TW_Emeritenweg": 3.7,
          "TW_Erlebnis Energie Entlebuch": 13,
          "TW_Geo-Pfad Escholzmatt": 11.5,
          "TW_Glasereipfad": 14.3,
          "TW_Koehlerweg Romoos": 11,
          "TW_Kulturweg Schuepfheim": 3.7,
          "TW_Maerchenweg Wurzilla - Lerne das Tannenwurzelkind kennen": 1.4,
          "TW_Moorpfad Mettilimoos": 14.1,
          "TW_Schutzwaldpfad Heiligkreuz": 1.9,
          "TW_Seelensteg Heiligkreuz - Ein Ort der Kraft": 1.2,
          "TW_Sonnentauweg Soerenberg": 1.4,
          "TW_Zyberliland Romoos - der abenteuerliche Baergmandli": 5.6}
    mb = {"MB_Aentlibuecher-Tour": 34,
          "MB_Choehler-Tour": 16.3,
          "MB_Clientis Flowtrail Marbachegg": 3.9,
          "MB_Farneren-Tour": 28,
          "MB_Kleiner Susten-Tour": 11.2,
          "MB_Marbacher Panoramarunde": 22.1,
          "MB_Napfbergland-Tour": 37,
          "MB_Rund um die Schrattenfluh": 52.1,
          "MB_Sattelpass Seewenegg": 39.1,
          "MB_Schimbrig-First-Tour": 31.4,
          "MB_Schuepfheimer Panoramatour": 26.7}

    densities = []
    densities.append(annotations_per_km(w, text_ann_lv))
    densities.append(annotations_per_km(tw, text_ann_lv))
    densities.append(annotations_per_km(mb, text_ann_lv))

    plt.figure(figsize=(6, 4))
    plt.boxplot(densities)
    # plt.xlabel('Activity')
    plt.ylabel('Frequency [Annotations/km]')
    plt.xticks([1, 2, 3], ['Hiking', 'Themed Trail', 'Mountain Biking'])
    plt.tight_layout()
    plt.show()

# This function runs all important functions in this thesis
def main():
    # This is to create the Flickr dataset
    # create_flickr_data() # run this multiple times
    # flickr_data_analysis()

    # #This part loads the annotations, edits them and returns plots of their counts
    # #These two functions take the downloaded CSV files and merge them together and edit them
    # edit_tour_annotations()
    # edit_picture_annotations(flickr_40_lv)

    # #Extract barcharts for all annotations
    # extract_numbers_tours(add_title='total')
    # extract_numbers_pictures(feature='Landscape Features', add_title='total', df = "Data/Annotations/picture_annotations_merged.csv")
    # extract_numbers_pictures(feature='Cultural Ecosystem Services', add_title='total', df = "Data/Annotations/picture_annotations_merged.csv")

    # #Calculate Jaccard-Indexes for Annotations of fellow geography student
    #jaccard_index()

    # Manual CES Detection
    #KDE_Plot('CES_Text', text_ann_lv_clear_ces, entlebuch_lv, municipalities_lv)
    #KDE_Plot('CES_Picture', flickr_ces_lv, entlebuch_lv, municipalities_lv)
    #KDE_Plot('CES_Combined', combined_manual_lv, entlebuch_lv, municipalities_lv)


    # #This part runs the hotspot analysis aka Automatic CES Detection
    # create_hotspots(point_file_lv = flickr_ann_lv, boundary = entlebuch_lv, grid_size = 1000, count = 'Contributors', data = 'Flickr Pictures')
    # create_hotspots(point_file_lv = flickr_ann_lv, boundary = entlebuch_lv, grid_size = 1000, count = 'Pictures', data = 'Flickr Pictures')
    # create_hotspots(point_file_lv = text_ann_lv_clear, boundary = entlebuch_lv, grid_size = 1000, count = 'Contributors', data = 'Text Data')
    # create_hotspots(point_file_lv = text_ann_lv, boundary = entlebuch_lv, grid_size = 1000, count = 'Pictures', data = 'Text Data')

    # create_hotspots(point_file_lv = combined_ann_lv, boundary = entlebuch_lv, grid_size = 1000, count = 'Contributors', data = 'Combined')
    # create_hotspots(point_file_lv = text_ann_lv_clear[text_ann_lv_clear['LF/CES'] == 'CES'], boundary = entlebuch_lv, grid_size = 1000, count = 'Contributors', data = 'Text Data')

    # #This part calculates the jaccard-indexes between the different hotspots rasters
    # jaccard_index_poly(hotspots_text_lv, hotspots_flickr_lv)
    # jaccard_index_poly(hotspots_text_lv, hotspots_combined_lv)
    # jaccard_index_poly(hotspots_flickr_lv, hotspots_combined_lv)
    # jaccard_index_poly(hotspots_text_lv, hotspots_union)
    # jaccard_index_poly(hotspots_flickr_lv, hotspots_union)
    # jaccard_index_poly(hotspots_combined_lv, hotspots_union)


    #extract_automatic_annotations(flickr_ann_lv, text_ann_lv_clear, hotspots_flickr_lv, 'Pictures')
    #extract_automatic_annotations(flickr_ann_lv, text_ann_lv_clear, hotspots_text_lv, 'Text')
    #extract_automatic_annotations(flickr_ann_lv, text_ann_lv_clear, hotspots_combined_lv, 'Combined')

    #extract_manual_annotations(flickr_annotations_lv, text_ann_lv_clear, 'Pictures', 20)
    #extract_manual_annotations(flickr_annotations_lv, text_ann_lv_clear, 'Text', 20)
    #extract_manual_annotations(flickr_annotations_lv, text_ann_lv_clear, 'Combined', 20)

    # #Random plots and analyses
    #LF_CES_Distances = min_distance(text_ann_lv[text_ann_lv['LF/CES'] == 'LF'], text_ann_lv[text_ann_lv['LF/CES'] == 'CES'])
    #min_distance_plot(LF_CES_Distances, 'LF CES Cum Distance', mode = 'cummulative')
    #hist_automatic()
    #kde_ces_plots(flickr_annotations_lv, text_ann_lv_clear, entlebuch_lv):
    #kde_gpx_files(municipalities_lv, entlebuch_lv)
    #count_per_contributor(flickr_40_lv)
    #boxplot_annotation_freq()
    pass

main()
