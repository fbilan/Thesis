import flickrapi
from A3_Data_Handling import *


# This part retrieves the pictures from the Flickr API and adds Pictures to the dataset and stores it as .csv and .shp

# This function is used to access the Flickr API and store all the data in a df
def create_photos_df(api_key, secret_api_key, photo, bbox):
    flickr = flickrapi.FlickrAPI(api_key, secret_api_key)
    df = pd.DataFrame()
    for page in range(1, photo['pages'] + 1):
        photo_list = flickr.photos.search(page=page, api_key=api_key, bbox=bbox, format='parsed-json', has_geo=1,
                                          min_taken_date='1000-01-01', extras='user_id,geo,tags,date_taken,url_c')
        photos = photo_list['photos']
        df_page = pd.DataFrame(photos['photo'])
        df_page = df_page[['id', 'owner', 'title', 'datetaken', 'tags', 'latitude', 'longitude', 'url_c']]
        df = df.append(df_page, ignore_index=True)
    return df

# This function creates a polygon-shapefile with the municipalities of Switzerland
def extract_municipalities():
    bfs_nr = [1002, 1004, 1005, 1007, 1008, 1010]
    all_municipalities = geopandas.read_file('X:/data/swisstopo/kostenlose_geodaten/swissboundaries3d/origdata/swissBOUNDARIES3D20150211/BOUNDARIES_2015/swissBOUNDARIES3D/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_2_TLM_HOHEITSGEBIET.shp')
    municipalities = all_municipalities[(all_municipalities['BFS_NUMMER'] == 1001)]
    for nr in bfs_nr:
        municipalities = municipalities.append(all_municipalities[(all_municipalities['BFS_NUMMER'] == nr)])

    # Edit some names to look better on the map
    municipalities.at[2415, 'NAME'] = ' '
    municipalities.at[1341, 'NAME'] = 'Dopple-\nschwand'
    municipalities.at[57, 'NAME'] = 'Escholzmatt-\nMarbach '
    municipalities.to_file("Data/Entlebuch_Municipalities.shp")

# This function creates a polygon-shapefile with the boundary of Switzerland
def extract_switzerland():
    switzerland = geopandas.read_file('X:/data/swisstopo/kostenlose_geodaten/swissboundaries3d/origdata/swissBOUNDARIES3D20150211/BOUNDARIES_2015/swissBOUNDARIES3D/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_2_TLM_HOHEITSGEBIET.shp')
    switzerland = switzerland.dissolve()
    switzerland.to_file("Data/Switzerland_Boundaries.shp")

#This function creates a point-shapefile of towns in the UBE
def extract_towns():
    towns = {'Schüpfheim': [46.9519505108981, 8.016441255355074], 'Entlebuch': [46.991814845373725, 8.064705961394244],
                      'Sörenberg': [46.82167879327203, 8.034617558349444], 'Flühli': [46.883137517561636, 8.016108916757403],
                      'Escholzmatt': [46.91452387751234, 7.9352426656478166], 'Marbach': [46.85403125355661, 7.899867711894361]}
    df = pd.DataFrame(columns=['town_name', 'longitude', 'latitude'])
    for key in towns:
        df = df.append({'town_name': key, 'longitude': towns[key][1], 'latitude': towns[key][0]}, ignore_index=True)
    towns = df_gdf_conversion(df, 'gdf')
    towns = towns.set_crs(epsg=4326)
    towns = towns.to_crs(epsg=2056)
    towns.to_file("Data/Entlebuch_Towns.shp")


