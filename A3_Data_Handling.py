import geopandas
import pandas as pd
import datetime



# This function converts a df to a gdf and vice-versa depending on argument
def df_gdf_conversion(input, output):
    if output == 'gdf': # input is df -> output is gdf
        gdf = geopandas.GeoDataFrame(input, geometry=geopandas.points_from_xy(input.longitude, input.latitude))
        return gdf
    elif output == 'df': # input is gdf -> output is df
        df = pd.DataFrame(input.drop(columns='geometry'))
        return df
    else:
        print('Error at df_gdf_conversion')

# This function converts a df to a csv and shp file
def save(df, output_file):
    gpd_df = df_gdf_conversion(df, 'gdf')
    gpd_df.to_file('Data/{0}.shp'.format(output_file))
    df = df_gdf_conversion(gpd_df, 'df')
    df.to_csv('Data/{0}.csv'.format(output_file), index=False)

# Add columns to Flickr Dataset which contain information about dates
def add_times(df):
    df['datetaken'] = pd.to_datetime(df['datetaken'])
    df['date'] = df['datetaken'].dt.date.astype(str) # datetime-format cannot be exported to csv or shp
    df['hour'] = pd.DatetimeIndex(df['datetaken']).hour.astype(int)
    df['month'] = pd.DatetimeIndex(df['datetaken']).month.astype(int)
    df['year'] = pd.DatetimeIndex(df['datetaken']).year.astype(int)
    df['datetaken'] = df['datetaken'].astype(str)
    return df



# This function converts a month number (int or str) to the name of the month
def month_to_string(month_number):
    datetime_object = datetime.datetime.strptime(str(month_number), "%m")
    month_name = datetime_object.strftime("%b")
    return month_name

# This function iterates through the dataset to count the number of entries per attribute (group_by)
def group_pictures_count(dataset, group_by):
    dict = {}
    for idx, line in dataset.iterrows():
        argument = line[group_by]
        if argument in dict:
            dict[argument] = dict[argument] + 1
        else:
            dict[argument] = 1
    return dict

# This function calculates the Photo-User-Days per month
def pud_per_month(df):
    months = {}
    for index, row in df.iterrows():
        month = row['month']
        date = row['date']
        owner = row['owner']
        pud = (date, owner)
        if month in months:
            if pud in months[month]:
                pass
            else:
                months[month].append(pud)
        else:
            months[month] = [pud]
    pud = {}
    for key in sorted(months):
        puds = len(months[key])
        key = month_to_string(key)
        pud[key] = puds
    return pud



# This function calculates the number of contributors and pictures for a given point file
def get_contributors_dict(point_file):
    owners_dict = {}
    for idx, photo in point_file.iterrows():
        owner = photo['owner']
        id = photo['id']
        if owner in owners_dict:
            ids = owners_dict[owner]
            ids.append(id)
            owners_dict[owner] = ids
        else:
            owners_dict[owner] = [id]
    print('Contributors: ', len(owners_dict), '\nPhotos: ', len(point_file),
          '\nAverage: ', round(len(point_file)/len(owners_dict), 2))
    return owners_dict