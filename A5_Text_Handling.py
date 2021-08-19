#from A6_Geometric_Functions import clipping
from A3_Data_Handling import df_gdf_conversion
from A3_Data_Handling import save
from A2_Visualisation import *
# This function loads the text annotations and edits some attributes and stores it as a new csv file
def edit_tour_annotations():
    activities = ['BT', 'EB', 'MB', 'TR', 'TW', 'VT', 'W', 'w_emmen']
    annotation_folder = 'Data/Annotations/'
    annotation_backup = annotation_folder + 'tour_annotations_backup.csv'
    annotation_file = annotation_folder + 'tour_annotations_merged.csv'

    # Hiking Trails for buffer of 40m
    trails_lv = geopandas.read_file('Data/Hiking_Trails/TLM_Hiking.shp')

    # Merge activities to one df
    annotation_df = pd.DataFrame()
    for activity in activities:
        data = pd.read_csv(annotation_folder + activity + '.csv', delimiter=',')
        annotation_df = annotation_df.append(data, ignore_index=True)
        annotation_df = annotation_df.astype('str')

    annotation_df = annotation_df.loc[:,['FILE', 'QUOTE_TRANSCRIPTION', 'TAGS', 'COMMENTS']]
    annotation_df.to_csv(annotation_backup)
    activity_dict = {'BT': 'Mountain Tour', 'EB': 'E-Bike', 'MB': 'Mountain Bike', 'TR': 'Trail Run', 'TW': 'Themed Trail', 'VT': 'Cycling Tour', 'W': 'Hiking Trail'}
    activity = []
    lat = []
    lon = []
    tag = []
    precision = []
    annotation_type = []
    type_dict = {'Natural Landscape': ["Bedrock", "Flower / Funghi", "Forest", "Grass- and Moorland", "Lake", "Natural Landscape", "River / Creek", "Rock", "Shrub", "Snow / Ice", "Summit", "Tree", "Waterfall", "Wild Animal"],
                 'Human influenced Landscape': ["Agriculture", "Human influenced Landscape", "Infrastructure", "Livestock", "Path / Trail", "Urban"],
                 'Cultural Heritage': ["Identity", "Information Board", "Information Office", "Local History", "Tradition", "Traditional Architecture"],
                 'Recreational': ["Recreational Facilities", "Signpost", "Viewpoint"],
                 'Social': ["Camping", "People", "Restaurant / Accommodation"],
                 'Spiritual': ["Dawn / Sunset", "Healing Powers", "Place Attachment", "Church", "Summit Cross"]}

    # Edit the df of text annotations
    for idx, annotation in annotation_df.iterrows():
        tag_list = annotation['TAGS'].split("|")
        #print(tag_list)
        tag.append(tag_list[0])
        #print(tag_list[0])
        for name in type_dict:
            if tag_list[0] in type_dict[name]:
                annotation_type.append(name)
                #print(name + '\n')

        coordinates = tag_list[-1].split(";")
        lat.append(coordinates[0])
        lon.append(coordinates[1])
        if len(tag_list) == 3:
            precision.append(tag_list[1])
        else:
            precision.append('clear')
        activity_file = annotation['FILE'].split("_")
        activity.append(activity_dict[activity_file[0]])

    annotation_df['ACTIVITY'] = activity
    annotation_df['LF/CES'] = annotation_df['COMMENTS']
    annotation_df['TYPE'] = annotation_type
    annotation_df['SUBTYPE'] = tag
    annotation_df['PRECISION'] = precision
    annotation_df['latitude'] = lat
    annotation_df['longitude'] = lon
    annotation_df['id'] = range(0, len(annotation_df))
    del annotation_df['TAGS']
    del annotation_df['COMMENTS']
    annotation_gdf = df_gdf_conversion(annotation_df, 'gdf')
    annotation_gdf = annotation_gdf.set_crs(epsg=4326)
    annotation_gdf_lv = annotation_gdf.to_crs(epsg=2056)
    trails_buff = trails_lv.buffer(distance=40)
    annotation_gdf_lv = geopandas.clip(annotation_gdf_lv, trails_buff)
    save(annotation_gdf_lv, 'annotated_text')
    #annotation_df.to_csv(annotation_file)

# This function extracts the counts of subtypes (LF and CES) and then plots both of them
def extract_numbers_tours(add_title, df = 'Data/Annotations/tour_annotations_merged.csv'):
    if isinstance(df, str):
        data = pd.read_csv(df, delimiter=',')
    #annotation_folder = 'Data/Annotations/'
    data = df
    subtype_dict = {}
    for idx, annotation in data.iterrows():
        if annotation['SUBTYPE'] in subtype_dict:
            subtype_list = subtype_dict[annotation['SUBTYPE']]
            if annotation['PRECISION'] == 'clear':
                subtype_list[0] += 1
            else:
                subtype_list[1] += 1
        else:
            subtype_list = [0, 0]
            if annotation['PRECISION'] == 'clear':
                subtype_list[0] += 1
            else:
                subtype_list[1] += 1
            subtype_dict[annotation['SUBTYPE']] = subtype_list
    order_plot_tours(subtype_dict, 'Landscape Features', add_title)
    order_plot_tours(subtype_dict, 'Cultural Ecosystem Services', add_title)

# This function plots a bar chart which shows the number of subtypes of LF and CES
def order_plot_tours(my_dict, output_name, add_title):
    lf_dict = {'Natural Landscape': ["Bedrock", "Flower / Funghi", "Forest", "Grass- and Moorland", "Lake", "Natural Landscape", "River / Creek", "Rock", "Shrub", "Snow / Ice", "Summit", "Tree", "Waterfall", "Wild Animal"],
               'Human influenced Landscape': ["Agriculture", "Human influenced Landscape", "Infrastructure", "Livestock", "Path / Trail", "Urban"]}
    ces_dict = {'Cultural Heritage': ["Identity", "Information Board", "Information Office", "Local History", "Tradition", "Traditional Architecture"],
                'Recreational': ["Recreational Facilities", "Signpost", "Viewpoint"],
                'Social': ["Camping", "People", "Restaurant / Accommodation"],
                'Spiritual': ["Dawn / Sunset", "Healing Powers", "Place Attachment", "Church", "Summit Cross"]}
    subtype_dict = {}
    if output_name == 'Landscape Features':
        for name in lf_dict:
            for subtype in lf_dict[name]:
                try:
                    subtype_dict[subtype] = my_dict[subtype]
                except KeyError:
                    subtype_dict[subtype] = [0,0]
    elif output_name == 'Cultural Ecosystem Services':
        for name in ces_dict:
            for subtype in ces_dict[name]:
                try:
                    subtype_dict[subtype] = my_dict[subtype]
                except KeyError:
                    subtype_dict[subtype] = [0,0]
    df = pd.DataFrame(columns=['Subtype', 'Clear', 'Unclear'])
    for key in subtype_dict:
        df = df.append({'Subtype': key, "Clear": subtype_dict[key][0], "Unclear": subtype_dict[key][1]}, ignore_index=True)

    plot_text(df, output_name, add_title)
    return(subtype_dict)

def plot_text(df, output_name, add_title):
    fig, ax = plt.subplots(figsize= (10,8))
    if add_title.startswith('Cluster'):
        df.plot.barh(x='Subtype', y='Clear', rot=40, ax=ax, color = 'yellowgreen')
    else:
        df.plot.barh(x='Subtype', y=['Clear', 'Unclear'], rot=40, ax=ax, color = ['yellowgreen', 'gold'])
    maximum = max(max(df.Clear), max(df.Unclear))
    plt.xlim(0, maximum+0.05*maximum)
    plt.xlabel('Count', fontsize = 16)
    plt.ylabel('Subtype', fontsize = 16)
    plt.xticks(fontsize = 11, rotation = 45)
    plt.yticks(fontsize = 11, rotation = 0)
    plt.gca().invert_yaxis()
    plt.xticks(range(0, maximum+10, 10))
    plt.legend(title = 'Ease of Geolocation of Subtype:')
    plt.title('Text Annotations: ' + add_title, fontsize = 18)
    plt.tight_layout()
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.savefig('Plots/' + 'Annotations_Tours_' + output_name + '_' + add_title + '.jpg')
    #plt.show()