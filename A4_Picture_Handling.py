import geopandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib
from A3_Data_Handling import save

# This function downloads flickr pictures and saves them
def download_pictures(shapefile):
    for idx, photo in shapefile.iterrows(): # shapefile.iloc[0:len(shapefile),:].iterrows()
        url = photo['url_c']
        path = str('Data/RQ1/Pictures/' + str(idx+1) + "_" + str(photo['id']) + "_" + str(photo['owner']) + '.jpg')
        urllib.request.urlretrieve(url, path)
        print("Picture done: ", idx+1)

def edit_picture_annotations(flickr_with_coords):
    annotation_folder = 'Data/Annotations/'
    annotation_backup = annotation_folder + 'picture_annotations_backup.csv'
    annotation_file = annotation_folder + 'picture_annotations_merged.csv'
    annotation_df = pd.DataFrame()

    # Append all subdata sets to one bigger file
    for number in range(1,16):
        data = pd.read_csv(annotation_folder + 'pictures_' + str(number) + '.csv', delimiter=',')
        annotation_df = annotation_df.append(data, ignore_index=True)
        annotation_df = annotation_df.astype('str')

    annotation_df.to_csv(annotation_backup)

    fbground = []
    annotation_type = []
    owner = []
    picture_id = []
    my_id = []
    type_dict = {'Natural Landscape': ["Bedrock", "Flower / Funghi", "Forest", "Grass- and Moorland", "Lake", "Natural Landscape", "River / Creek", "Rock", "Shrub", "Snow / Ice", "Summit", "Tree", "Waterfall", "Wild Animal"],
                 'Human influenced Landscape': ["Agriculture", "Human influenced Landscape", "Infrastructure", "Livestock", "Path / Trail", "Urban"],
                 'Cultural Heritage': ["Identity", "Information Board", "Information Office", "Local History", "Tradition", "Traditional Architecture"],
                 'Recreational': ["Recreational Facilities", "Signpost", "Viewpoint"],
                 'Social': ["Camping", "People", "Restaurant / Accommodation"],
                 'Spiritual': ["Dawn / Sunset", "Healing Powers", "Place Attachment", "Church", "Summit Cross"]}

    for idx, annotation in annotation_df.iterrows():
        for name in type_dict:
            if annotation['TAGS'] in type_dict[name]:
                annotation_type.append(name)
                break

        if annotation['COMMENTS'] in ['Background', 'Foreground']:
            fbground.append(annotation['COMMENTS'])
        else:
            fbground.append('None')

        file_split = annotation['FILE'].split('_')
        my_id.append(file_split[0])
        picture_id.append(file_split[1])
        owner.append(file_split[2][:-4])

    annotation_df = annotation_df.loc[:, ['FILE', 'QUOTE_TRANSCRIPTION', 'TAGS', 'COMMENTS']]
    annotation_df['LF/CES'] = annotation_df['QUOTE_TRANSCRIPTION']
    annotation_df['TYPE'] = annotation_type
    annotation_df['SUBTYPE'] = annotation_df['TAGS']
    annotation_df['FORE-/BACKGROUND'] = fbground
    annotation_df['MY_ID'] = my_id
    annotation_df['id'] = picture_id
    annotation_df['OWNER'] = owner

    del annotation_df['QUOTE_TRANSCRIPTION']
    del annotation_df['TAGS']
    del annotation_df['COMMENTS']
    del annotation_df['FILE']

    # Join the coordinates to the annotations
    annotation_df = annotation_df.merge(flickr_with_coords[['id', 'latitude', 'longitude']])

    # drop all pictures which were not annotated
    annot_id = annotation_df['id'].drop_duplicates()
    pic_id = flickr_with_coords['id']
    liste = list(set(pic_id)-set(annot_id))
    for id in liste:
        index = flickr_with_coords[flickr_with_coords['id'] == id].index.tolist()[0]
        flickr_with_coords = flickr_with_coords.drop(index)
    save(flickr_with_coords, 'annotated_pictures')

    # Save new annotation file
    annotation_df = annotation_df.rename(columns={'id':'PICTURE_ID', 'latitude':'LATITUDE', 'longitude':'LONGITUDE'})
    annotation_df.to_csv(annotation_file)

# This function extracts the counts of subtypes (LF or CES) and then plots both of them
def extract_numbers_pictures(feature, add_title, df):
    if isinstance(df, str):
        data = pd.read_csv(df, delimiter=',')
    else:
        data = df


    subtype_dict = {}
    for idx, annotation in data.iterrows():
        if feature == 'Landscape Features':
            if annotation['LF/CES'] == 'LF':
                if annotation['SUBTYPE'] in subtype_dict:
                    subtype_list = subtype_dict[annotation['SUBTYPE']]
                    if annotation['FORE-/BACKGROUND'] == 'Foreground':
                        subtype_list[0] += 1
                    else:
                        subtype_list[1] += 1

                else:
                    subtype_list = [0, 0]
                    if annotation['FORE-/BACKGROUND'] == 'Foreground':
                        subtype_list[0] += 1
                    else:
                        subtype_list[1] += 1
                subtype_dict[annotation['SUBTYPE']] = subtype_list
        elif feature == 'Cultural Ecosystem Services':
            if annotation['LF/CES'] == 'CES':
                if annotation['SUBTYPE'] in subtype_dict:
                    subtype_list = subtype_dict[annotation['SUBTYPE']]
                    subtype_list[0] += 1
                else:
                    subtype_list = [0]
                    subtype_list[0] += 1
                subtype_dict[annotation['SUBTYPE']] = subtype_list
    order_plot_pictures(subtype_dict, feature, add_title)

# This function plots a bar chart which shows the number of subtypes of LF and CES
def order_plot_pictures(my_dict, feature, add_title):
    lf_dict = {'Natural Landscape': ["Bedrock", "Flower / Funghi", "Forest", "Grass- and Moorland", "Lake", "Natural Landscape", "River / Creek", "Rock", "Shrub", "Snow / Ice", "Summit", "Tree", "Waterfall", "Wild Animal"],
               'Human influenced Landscape': ["Agriculture", "Human influenced Landscape", "Infrastructure", "Livestock", "Path / Trail", "Urban"]}
    ces_dict = {'Cultural Heritage': ["Identity", "Information Board", "Information Office", "Local History", "Tradition", "Traditional Architecture"],
                'Recreational': ["Recreational Facilities", "Signpost", "Viewpoint"],
                'Social': ["Camping", "People", "Restaurant / Accommodation"],
                'Spiritual': ["Dawn / Sunset", "Healing Powers", "Place Attachment", "Church", "Summit Cross"]}
    subtype_dict = {}
    if feature == 'Landscape Features':
        for name in lf_dict:
            for subtype in lf_dict[name]:
                try:
                    subtype_dict[subtype] = my_dict[subtype]
                except KeyError:
                    subtype_dict[subtype] = [0,0]
            df = pd.DataFrame(columns=['Subtype', 'Foreground', 'Background'])
            for key in subtype_dict:
                df = df.append({'Subtype': key, "Foreground": subtype_dict[key][0], "Background": subtype_dict[key][1]}, ignore_index=True)
        plot_picture_lf(df, feature, add_title)
        return(subtype_dict)

    elif feature == 'Cultural Ecosystem Services':
        for name in ces_dict:
            for subtype in ces_dict[name]:
                try:
                    subtype_dict[subtype] = my_dict[subtype]
                except KeyError:
                    subtype_dict[subtype] = [0]

        df = pd.DataFrame(columns=['Subtype', 'Count'])
        for key in subtype_dict:
            df = df.append({'Subtype': key, 'Count': subtype_dict[key][0]},
                           ignore_index=True)
        plot_picture_ces(df, feature, add_title)
        return (subtype_dict)

def plot_picture_lf(df, feature, add_title):
    fig, ax = plt.subplots(figsize=(10, 8))
    df.plot.barh(x='Subtype', y=['Foreground', 'Background'], rot=40, ax=ax)
    maximum = max(max(df.Foreground), max(df.Background))
    plt.xlim(0, maximum + 0.05 * maximum)
    plt.xlabel('Count', fontsize=16)
    plt.ylabel('Subtype', fontsize=16)
    plt.xticks(fontsize=11, rotation=45)
    plt.yticks(fontsize=11, rotation=0)
    plt.gca().invert_yaxis()
    plt.xticks(range(0, maximum + 10, 10))
    plt.legend(title='Location of the Annotation:')
    plt.title(add_title, fontsize=18)
    plt.tight_layout()
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.savefig('Plots/' + 'Annotations_Pictures_' + feature + '_' + add_title + '.jpg')
    #plt.show()

def plot_picture_ces(df, feature, add_title):
    fig, ax = plt.subplots(figsize=(10, 8))
    df.plot.barh(x='Subtype', y='Count', ax=ax)
    maximum = max(df.Count)
    plt.xlim(0, maximum + 0.05 * maximum)
    plt.xlabel('Count', fontsize=16)
    plt.ylabel('Subtype', fontsize=16)
    plt.xticks(fontsize=11, rotation=45)
    plt.yticks(fontsize=11, rotation=0)
    plt.gca().invert_yaxis()
    plt.xticks(range(0, maximum + 10, 10))
    plt.title(add_title, fontsize=18)
    plt.tight_layout()
    ax.get_legend().remove()
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.savefig('Plots/' + 'Annotations_Pictures_' + feature + '_' + add_title + '.jpg')
    #plt.show()

def add_coords_to_annotations():
    flickr_ann_wgs = geopandas.read_file('Data/annotated_pictures.shp')
    flickr_ann_wgs = flickr_ann_wgs.set_crs(epsg=4326)
    flickr_coords = pd.DataFrame(flickr_ann_wgs[['id', 'latitude', 'longitude']])
    flickr_coords = flickr_coords.rename(columns={"id": "PICTURE_ID"})
    flickr_coords['PICTURE_ID'] = flickr_coords['PICTURE_ID'].astype(str)

    annotation_folder = 'Data/Annotations/'
    annotation_file = annotation_folder + 'picture_annotations_merged.csv'
    annotation_file = pd.read_csv(annotation_file, delimiter=',')

    annotation_file['PICTURE_ID'] = annotation_file['PICTURE_ID'].astype(str)
    annotation_file = annotation_file.merge(flickr_coords, on='PICTURE_ID', how='left')

    annotation_file = pd.concat([annotation_file, flickr_coords], join='outer', axis=1, sort=False)
    save(annotation_file, 'Flickr_Annotations')

def count_per_contributor(flickr_40_lv):
    contr = flickr_40_lv['owner'].drop_duplicates()
    count_list = []
    for contributor in contr:
        count = len(flickr_40_lv[flickr_40_lv['owner'] == contributor])
        count_list.append(count)
    print(np.mean(count_list), np.std(count_list), np.median(count_list), max(count_list))
    plt.figure(figsize=(4,5))
    plt.boxplot(count_list, showfliers=False, )
    plt.xlabel('Flickr Contributors')
    plt.ylabel('Count')
    plt.title('Number of Pictures per Contributor')
    plt.tight_layout()
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.show()