from sklearn.metrics import cohen_kappa_score
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
### Picture Data

def pictures_jaccard(mode):
    annotation_folder = 'Data/Annotations/'
    #test_folder = 'Data/Base Data/Test - Pictures/20210505_Pictures_Test.csv'
    fabian_annotation_file = annotation_folder + 'picture_annotations_merged.csv'
    fabienne_annotation_file = annotation_folder + 'picture_annotations_fabienne.csv'
    fabian_data = pd.read_csv(fabian_annotation_file, delimiter=',')
    fabian_data['SUBTYPE'] = fabian_data['SUBTYPE'].str.lower()
    fabian_data['FORE-/BACKGROUND'] = fabian_data['FORE-/BACKGROUND'].str.lower()
    fabienne_data = pd.read_csv(fabienne_annotation_file, delimiter=';')
    picture_ids = fabienne_data['PICTURE_ID'].drop_duplicates()
    df = pd.DataFrame(columns=['PICTURE_ID', 'OVERLAP', 'UNION', 'J-INDEX'])
    if mode == 'without':
        for pic_id in picture_ids:
            fabian_subset = fabian_data[(fabian_data['PICTURE_ID'] == pic_id)].drop_duplicates(subset='SUBTYPE')
            fabienne_subset = fabienne_data[(fabienne_data['PICTURE_ID'] == pic_id)].drop_duplicates(subset='SUBTYPE')
            fab_list = list(fabian_subset.SUBTYPE)
            fan_list = list(fabienne_subset.SUBTYPE)
            subtype_list = []
            subtype_list.extend(fan_list)
            subtype_list.extend(fab_list)
            df_subtypes = pd.DataFrame(subtype_list, columns=['SUBTYPE'])
            union = df_subtypes.drop_duplicates()
            overlap = df_subtypes.duplicated()
            df_subtypes['OVERLAP'] = overlap
            overlap = df_subtypes[(df_subtypes['OVERLAP'] == True)]
            overlap = len(overlap)
            union = len(union)
            df = df.append({'PICTURE_ID': pic_id, 'OVERLAP': overlap, 'UNION': union, 'J-INDEX': overlap/union*100}, ignore_index=True)
    elif mode == 'with':
        for pic_id in picture_ids:
            fabian_subset = fabian_data[(fabian_data['PICTURE_ID'] == pic_id)][['SUBTYPE', 'FORE-/BACKGROUND']].drop_duplicates(subset=['SUBTYPE', 'FORE-/BACKGROUND'])
            fabienne_subset = fabienne_data[(fabienne_data['PICTURE_ID'] == pic_id)][['SUBTYPE', 'FORE-/BACKGROUND']].drop_duplicates(subset=['SUBTYPE', 'FORE-/BACKGROUND'])
            df_subtypes = fabian_subset.append(fabienne_subset)
            union = df_subtypes.drop_duplicates()
            overlap = df_subtypes.duplicated()
            df_subtypes['OVERLAP'] = overlap
            overlap = df_subtypes[(df_subtypes['OVERLAP'] == True)]
            overlap = len(overlap)
            union = len(union)
            df = df.append({'PICTURE_ID': pic_id, 'OVERLAP': overlap, 'UNION': union, 'J-INDEX': overlap/union*100}, ignore_index=True)
    else:
        print('Unexpected mode')
    return df

def pictures_kappa(mode):
    annotation_folder = 'Data/Annotations/'
    test_folder = 'Data/Base Data/Test - Pictures/20210505_Pictures_Test.csv'
    fabian_annotation_file = annotation_folder + 'picture_annotations_merged.csv'
    fabienne_annotation_file = annotation_folder + 'picture_annotations_fabienne.csv'
    fabian_data = pd.read_csv(fabian_annotation_file, delimiter=',')
    fabian_data['SUBTYPE'] = fabian_data['SUBTYPE'].str.lower()
    fabienne_data = pd.read_csv(fabienne_annotation_file, delimiter=';')
    picture_ids = fabienne_data['PICTURE_ID'].drop_duplicates()

    #picture_ids = [50429908798]#, 50359582678, 50360443562]

    ces_list = ["Identity", "Information Board", "Information Office", "Local History", "Tradition", "Traditional Architecture",
                "Recreational Facilities", "Signpost", "Viewpoint", "Camping", "People", "Restaurant / Accommodation",
                "Dawn / Sunset", "Healing Powers", "Place Attachment", "Church", "Summit Cross"]
    #ces_list = [x.lower() for x in ces_list]

    df = pd.DataFrame(columns=['PICTURE_ID', 'SUBTYPE', 'FORE-/BACKGROUND', 'TRUE', 'PRED'])

    ## With Fore- and Background
    if mode == 'with':
        for pic_id in picture_ids:
            fabian_subset = fabian_data[(fabian_data['PICTURE_ID'] == pic_id)].drop_duplicates(subset=['SUBTYPE', 'FORE-/BACKGROUND'])
            fabienne_subset = fabienne_data[(fabienne_data['PICTURE_ID'] == pic_id)].drop_duplicates(subset=['SUBTYPE', 'FORE-/BACKGROUND'])

            # add annotations made by me:
            for idx, fab_ann in fabian_subset.iterrows():
                fab_subtype = fab_ann['SUBTYPE']
                fab_ground = fab_ann['FORE-/BACKGROUND']

                # check if subtype is ces -> if true fore-/background is irrelevant
                if fab_subtype in ces_list:
                    fan_list = fabienne_subset[(fabienne_subset['SUBTYPE'] == fab_subtype)]
                else:
                    fan_list = fabienne_subset[(fabienne_subset['SUBTYPE'] == fab_subtype) & (fabienne_subset['FORE-/BACKGROUND'] == fab_ground)]

                # if annotation was made by me and fabienne
                if len(fan_list) > 0:
                    df = df.append(
                        {'PICTURE_ID': pic_id, 'SUBTYPE': fab_subtype, 'FORE-/BACKGROUND': fab_ground, 'TRUE': 'yes', 'PRED': 'yes'},
                        ignore_index=True)

                # if annotation was made only by me
                else:
                    df = df.append(
                        {'PICTURE_ID': pic_id, 'SUBTYPE': fab_subtype, 'FORE-/BACKGROUND': fab_ground, 'TRUE': 'yes', 'PRED': 'no'},
                        ignore_index=True)

            # add annotations made by fabienne, but not by fabian
            for idx, fan_ann in fabienne_subset.iterrows():
                fan_subtype = fan_ann['SUBTYPE']
                fan_ground = fan_ann['FORE-/BACKGROUND']

                # check if subtype is ces -> if true fore-/background is irrelevant
                if fan_subtype in ces_list:
                    df_list = df[(df['SUBTYPE'] == fan_subtype) & (df['PICTURE_ID'] == pic_id)]
                else:
                    df_list = df[(df['SUBTYPE'] == fan_subtype) & (df['FORE-/BACKGROUND'] == fan_ground) & (df['PICTURE_ID'] == pic_id)]

                # if annotation has not been added yet, add it
                if len(df_list) < 1:
                    df = df.append(
                        {'PICTURE_ID': pic_id, 'SUBTYPE': fan_subtype, 'FORE-/BACKGROUND': fan_ground, 'TRUE': 'no', 'PRED': 'yes'},
                        ignore_index=True)

    ## Without Fore- and Background
    elif mode == 'without':
        for pic_id in picture_ids:
            fabian_subset = fabian_data[(fabian_data['PICTURE_ID'] == pic_id)].drop_duplicates(subset = 'SUBTYPE')
            fabienne_subset = fabienne_data[(fabienne_data['PICTURE_ID'] == pic_id)].drop_duplicates(subset = 'SUBTYPE')
            fab_list = list(fabian_subset.SUBTYPE)
            fan_list = list(fabienne_subset.SUBTYPE)
            pic_id_subtype = list(df[(df['PICTURE_ID'] == pic_id)].SUBTYPE)
            # check whether my annotations are in fabiennes (if yes - one one, if not - one two)
            for subtype in fab_list:
                if subtype in fan_list and subtype not in pic_id_subtype:
                    df = df.append({'PICTURE_ID': pic_id, 'SUBTYPE': subtype, 'TRUE': 'yes', 'PRED': 'yes'},
                                   ignore_index=True)
                elif subtype not in fan_list and subtype not in pic_id_subtype:
                    df = df.append({'PICTURE_ID': pic_id, 'SUBTYPE': subtype, 'TRUE': 'yes', 'PRED': 'no'},
                                   ignore_index = True)
                else:
                    pass
            pic_id_subtype = list(df[(df['PICTURE_ID'] == pic_id)].SUBTYPE)
            for subtype in fan_list:
                if subtype not in pic_id_subtype:
                    df = df.append({'PICTURE_ID': pic_id, 'SUBTYPE': subtype, 'TRUE': 'no', 'PRED': 'yes'}, ignore_index=True)
        else:
            print("Unclear mode")
    return(df)


def jaccard_hist_pictures(df, output_name):
    df = df[['J-INDEX_with', 'J-INDEX_without']]
    binwidth = 10
    plt.hist([df['J-INDEX_with'],df['J-INDEX_without']], alpha = 1, bins=range(0, 100 + binwidth, binwidth), label=['Annotations with\nFore-/Background', 'Annotations without\nFore-/Background'])
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.xticks(np.arange(0, 110, 10))
    plt.xlabel('Jaccard-Index [%]')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.savefig('Plots/' + output_name + '.png', dpi = 1000, bbox_inches="tight")
    plt.show()

def jaccard_box_pictures(df, output_name):
    df = df[['J-INDEX_with', 'J-INDEX_without']]
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.boxplot(df, vert = False, widths = 0.5)
    plt.xlabel('Jaccard-Index [%]')
    plt.tight_layout()
    plt.xticks(np.arange(0, 120, 20))
    plt.yticks([1,2], ['Annotations with\nFore-/Background', 'Annotations without\nFore-/Background'])
    plt.savefig('Plots/' + output_name + '.png', dpi = 1000, bbox_inches="tight")
    plt.show()

def jaccard_index_pictures():
    df1 = pictures_jaccard(mode = 'with')[['PICTURE_ID', 'J-INDEX']]
    df2 = pictures_jaccard(mode = 'without')[['PICTURE_ID', 'J-INDEX']]
    df = df1.set_index('PICTURE_ID').join(df2.set_index('PICTURE_ID'), lsuffix='_with', rsuffix='_without')
    #jaccard_hist_pictures(df, 'histogram_jaccard_pictures')
    jaccard_box_pictures(df, 'boxplot_jaccard_pictures')
    print('Number of pictures: ', len(df))
    print('With - Mean: {0:.2f}, Median: {1:.2f}, Standard Deviation: {2:.2f}'.format(statistics.mean(df['J-INDEX_with']),statistics.median(df['J-INDEX_with']),statistics.stdev(df['J-INDEX_with'])))
    print('Without - Mean: {0:.2f}, Median: {1:.2f}, Standard Deviation: {2:.2f}'.format(statistics.mean(df['J-INDEX_without']),statistics.median(df['J-INDEX_without']),statistics.stdev(df['J-INDEX_without'])))

### Text Data
def extend_subtypes(liste):
    singles = list(dict.fromkeys(liste))
    final_list = []
    for entry in singles:
        entry_count = liste.count(entry)
        for x in range(entry_count):
            new_entry = str(entry)+str(x)
            final_list.append(new_entry)
    return final_list

def jaccard_box_text(df, output_name):
    df = df[['J-INDEX_both', 'J-INDEX_clear']]
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.boxplot(df, vert = False, widths = 0.5)
    plt.xlabel('Jaccard-Index [%]')
    plt.tight_layout()
    plt.yticks([1,2], ['All Annotations', 'Only clear Annotations'])
    plt.xticks(np.arange(0, 120, 20))
    plt.savefig('Plots/' + output_name + '.png', dpi = 1000, bbox_inches="tight")
    plt.show()

def jaccard_hist_text (df, output_name):
    df = df[['J-INDEX_both', 'J-INDEX_clear']]
    binwidth = 10
    plt.hist([df['J-INDEX_both'],df['J-INDEX_clear']], alpha = 1, bins=range(0, 100 + binwidth, binwidth), label=['All Annotations', 'Only clear Annotations'], color = ['yellowgreen', 'gold'])
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.xticks(np.arange(0, 110, 10))
    plt.xlabel('Jaccard-Index [%]')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.savefig('Plots/' + output_name + '.png', dpi = 1000, bbox_inches="tight")
    plt.show()

def tours_jaccard(mode):
    annotation_folder = 'Data/Annotations/'
    # test_folder = 'Data/Base Data/Test - Pictures/20210505_Pictures_Test.csv'
    fabian_annotation_file = annotation_folder + 'tour_annotations_merged.csv'
    fabienne_annotation_file = annotation_folder + 'tour_annotations_fabienne.csv'
    fabian_data = pd.read_csv(fabian_annotation_file, delimiter=';')
    fabian_data['SUBTYPE'] = fabian_data['SUBTYPE'].str.lower()
    fabienne_data = pd.read_csv(fabienne_annotation_file, delimiter=';')
    tours = fabienne_data['FILE'].drop_duplicates()
    df = pd.DataFrame(columns=['TOURS', 'OVERLAP', 'UNION', 'J-INDEX'])
    for tour in tours:
        if mode == 'both':
            fabian_subset = fabian_data[(fabian_data['FILE'] == tour)]
        elif mode == 'clear':
            fabian_subset = fabian_data[(fabian_data['FILE'] == tour) & (fabian_data['PRECISION'] == 'clear')]
        else:
            print('Unexpected mode')
        fabienne_subset = fabienne_data[(fabienne_data['FILE'] == tour)]
        fab_list = list(fabian_subset.SUBTYPE)
        fan_list = list(fabienne_subset.SUBTYPE)
        fab_list = extend_subtypes(fab_list)
        fan_list = extend_subtypes(fan_list)
        subtype_list = []
        subtype_list.extend(fan_list)
        subtype_list.extend(fab_list)
        df_subtypes = pd.DataFrame(subtype_list, columns=['SUBTYPE'])
        union = df_subtypes.drop_duplicates()
        overlap = df_subtypes.duplicated()
        df_subtypes['OVERLAP'] = overlap
        overlap = df_subtypes[(df_subtypes['OVERLAP'] == True)]
        overlap = len(overlap)
        union = len(union)
        df = df.append({'TOURS': tour, 'OVERLAP': overlap, 'UNION': union, 'J-INDEX': overlap/union*100}, ignore_index=True)
    return df

def jaccard_index_tours():
    df1 = tours_jaccard('both')
    df2 = tours_jaccard('clear')
    df = df1.set_index('TOURS').join(df2.set_index('TOURS'), lsuffix='_both', rsuffix='_clear')
    #jaccard_hist_text(df, 'histogram_jaccard_text')
    jaccard_box_text(df, 'boxplot_jaccard_text')
    print('Number of pictures: ', len(df))
    print('Both - Mean: {0:.2f}, Median: {1:.2f}, Standard Deviation: {2:.2f}'.format(statistics.mean(df['J-INDEX_both']),
                                                                                      statistics.median(df['J-INDEX_both']),
                                                                                    statistics.stdev(df['J-INDEX_both'])))
    print('Clear - Mean: {0:.2f}, Median: {1:.2f}, Standard Deviation: {2:.2f}'.format(statistics.mean(df['J-INDEX_clear']),
                                                                                       statistics.median(df['J-INDEX_clear']),
                                                                                       statistics.stdev(df['J-INDEX_clear'])))
def jaccard_index():
    jaccard_index_pictures()
    jaccard_index_tours()

def jaccard_index_poly(raster1, raster2):
    raster1 = raster1.dissolve()
    raster2 = raster2.dissolve()
    union = raster1.append(raster2)
    union = union.dissolve()
    union = sum(union['geometry'].area)
    overlap = raster1.intersection(raster2).area
    index = overlap / union
    print(round(index, 2))