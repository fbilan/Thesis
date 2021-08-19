from matplotlib.patches import Polygon
import A5_Text_Handling
from A2_Visualisation import *
from A4_Picture_Handling import *
from A5_Text_Handling import *
import math

# This function extracts bounding box coordinates and returns a string which can be used with Flickr API
def bbox_coords(wgs_polygon):
    bbox = wgs_polygon.total_bounds
    gap = ", "
    bbox = str(bbox[0]) + gap + str(bbox[1]) + gap + str(bbox[2]) + gap + str(bbox[3])
    return bbox

# This function clips a point gdf with the boundaries of a polygon and returns the gdf in wgs84 format
def clipping(gdf, polygon):
    gdf_wgs = gdf.set_crs(epsg=4326)
    gdf_lv = gdf_wgs.to_crs(epsg=2056)
    if polygon.crs == gdf_lv.crs:
        gdf_lv = geopandas.clip(gdf_lv, polygon)
    else:
        print('CRS not identical')
    gdf_wgs = gdf_lv.to_crs(epsg=4326)
    return gdf_wgs

# This function calculates the distance from a point file to the closest line feature and stores it in the point file
def min_distance(df, lines_file):
    dist = []
    for idx, point in df.iterrows():
        dist.append(lines_file.distance(point['geometry']).min())
    #df['distance_to_line'] = dist
    print("Distances calculated")
    return dist


# This function converts the raster cell to a polygon based on its center point
def raster_center_to_polygon(x_coord, y_coord, grid_size):
    half_cell = 0.5*grid_size
    ll = (x_coord-half_cell, y_coord-half_cell)
    ul = (x_coord-half_cell, y_coord+half_cell)
    ur = (x_coord+half_cell, y_coord+half_cell)
    lr = (x_coord+half_cell, y_coord-half_cell)
    return(Polygon([ll,ul,ur,lr]))

# This function uses the previous function to create a polygon set from the raster cells
def convert_raster_to_polygon(x_mesh, y_mesh, count_array, grid_size):
    polygons = geopandas.GeoDataFrame() # empty geodataframe
    for row in range(len(count_array)):
        for cell in range(len(count_array[row])):
            if count_array[row][cell] >= 0:
                x_coord = x_mesh[0][cell]
                y_coord = y_mesh[row][0]
                polygon = raster_center_to_polygon(x_coord, y_coord, grid_size)
                polygon = geopandas.GeoDataFrame(geometry=geopandas.GeoSeries(polygon))
                polygon['value'] = count_array[row][cell]
                polygons = polygons.append(polygon)
    return(polygons)

def moran_local_hh(shapefile):
    # Create Shapefile of meshgrids (multipolygon)
    weights = ps.weights.Queen.from_dataframe(shapefile)  # generate spatial weights (Queen in this case)
    moran_loc = esda.Moran_Local(shapefile[['value']], weights)  # calculate Moran's I
    # fig, ax = moran_scatterplot(moran_loc, p=0.05)
    # ax.set_xlabel('Donatns')
    # ax.set_ylabel('Spatial Lag of Donatns')
    # plt.show()
    shapefile['significant'] = moran_loc.p_sim < 0.05 # Break observations into significant or not
    shapefile['quadrant'] = moran_loc.q # Store the quadrant they belong to
    hh = shapefile.loc[(shapefile['quadrant'] == 1) & (shapefile['significant'] == True), 'geometry'] # extract HH
    return(hh)


    # ax = municipalities_lv.plot(facecolor='none', edgecolor='grey', figsize=(9, 9))
    # lisa_cluster(moran_loc, shapefile, p=0.05, legend = True, alpha = 0.5, ax = ax)
    # base_plot(ax = ax, output_name='test_lisa')
    # plt.show()

    # input_img = input_grid
    # w = ps.weights.lat2W(len(input_img), len(input_img[0]), rook=False, id_type="int")
    # lm = Moran_Local(input_img, w)
    # moran_significance = np.reshape(lm.p_sim, (len(input_img), len(input_img[0])))
    # #print(input_grid)
    # #print(moran_significance)

def kde_quartic(d,h):
    dn=d/h
    P=(15/16)*(1-dn**2)**2
    return P

def kde(point_file_lv, boundary, grid_size, h):
    # DEFINE GRID SIZE AND RADIUS(h)
    x = point_file_lv['geometry'].x.reset_index(drop=True, inplace=False)
    y = point_file_lv['geometry'].y.reset_index(drop=True, inplace=False)
    grid_size = grid_size
    h = h
    # GETTING X,Y MIN AND MAX
    bbox = boundary.total_bounds

    x_min = bbox[0] - grid_size
    x_max = bbox[2] + grid_size
    y_min = bbox[1] - grid_size
    y_max = bbox[3] + grid_size

    # CONSTRUCT GRID
    x_grid = np.arange(x_min - h, x_max + h, grid_size)
    y_grid = np.arange(y_min - h, y_max + h, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    # GRID CENTER POINT
    xc = x_mesh + (grid_size / 2)
    yc = y_mesh + (grid_size / 2)

    intensity_list = []
    for j in range(len(xc)):
        intensity_row = []
        for k in range(len(xc[0])):
            kde_value_list = []
            for i in range(len(x)):
                # CALCULATE DISTANCE
                d = math.sqrt((xc[j][k] - x[i]) ** 2 + (yc[j][k] - y[i]) ** 2)
                if d <= h:
                    p = kde_quartic(d, h)
                else:
                    p = 0
                kde_value_list.append(p)
            # SUM ALL INTENSITY VALUE
            p_total = sum(kde_value_list)
            intensity_row.append(p_total)
        intensity_list.append(intensity_row)

    intensity = np.array(intensity_list)
    intensity = np.ma.masked_array(intensity, intensity < 1)
    return x_mesh, y_mesh, intensity

def extract_automatic_annotations(point_file_lv_picture, point_file_lv_text, hotspot_file, feature, picture_annotations ='Data/Annotations/picture_annotations_merged.csv'):
    if feature == 'Pictures':
        annotations = geopandas.clip(point_file_lv_picture, hotspot_file)
        annotations = geopandas.sjoin(annotations, hotspot_file, how='inner', op='within')
        picture_annotations = pd.read_csv(picture_annotations, delimiter=',')
        # remove fore-/background
        picture_annotations = picture_annotations.drop_duplicates(subset = ['PICTURE_ID', 'SUBTYPE'])
        for cluster in range(max(annotations['CLUSTER'])):
            df = pd.DataFrame()  # columns=['Subtype', 'Clear', 'Unclear']
            annotations_cluster = annotations[annotations['CLUSTER'] == cluster+1]
            print('\nCluster ' + str(cluster + 1))
            for index, row in annotations_cluster.iterrows():
                id = int(row['id'])
                #select annotations based on picture_id
                annotation_of_id = picture_annotations[picture_annotations['PICTURE_ID'] == id]
                #annotation_of_id = annotation_of_id.drop_duplicates(subset = ['PICTURE_ID', 'SUBTYPE'])
                df = df.append(annotation_of_id)
            df = df[df['LF/CES'] == 'LF']
            subtypes = df['SUBTYPE'].drop_duplicates()
            hotspot_area = hotspot_file[hotspot_file['CLUSTER'] == cluster+1]
            hotspot_area = int(hotspot_area['geometry'].area/1000000)
            for subtype in subtypes:
                total = round(len(df[df['SUBTYPE'] == subtype])/hotspot_area, 2)
                ratio = round(len(df[df['SUBTYPE'] == subtype])/len(picture_annotations[picture_annotations['SUBTYPE'] == subtype])*100/hotspot_area, 2)
                print(subtype, '&', total, '&', ratio)

        print('\nTotal')
        df = pd.DataFrame()
        annotations_cluster = annotations
        for index, row in annotations_cluster.iterrows():
            id = int(row['id'])
            annotation_of_id = picture_annotations[picture_annotations['PICTURE_ID'] == id]
            df = df.append(annotation_of_id)
        df = df[df['LF/CES'] == 'LF']
        subtypes = df['SUBTYPE'].drop_duplicates()
        hotspot_area = sum(hotspot_file['geometry'].area) / 1000000
        print(hotspot_area)
        for subtype in subtypes:
            total = round(len(df[df['SUBTYPE'] == subtype]) / hotspot_area, 2)
            ratio = round(len(df[df['SUBTYPE'] == subtype]) / len(
                picture_annotations[picture_annotations['SUBTYPE'] == subtype]) * 100 / hotspot_area, 2)
            print(subtype, '&', total, '&', ratio)

            #print('Contributors:', len(df['OWNER'].drop_duplicates()))
            #extract_numbers_pictures('Landscape Features', add_title= 'Cluster '+str(cluster+1), df = df)
            #extract_numbers_pictures('Cultural Ecosystem Services', add_title= 'Cluster '+str(cluster+1), df = df)

    elif feature == 'Text':
        annotations = geopandas.clip(point_file_lv_text, hotspot_file)
        annotations = geopandas.sjoin(annotations, hotspot_file, how='inner', op='within')
        for cluster in range(max(annotations['CLUSTER'])):
            print('\nCluster ' + str(cluster + 1))
            df = pd.DataFrame()
            annotations_cluster = annotations[annotations['CLUSTER'] == cluster+1]
            df = df.append(annotations_cluster)
            df = df[df['LF/CES'] == 'LF']
            subtypes = df['SUBTYPE'].drop_duplicates()
            hotspot_area = hotspot_file[hotspot_file['CLUSTER'] == cluster + 1]
            hotspot_area = int(hotspot_area['geometry'].area / 1000000)
            for subtype in subtypes:
                total = round(len(df[df['SUBTYPE'] == subtype]) / hotspot_area, 2)
                ratio = round(len(df[df['SUBTYPE'] == subtype]) / len(
                    point_file_lv_text[point_file_lv_text['SUBTYPE'] == subtype]) * 100 / hotspot_area, 2)
                print(subtype, total, '&', ratio)

        print('\nTotal')
        df = pd.DataFrame()
        df = df.append(annotations)
        df = df[df['LF/CES'] == 'LF']
        subtypes = df['SUBTYPE'].drop_duplicates()
        hotspot_area = sum(hotspot_file['geometry'].area) / 1000000
        print(hotspot_area)
        for subtype in subtypes:
            total = round(len(df[df['SUBTYPE'] == subtype]) / hotspot_area, 2)
            ratio = round(len(df[df['SUBTYPE'] == subtype]) / len(
                point_file_lv_text[point_file_lv_text['SUBTYPE'] == subtype]) * 100 / hotspot_area, 2)
            print(subtype, total, '&', ratio)
            #A5_Text_Handling.extract_numbers_tours(add_title='Cluster '+str(cluster+1), df = annotations_cluster)

    elif feature == 'Combined':
        annotations_pic = geopandas.clip(point_file_lv_picture, hotspot_file)
        annotations_pic = geopandas.sjoin(annotations_pic, hotspot_file, how='inner', op='within')
        annotations_txt = geopandas.clip(point_file_lv_text, hotspot_file)
        annotations_txt = geopandas.sjoin(annotations_txt, hotspot_file, how='inner', op='within')
        picture_annotations = pd.read_csv(picture_annotations, delimiter=',')
        picture_annotations = picture_annotations.drop_duplicates(subset=['PICTURE_ID', 'SUBTYPE'])
        for cluster in range(max(annotations_pic['CLUSTER'])):
            print('\nCluster ' + str(cluster + 1))
            df = pd.DataFrame()  # columns=['Subtype', 'Clear', 'Unclear']
            annotations_cluster_pic = annotations_pic[annotations_pic['CLUSTER'] == cluster + 1]
            annotations_cluster_txt = annotations_txt[annotations_txt['CLUSTER'] == cluster + 1]
            df = df.append(annotations_cluster_txt)
            for index, row in annotations_cluster_pic.iterrows():
                id = int(row['id'])
                annotation_of_id = picture_annotations[picture_annotations['PICTURE_ID'] == id]
                annotation_of_id = annotation_of_id.drop_duplicates(subset = ['PICTURE_ID', 'SUBTYPE'])
                df = df.append(annotation_of_id)
            df = df[df['LF/CES'] == 'LF']
            subtypes = df['SUBTYPE'].drop_duplicates()
            hotspot_area = hotspot_file[hotspot_file['CLUSTER'] == cluster + 1]
            hotspot_area = int(hotspot_area['geometry'].area / 1000000)
            for subtype in subtypes:
                total = round(len(df[df['SUBTYPE'] == subtype]) / hotspot_area, 2)
                ratio = round(len(df[df['SUBTYPE'] == subtype]) / (len(
                    picture_annotations[picture_annotations['SUBTYPE'] == subtype])+len(annotations_txt[annotations_txt['SUBTYPE'] == subtype])) * 100 / hotspot_area, 2)
                print(subtype, '&', total, '&', ratio)

        print('\nTotal')
        df = pd.DataFrame()
        annotations_cluster = annotations_pic
        for index, row in annotations_cluster.iterrows():
            id = int(row['id'])
            annotation_of_id = picture_annotations[picture_annotations['PICTURE_ID'] == id]
            df = df.append(annotation_of_id)
        df = df.append(annotations_txt)
        df = df[df['LF/CES'] == 'LF']
        subtypes = df['SUBTYPE'].drop_duplicates()
        hotspot_area = sum(hotspot_file['geometry'].area) / 1000000
        print(hotspot_area)
        for subtype in subtypes:
            total = round(len(df[df['SUBTYPE'] == subtype]) / hotspot_area, 2)
            ratio = round(len(df[df['SUBTYPE'] == subtype]) / (len(
                    picture_annotations[picture_annotations['SUBTYPE'] == subtype])+len(annotations_txt[annotations_txt['SUBTYPE'] == subtype])) * 100 / hotspot_area, 2)
            print(subtype, '&', total, '&', ratio)

    else:
        print('Wrong Feature')

def extract_manual_annotations(picture_annotations_lv, text_annotations_lv, feature, distance):
    lf_subtypes = ["Bedrock", "Flower / Funghi", "Forest", "Grass- and Moorland", "Lake", "Natural Landscape", "River / Creek", "Rock", "Shrub", "Snow / Ice", "Summit", "Tree", "Waterfall", "Wild Animal", "Agriculture", "Human influenced Landscape", "Infrastructure", "Livestock", "Path / Trail", "Urban"]
    ces_subtypes = ["Identity", "Information Board", "Information Office", "Local History", "Tradition",
                    "Traditional Architecture", "Recreational Facilities", "Signpost", "Viewpoint", "Camping", "People",
                    "Restaurant / Accommodation", "Dawn / Sunset", "Healing Powers", "Place Attachment", "Church",
                    "Summit Cross", "Total"]
    ces_subtypes = ["Camping", "People", "Restaurant / Accommodation", "Dawn / Sunset", "Healing Powers", "Place Attachment", "Church", "Summit Cross"]
    results_dict = {}
    for lf_subtype in lf_subtypes:
        lf_dict = {}
        results_dict[lf_subtype] = lf_dict
        for ces_subtype in ces_subtypes:
            ces_dict = {}
            results_dict[lf_subtype][ces_subtype] = ces_dict
            results_dict[lf_subtype][ces_subtype] = [0.0, 0.0, 0.0]
    # get rid of fore-/background
    picture_annotations_lv = picture_annotations_lv.drop_duplicates(subset = ['PICTURE_ID', 'SUBTYPE'])
    picture_annotations_lv = picture_annotations_lv[['LF/CES', 'SUBTYPE', 'latitude', 'longitude', 'geometry']]
    ces_pic_annotations = picture_annotations_lv[picture_annotations_lv['LF/CES'] == 'CES']

    text_annotations_lv = text_annotations_lv[['LF/CES', 'SUBTYPE', 'latitude', 'longitude', 'geometry']]
    ces_text_annotations = text_annotations_lv[text_annotations_lv['LF/CES'] == 'CES']

    if feature == 'Pictures':
        for ces_subtype in ces_subtypes:
            print(ces_subtype)
            if ces_subtype == "Total":
                ces_annotations_buff = ces_pic_annotations.buffer(distance=distance)
                ces_lf_clip = geopandas.clip(picture_annotations_lv[picture_annotations_lv['LF/CES'] == 'LF'], ces_annotations_buff)
                lf_clip = list(ces_lf_clip['SUBTYPE'].drop_duplicates())

                for lf_subtype in lf_clip:
                    ces_captured = geopandas.clip(ces_pic_annotations,
                                                  ces_lf_clip[ces_lf_clip['SUBTYPE'] == lf_subtype].buffer(
                                                      distance=distance))
                    count = len(ces_lf_clip[ces_lf_clip['SUBTYPE'] == lf_subtype])
                    ratio_lf = round(count /
                        len(picture_annotations_lv[picture_annotations_lv['SUBTYPE'] == lf_subtype])*100, 2)
                    ratio_ces = round(len(ces_captured)/len(ces_pic_annotations)*100, 2)
                    # print(lf_subtype, count, ratio)
                    results_dict[lf_subtype][ces_subtype][0] = count
                    results_dict[lf_subtype][ces_subtype][1] = ratio_lf
                    results_dict[lf_subtype][ces_subtype][2] = ratio_ces
            else:
                ces_subtype_annotations = ces_pic_annotations[ces_pic_annotations['SUBTYPE'] == ces_subtype]
                ces_annotations_buff = ces_subtype_annotations.buffer(distance=distance)
                ces_lf_clip = geopandas.clip(picture_annotations_lv[picture_annotations_lv['LF/CES'] == 'LF'], ces_annotations_buff)
                lf_clip = list(ces_lf_clip['SUBTYPE'].drop_duplicates())

                for lf_subtype in lf_clip:
                    ces_captured = geopandas.clip(ces_subtype_annotations, ces_lf_clip[ces_lf_clip['SUBTYPE'] == lf_subtype].buffer(distance=distance))
                    count = len(ces_lf_clip[ces_lf_clip['SUBTYPE'] == lf_subtype])
                    ratio_lf = round(count / len(picture_annotations_lv[picture_annotations_lv['SUBTYPE'] == lf_subtype])*100, 2)
                    ratio_ces = round(len(ces_captured)/len(ces_subtype_annotations)*100, 2)
                    #print(lf_subtype, count, ratio)
                    results_dict[lf_subtype][ces_subtype][0] = count
                    results_dict[lf_subtype][ces_subtype][1] = ratio_lf
                    results_dict[lf_subtype][ces_subtype][2] = ratio_ces

        print(results_dict)
        for key in results_dict:
            text = str(key)
            #total_text = str(key) + " & " + str(results_dict[key]['Total'][0]) + " & " + str(results_dict[key]['Total'][1]) + " & " + str(results_dict[key]['Total'][2])
            #print(total_text)
            for key_2 in results_dict[key]:
                text = text + " & " + str(results_dict[key][key_2][0]) + " & " + str(results_dict[key][key_2][1]) + " & " + str(results_dict[key][key_2][2])
            print(text)

    elif feature == 'Text':
        for ces_subtype in ces_subtypes:
            print(ces_subtype)
            if ces_subtype == "Total":
                ces_annotations_buff = ces_text_annotations.buffer(distance=distance)
                ces_lf_clip = geopandas.clip(text_annotations_lv[text_annotations_lv['LF/CES'] == 'LF'], ces_annotations_buff)
                lf_clip = list(ces_lf_clip['SUBTYPE'].drop_duplicates())

                for lf_subtype in lf_clip:
                    ces_captured = geopandas.clip(ces_text_annotations,
                                                  ces_lf_clip[ces_lf_clip['SUBTYPE'] == lf_subtype].buffer(
                                                      distance=distance))
                    count = len(ces_lf_clip[ces_lf_clip['SUBTYPE'] == lf_subtype])
                    ratio_lf = round(count / len(text_annotations_lv[text_annotations_lv['SUBTYPE'] == lf_subtype])*100, 2)
                    ratio_ces = round(len(ces_captured)/len(ces_text_annotations)*100, 2)
                    # print(lf_subtype, count, ratio)
                    results_dict[lf_subtype][ces_subtype][0] = count
                    results_dict[lf_subtype][ces_subtype][1] = ratio_lf
                    results_dict[lf_subtype][ces_subtype][2] = ratio_ces
            else:
                ces_subtype_annotations = ces_text_annotations[ces_text_annotations['SUBTYPE'] == ces_subtype]
                ces_annotations_buff = ces_subtype_annotations.buffer(distance=distance)
                ces_lf_clip = geopandas.clip(text_annotations_lv[text_annotations_lv['LF/CES'] == 'LF'], ces_annotations_buff)
                lf_clip = list(ces_lf_clip['SUBTYPE'].drop_duplicates())

                for lf_subtype in lf_clip:
                    ces_captured = geopandas.clip(ces_subtype_annotations,
                                                  ces_lf_clip[ces_lf_clip['SUBTYPE'] == lf_subtype].buffer(
                                                      distance=distance))
                    count = len(ces_lf_clip[ces_lf_clip['SUBTYPE'] == lf_subtype])
                    ratio_lf = round(count / len(text_annotations_lv[text_annotations_lv['SUBTYPE'] == lf_subtype])*100, 2)
                    ratio_ces = round(len(ces_captured)/len(ces_subtype_annotations)*100, 2)
                    #print(lf_subtype, count, ratio)
                    results_dict[lf_subtype][ces_subtype][0] = count
                    results_dict[lf_subtype][ces_subtype][1] = ratio_lf
                    results_dict[lf_subtype][ces_subtype][2] = ratio_ces
        print(results_dict)
        for key in results_dict:
            text = str(key)
            #total_text = str(key) + " & " + str(results_dict[key]['Total'][0]) + " & " + str(results_dict[key]['Total'][1]) + " & " + str(results_dict[key]['Total'][2])
            #print(total_text)
            for key_2 in results_dict[key]:
                text = text + " & " + str(results_dict[key][key_2][0]) + " & " + str(results_dict[key][key_2][1]) + " & " + str(results_dict[key][key_2][2])
            print(text)

    elif feature == 'Combined':
        ces_subtype_annotations = ces_text_annotations.append(ces_pic_annotations).reset_index(drop=True, inplace = False)
        for ces_subtype in ces_subtypes:
            print(ces_subtype)
            if ces_subtype == "Total":
                ces_annotations_buff = ces_subtype_annotations.buffer(distance=distance)
                ces_lf_clip_text = geopandas.clip(text_annotations_lv[text_annotations_lv['LF/CES'] == 'LF'], ces_annotations_buff)
                ces_lf_clip_pic = geopandas.clip(picture_annotations_lv[picture_annotations_lv['LF/CES'] == 'LF'], ces_annotations_buff)
                ces_lf_clip = ces_lf_clip_pic.append(ces_lf_clip_text)
                lf_clip = list(ces_lf_clip['SUBTYPE'].drop_duplicates())
                for lf_subtype in lf_clip:
                    ces_captured = geopandas.clip(ces_subtype_annotations,
                                                  ces_lf_clip[ces_lf_clip['SUBTYPE'] == lf_subtype].buffer(
                                                      distance=distance))
                    count = len(ces_lf_clip[ces_lf_clip['SUBTYPE'] == lf_subtype])
                    ratio_lf = round(count / (len(text_annotations_lv[text_annotations_lv['SUBTYPE'] == lf_subtype]) + (
                        len(picture_annotations_lv[picture_annotations_lv['SUBTYPE'] == lf_subtype])))*100, 2)
                    ratio_ces = round(len(ces_captured) / len(ces_subtype_annotations) * 100, 2)
                    # print(lf_subtype, count, ratio)
                    results_dict[lf_subtype][ces_subtype][0] = count
                    results_dict[lf_subtype][ces_subtype][1] = ratio_lf
                    results_dict[lf_subtype][ces_subtype][2] = ratio_ces
            else:
                ces_subtypes_annotations_subtype = ces_subtype_annotations[ces_subtype_annotations['SUBTYPE'] == ces_subtype]
                #print(len(ces_subtypes_annotations_subtype), " ", ces_subtype, " selected")
                ces_annotations_buff = ces_subtypes_annotations_subtype.buffer(distance=distance)
                ces_lf_clip_text = geopandas.clip(text_annotations_lv[text_annotations_lv['LF/CES'] == 'LF'],
                                                  ces_annotations_buff)
                ces_lf_clip_pic = geopandas.clip(picture_annotations_lv[picture_annotations_lv['LF/CES'] == 'LF'],
                                                 ces_annotations_buff)
                ces_lf_clip = ces_lf_clip_text.append(ces_lf_clip_pic)
                lf_clip = list(ces_lf_clip['SUBTYPE'].drop_duplicates())

                for lf_subtype in lf_clip:
                    ces_captured = geopandas.clip(ces_subtypes_annotations_subtype,
                                                  ces_lf_clip[ces_lf_clip['SUBTYPE'] == lf_subtype].buffer(
                                                      distance=distance))
                    count = len(ces_lf_clip[ces_lf_clip['SUBTYPE'] == lf_subtype])
                    ratio_lf = round(count / (len(text_annotations_lv[text_annotations_lv['SUBTYPE'] == lf_subtype]) + (len(picture_annotations_lv[picture_annotations_lv['SUBTYPE'] == lf_subtype])))*100, 2)
                    ratio_ces = round(len(ces_captured) / len(ces_subtypes_annotations_subtype) * 100, 2)
                    results_dict[lf_subtype][ces_subtype][0] = count
                    results_dict[lf_subtype][ces_subtype][1] = ratio_lf
                    results_dict[lf_subtype][ces_subtype][2] = ratio_ces
        print(results_dict)
        x = []
        y = []
        for key in results_dict:
            text = str(key)
            total_text = str(key) + " & " + str(results_dict[key]['Total'][0]) + " & " + str(results_dict[key]['Total'][1]) + " & " + str(results_dict[key]['Total'][2])
            #print(total_text)
            for key_2 in results_dict[key]:
                text = text + " & " + str(results_dict[key][key_2][0]) + " & " + str(results_dict[key][key_2][1]) + " & " + str(results_dict[key][key_2][2])
                x.append(results_dict[key][key_2][1])
                y.append(results_dict[key][key_2][2])
            print(text)

        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        df = pd.DataFrame()
        df['x'] = x
        df['y'] = y
        # plt.boxplot(x)
        # plt.show()
        # plt.boxplot(y)
        # plt.show()
        #fig, ax = plt.subplots()

        fig = plt.figure(figsize=(8, 8))
        #plt.plot(x, y, 'o', color='black')
        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)

        # use the previously defined function
        scatter_hist(x, y, ax, ax_histx, ax_histy)

        # axHistx.axis["bottom"].major_ticklabels.set_visible(False)
        for tl in ax_histx.get_xticklabels():
            tl.set_visible(False)
        ax_histx.set_yticks([0, 5, 10, 15])

        # axHisty.axis["left"].major_ticklabels.set_visible(False)
        for tl in ax_histy.get_yticklabels():
            tl.set_visible(False)
        ax_histy.set_xticks([0, 5, 10, 15])
        plt.show()

    else:
        print('Wrong Feature')

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, c = 'black')
    ax.set_xlabel('%LF')
    ax.set_ylabel('%CES')
    alpha = 0.6
    ll = plt.Rectangle((0, 0), 9, 12, fc='#dddddd', alpha=alpha)
    lm = plt.Rectangle((0, 12), 9, 15, fc='#7bb3d1', alpha=alpha)
    lh = plt.Rectangle((0, 27), 9, 73, fc='#016eae', alpha=alpha)
    ml = plt.Rectangle((9, 0), 9, 12, fc='#dd7c8a', alpha=alpha)
    mm = plt.Rectangle((9, 12), 9, 15, fc='#8d6c8f', alpha=alpha)
    mh = plt.Rectangle((9, 27), 9, 73, fc='#4a4779', alpha=alpha)
    hl = plt.Rectangle((18, 0), 82, 12, fc='#cc0024', alpha=alpha)
    hm = plt.Rectangle((18, 12), 82, 15, fc='#8a274a', alpha=alpha)
    hh = plt.Rectangle((18, 27), 82, 73, fc='#4b264d', alpha=alpha)
    ax.add_patch(ll)
    ax.add_patch(lm)
    ax.add_patch(lh)
    ax.add_patch(ml)
    ax.add_patch(mm)
    ax.add_patch(mh)
    ax.add_patch(hl)
    ax.add_patch(hm)
    ax.add_patch(hh)
    # now determine nice limits by hand:
    binwidth = 3
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(0, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color = 'grey')
    ax_histy.hist(y, bins=bins, orientation='horizontal', color='grey')

def KDE_Plot(output, point_file_lv, boundary, municipalities_lv):
    x_mesh, y_mesh, intensity = kde(point_file_lv, boundary, 100, 1000)
    fig, ax = plt.subplots(figsize=(9, 9))
    municipalities_lv.plot(ax=ax, facecolor='none', edgecolor='grey')
    boundary.plot(facecolor='none', edgecolor='black', ax=ax)
    municipalities_lv.apply(
        lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=8, color='black',
                              path_effects=[pe.withStroke(linewidth=1.2, foreground="white")]), axis=1)
    cax = ax.pcolormesh(x_mesh, y_mesh, intensity, cmap='hot_r', alpha=0.6)
    cbar = plt.colorbar(cax, ticks=[np.min(intensity) + 0.1 * np.max(intensity),
                                    np.max(intensity) - np.max(intensity) * 0.1], shrink=0.75)
    cbar.ax.set_yticklabels(['Low', 'High'])
    base_plot(ax=ax, output_name='kde_' + output)
    plt.show()

