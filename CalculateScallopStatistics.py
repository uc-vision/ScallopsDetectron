import zipfile
import tempfile
import glob
import xml.etree.cElementTree as ET
import geopandas as gp
import pandas as pd
from shapely.geometry import *
import numpy as np
from utils import geo_utils
from utils.transect_mapper import transect_mapper
import os

# Will break code:
# Overlapping polygons in the same class of either include OR exclude
# Tape reference must be part of vpz file not imported shape

PROCESSED_BASEDIR = "/csse/research/CVlab/processed_bluerov_data/"
DONE_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_done.txt'

def get_poly_arr_2d(poly):
    return np.array(poly.exterior.coords)[:, :2]

def get_local_poly_arr(poly_gps):
    poly_arr_2d = get_poly_arr_2d(poly_gps)
    poly_arr_m = geo_utils.convert_gps2local(poly_arr_2d[0], poly_arr_2d)
    return poly_arr_m

def get_poly_area_m2(poly_gps):
    return Polygon(get_local_poly_arr(poly_gps)).area


if __name__ == "__main__":
    with open(DONE_DIRS_FILE, 'r') as f:
        dirs_list = f.readlines()

    dirs_list = ['240714-140552\n']

    for dir_entry in dirs_list:
        if len(dir_entry) == 1 or '#' in dir_entry:
            continue
        dir_name = dir_entry[:-1]
        dir_full = PROCESSED_BASEDIR + dir_name + '/'

        # TODO: multiple shape files - yes but need to be careful not to double count scallops
        # Load scallop polygons
        scallop_gpkg_paths = glob.glob(dir_full + '*detections_Filtered*.gpkg')
        scallop_polygons = []
        for spoly_path in scallop_gpkg_paths:
            spoly_gdf = gp.read_file(spoly_path)
            scallop_polygons.extend(list(spoly_gdf.geometry))

        # get include / exclude regions from viewer file
        exclude_polys = []
        include_polys = []
        transect_map = None
        zf = zipfile.ZipFile(dir_full + dir_name + '.vpz')
        with tempfile.TemporaryDirectory() as tempdir:
            zf.extractall(tempdir)
            vpz_root = ET.parse(tempdir + '/doc.xml').getroot()
            for child in vpz_root.iter('layer'):
                if child.attrib['type'] == 'shapes':
                    elem_data = list(child.iter('data'))
                    elem_src = list(child.iter('source'))
                    if len(elem_data):
                        shape_fn = tempdir + '/' + elem_data[0].attrib['path']
                    elif len(elem_src):
                        shape_fn = dir_full + elem_src[0].attrib['path']
                    else:
                        continue
                    if child.attrib['label'] in ['Exclude Areas', 'Include Areas']:
                        dst_list = exclude_polys if child.attrib['label'] == 'Exclude Areas' else include_polys
                        for i, row in gp.read_file(shape_fn).iterrows():
                            if isinstance(row.geometry, Polygon):
                                dst_list.append(row.geometry)
                    if child.attrib['label'] == "Tape Reference":
                        transect_map = transect_mapper.TransectMapper()
                        transect_map.create_map_from_gpkg(shape_fn)
                        print("Tape reference found")

        # TODO: check for overlap interclass and trim

        # Calculate total valid survey area
        total_inc_area = 0.0
        for inc_poly in include_polys:
            inc_area = get_poly_area_m2(inc_poly)
            exc_area = 0.0
            for exc_poly in exclude_polys:
                if inc_poly.intersects(exc_poly):
                    inters_poly = inc_poly.intersection(exc_poly)
                    exc_area += get_poly_area_m2(inters_poly)
            total_inc_area += inc_area - exc_area
        print("Survey area, m2 =", round(total_inc_area, 2))

        # Filter scallops in valid survey area(s)
        print("Total number scallops =", len(scallop_polygons))
        valid_scallop_polygons = []
        for spoly in scallop_polygons:
            # check if scallop is over 50% in include area and under 50% in any exclude areas
            valid = False
            scallop_area = get_poly_area_m2(spoly)
            for bound_polys, keep in [[include_polys, True], [exclude_polys, False]]:
                for b_poly in bound_polys:
                    if b_poly.intersects(spoly):
                        intersection_polygon = b_poly.intersection(spoly)
                        if intersection_polygon.is_empty:
                            continue
                        if isinstance(intersection_polygon, Polygon):
                            interestion_area = get_poly_area_m2(intersection_polygon)
                        elif isinstance(intersection_polygon, MultiPolygon) or isinstance(intersection_polygon, GeometryCollection):
                            interestion_area = np.sum([get_poly_area_m2(p) for p in list(intersection_polygon.geoms) if isinstance(p, Polygon)])
                        else:
                            raise ValueError
                        if interestion_area > scallop_area / 2:
                            valid = keep
                            break
            if valid:
                valid_scallop_polygons.append(spoly)
        print("Number of valid scallops =", len(valid_scallop_polygons))

        # test_write_gdf = gp.GeoDataFrame({'NAME': 'test', 'geometry': valid_scallop_polygons}, geometry='geometry')
        # test_write_gdf.to_file('/csse/users/tkr25/Desktop/valid_scallops.geojson', driver='GeoJSON')

        # TODO: Get 3D scallop polygon values from DEM??????? size in 2D for now...

        # Calculate valid scallop widths
        scallop_stats = {'lat': [], 'lon': [], 'width': []}
        for vspoly in valid_scallop_polygons:
            local_poly = get_local_poly_arr(vspoly)

            # TODO: improve sizing (shape fitting?)
            # naive_max_w:
            scallop_vert_mat = np.repeat(local_poly[None], local_poly.shape[0], axis=0)
            scallop_vert_dists = np.linalg.norm(scallop_vert_mat - scallop_vert_mat.transpose([1, 0, 2]), axis=2)
            max_width = np.max(scallop_vert_dists)

            poly_arr = get_poly_arr_2d(vspoly)
            scallop_stats['width'].append(max_width)
            lon, lat = np.mean(poly_arr, axis=0)
            scallop_stats['lat'].append(lat)
            scallop_stats['lon'].append(lon)

        # If paired site, read from diver data and process
        diver_data_fn = PROCESSED_BASEDIR + 'ruk2401_dive_slate_data_entry Kura Reihana.csv'
        print(transect_map)
        if transect_map and os.path.isfile(diver_data_fn):
            diver_points_gps = []
            diver_measurements = []
            diver_measurements_df = pd.read_csv(diver_data_fn)
            for i, csv_row in diver_measurements_df.iterrows():
                t_para = csv_row['y']
                t_perp = csv_row["x"]
                left_side = 'Left' in str(csv_row['diver'])
                t_perp = (-1 if left_side else 1) * t_perp / 100
                if csv_row['site'] == 'UQ 19':
                    gps_coord = transect_map.transect2gps([t_para, t_perp])
                    if gps_coord is not None:
                        diver_points_gps.append(Point(gps_coord))
                        diver_measurements.append(csv_row["SCA_mm"] / 1000)
            # TODO: compare diver measured and detected / annotated scallops, produce plots and statistics

            tags = [str(t) for t in diver_measurements]
            diver_meas_gdf = gp.GeoDataFrame({'NAME': tags, 'geometry': diver_points_gps}, geometry='geometry')
            diver_meas_gdf.to_file(dir_full + 'Diver Measurements.geojson', driver='GeoJSON')

        site_dataframe = pd.DataFrame(scallop_stats)
        with open(dir_full + 'valid_scallop_sizes.csv', 'w') as f:
            site_dataframe.to_csv(f, header=True, index=False)

        # TODO: work out what James wants - count, size stats, gps point
        survey_stats_row = pd.DataFrame({'lat': [0], 'lon': [0], 'count': [0], 'size': [0]})
        survey_stats_fp = PROCESSED_BASEDIR + 'scallop_survey_stats.csv'
        csv_exists = os.path.isfile(survey_stats_fp)
        with open(survey_stats_fp, 'a' if csv_exists else 'w') as f:
            survey_stats_row.to_csv(f, header=not csv_exists, index=False)

        break