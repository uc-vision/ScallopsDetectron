import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Polygon, LineString, Point
import glob
import numpy as np
from utils import polygon_functions as spf, geo_utils, file_utils
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


HOME_DIR = '/local'  # /home/tim
RECON_DIR = HOME_DIR + '/Dropbox/NIWA_UC/January_2021/Station_3_Grid/'

POLYGON_SCORE_THRESHOLD = 0.95

ANNS_ONLY = False
PREDS_ONLY = False

SHOW_INDIVIDUAL_PLT = True
SHOW_CLUSTER_PLT = False
NUM_POLYGON_PIX_SAMPLES = 4000


def grid_within_bounds(polygon, n_pnts):
    minx, miny, maxx, maxy = polygon.bounds
    n = int(math.sqrt(n_pnts))
    x = np.linspace(minx, maxx, n)
    y = np.linspace(miny, maxy, n)
    xv, yv = np.meshgrid(x, y)
    return xv.flatten(), yv.flatten()


def cloud_x_extent_line(polygon, ref_frame, center):
    pnts_ref = np.matmul(ref_frame.T, polygon.T).T
    x_mat = np.repeat(pnts_ref[None, :, 0], pnts_ref.shape[0], axis=0)
    distances = x_mat - x_mat.T
    max_dist = np.max(distances)
    pred_line = np.array([center + ref_frame[:, 0] * max_dist / 2, center - ref_frame[:, 0] * max_dist / 2])
    return pred_line


def plot_axes(ax, cntr, frame, scale=0.05):
    x_axes = np.array([cntr, cntr+scale*frame[:, 0]])
    ax.plot3D(x_axes[:, 0], x_axes[:, 1], x_axes[:, 2], c='red')
    y_axes = np.array([cntr, cntr+scale*frame[:, 1]])
    ax.plot3D(y_axes[:, 0], y_axes[:, 1], y_axes[:, 2], c='green')
    z_axes = np.array([cntr, cntr+scale*frame[:, 2]])
    ax.plot3D(z_axes[:, 0], z_axes[:, 1], z_axes[:, 2], c='blue')


def main():
    tiff_files = glob.glob(RECON_DIR + 'tiffs/dem_*')
    assert len(tiff_files) == 1
    dem_tif = rasterio.open(tiff_files[0])

    print("Extracting shape layers from viewer file...")
    file_utils.extract_vpz_shapes(RECON_DIR)

    shape_files = glob.glob(RECON_DIR + 'gpkg_files/*.gpkg')
    for sf in shape_files:
        label = sf.split('/')[-1].split('.')[0]
        if not any(k in label for k in ['Pred', 'Ann']) or any(k in label for k in ['filt', 'proc', 'ref', '2D']):
            continue
        if 'line' in label.lower() or (ANNS_ONLY and 'Pred' in label):
            continue

        if PREDS_ONLY and 'Anns' in label:
            continue

        gdf = gpd.read_file(sf)
        print(label)
        print(gdf.crs)
        names = []
        width_lines_local = []
        polygon_interiors_local = []
        polygon_bounds_local = []
        datum = gdf.geometry[0].exterior.coords[0]
        for name, polygon in tqdm(zip(gdf.NAME, gdf.geometry)):
            gdf_poly = gpd.GeoDataFrame(index=["scallop_tmp"], geometry=[polygon])
            x, y = grid_within_bounds(polygon, NUM_POLYGON_PIX_SAMPLES)
            points = [Point(xy) for xy in list(zip(x, y))]
            gdf_points = gpd.GeoDataFrame({'points': points}, geometry='points')
            Sjoin = gpd.tools.sjoin(gdf_points, gdf_poly, predicate="within", how='left')
            pnts_in_poly = gdf_points[Sjoin.index_right == 'scallop_tmp']
            scallop_int_pnts = np.array([(p.x, p.y) for p in pnts_in_poly.geometry])
            if scallop_int_pnts.shape[0] < 100:
                continue
            elevation_vals = np.array(list(dem_tif.sample(scallop_int_pnts))).flatten()
            scallop_int_pnts = np.hstack([scallop_int_pnts, elevation_vals[:, None]])

            poly_boundary = np.array(polygon.exterior.coords)
            elevation_vals = np.array(list(dem_tif.sample(poly_boundary[:, :2]))).flatten()
            poly_boundary[:, 2] = elevation_vals

            if len(poly_boundary) < 3:
                print('Invalid polygon: {}'.format(name))
            if name and name[0] == '{':
                properties = eval(name)
                properties['score'] = float(properties['conf']) * float(properties['cntr_cos'])
            else:
                properties = {'label': label}

            if 'score' not in properties or properties['score'] > POLYGON_SCORE_THRESHOLD:
                scallop_int_pnts_local = geo_utils.convert_gps2local(datum, scallop_int_pnts)
                polygon_interiors_local.append(scallop_int_pnts_local)
                poly_boundary_local = geo_utils.convert_gps2local(datum, poly_boundary)
                polygon_bounds_local.append(poly_boundary_local)

                eig_vecs, eig_vals, cntr = spf.pca(scallop_int_pnts_local)
                eig_vecs[:, 2] *= -1    # flip Eigen frame to right-handed
                x_extent_line = cloud_x_extent_line(scallop_int_pnts_local, eig_vecs, cntr)
                width_3D = np.linalg.norm(x_extent_line[1] - x_extent_line[0])
                properties['W_3D'] = round(width_3D, 4)

                width_lines_local.append(x_extent_line)
                names.append(str(properties))

                if SHOW_INDIVIDUAL_PLT:
                    fig = plt.figure(figsize=(10, 10))
                    ax = plt.axes(projection='3d')
                    ax.grid()
                    ax.plot3D(poly_boundary_local[:, 0], poly_boundary_local[:, 1], poly_boundary_local[:, 2])
                    x_extent_line_plt = x_extent_line - 0.002 * eig_vecs[:, 1]
                    ax.plot3D(x_extent_line_plt[:, 0], x_extent_line_plt[:, 1], x_extent_line_plt[:, 2], c='black')
                    plot_axes(ax, cntr, eig_vecs)
                    ax.scatter3D(scallop_int_pnts_local[:, 0], scallop_int_pnts_local[:, 1], scallop_int_pnts_local[:, 2], marker='.')
                    plt.show()

                    # fig.canvas.draw()
                    # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))[:, :, ::-1]
                    # cv2.imshow("plot", img)
                    # cv2.waitKey()

        # Save raw 3D prediction lines
        linestrings_gps = [LineString(geo_utils.convert_local2gps(datum, l)) for l in width_lines_local]
        gdf_3D = gpd.GeoDataFrame({'geometry': linestrings_gps, 'NAME': names}, geometry='geometry', crs=gdf.crs)
        shapes_fp_3D = RECON_DIR + 'gpkg_files/' + '_'.join(label.split(' ')) + '_proc_3D' + '.gpkg'
        file_utils.del_if_exists(shapes_fp_3D)
        gdf_3D.to_file(shapes_fp_3D)

        # Save raw 2D prediction lines
        linestrings_gps_2D = [LineString([xy[:2] for xy in list(line.coords)]) for line in linestrings_gps]
        gdf_2D = gpd.GeoDataFrame({'geometry': linestrings_gps_2D, 'NAME': names}, geometry='geometry', crs=gdf.crs)
        shapes_fp_2D = RECON_DIR + 'gpkg_files/' + '_'.join(label.split(' ')) + '_proc_2D' + '.gpkg'
        file_utils.del_if_exists(shapes_fp_2D)
        gdf_2D.to_file(shapes_fp_2D)

        # Skip non prediction files as these won't have multiple overlapping predictions
        if not 'Pred' in label:
            print("Adding processed width lines to vpz...")
            file_utils.append_vpz_shapes(RECON_DIR, [shapes_fp_2D], col_rgb_l=[(100, 255, 0)])
            continue

        print("RNN Clustering...")
        clustered_indices = spf.rnn_clustering(width_lines_local)
        print("{} Clusters found".format(len(clustered_indices)))
        filtered_lines = []
        filtered_labels = []
        for indices in tqdm(clustered_indices):
            lines = [width_lines_local[idx] for idx in indices]
            tags = [names[idx] for idx in indices]
            lengths = []
            scores = []
            for tag in tags:
                lengths.append(eval(tag)['W_3D'])
                scores.append(eval(tag)['score'])
            avg_length = round(np.array(lengths).mean(), 4)
            avg_score = round(np.array(scores).mean(), 4)
            filtered_lines.append(np.array(lines[0], dtype=float))
            filtered_labels.append(str({'label': 'live_pred_flt', 'score': avg_score, 'W_3D': avg_length}))

            if SHOW_CLUSTER_PLT:
                fig = plt.figure(figsize=(10, 10))
                ax = plt.axes(projection='3d')
                ax.grid()
                ax.clear()
                cld = polygon_interiors_local[indices[0]]
                ax.scatter3D(cld[:, 0], cld[:, 1], cld[:, 2], marker='.', color='black')
                bounds = [polygon_bounds_local[idx] for idx in indices]
                for bound in bounds:
                    ax.plot3D(bound[:, 0], bound[:, 1], bound[:, 2])
                plt.show()

        # Save filtered 3D prediction lines
        linestrings_gps = [LineString(geo_utils.convert_local2gps(datum, l)) for l in filtered_lines]
        gdf_3D_filt = gpd.GeoDataFrame({'geometry': linestrings_gps, 'NAME': filtered_labels}, geometry='geometry', crs=gdf.crs)
        shapes_fp_3D = RECON_DIR + 'gpkg_files/' + '_'.join(label.split(' ')) + '_proc_filt_3D' + '.gpkg'
        file_utils.del_if_exists(shapes_fp_3D)
        gdf_3D_filt.to_file(shapes_fp_3D)

        # Save filtered 2D prediction lines
        linestrings_gps_2D = [LineString([xy[:2] for xy in list(line.coords)]) for line in linestrings_gps]
        gdf_2D_filt = gpd.GeoDataFrame({'geometry': linestrings_gps_2D, 'NAME': filtered_labels}, geometry='geometry', crs=gdf.crs)
        shapes_fp_2D = RECON_DIR + 'gpkg_files/' + '_'.join(label.split(' ')) + '_proc_filt_2D' + '.gpkg'
        file_utils.del_if_exists(shapes_fp_2D)
        gdf_2D_filt.to_file(shapes_fp_2D)

        print("Adding processed filtered width lines to vpz...")
        file_utils.append_vpz_shapes(RECON_DIR, [shapes_fp_2D], col_rgb_l=[(255, 0, 0)])

        # print("Filtering {} polygons...".format(len(polygons)))
        # scallop_polygons, invalid_polygons = spf.filter_polygon_detections(list(zip(polygons, confs)))
        # print("Filtered down to {} polygons".format(len(scallop_polygons)))
        #
        # print("RNN clustering polygons...")
        # polygon_clusters = spf.polygon_rnn_clustering(scallop_polygons)
        # print("Num clusters: {}".format(len(polygon_clusters)))
        #
        # print("Filtering clusters...")
        # polygon_clusters, invalid_clusters = spf.filter_clusters(polygon_clusters)
        # print("Filtered down to {} clusters".format(len(polygon_clusters)))
        #
        # # print("Combining cluster masks...")
        # # filtered_polygons = [clust[0] for clust in polygon_clusters]
        # #
        # # print("Reducing filtered polygon complexity")
        # # filtered_polygons = [poly[::(1 + len(poly) // 40)] for poly in filtered_polygons]
        #
        # print("Calculating cluster sizes...")
        # scallop_sizes = spf.calc_cluster_widths(polygon_clusters, mode='max')
        #
        # print(scallop_sizes)

    dem_tif.close()


if __name__ == '__main__':
    main()