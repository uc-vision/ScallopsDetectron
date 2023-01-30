import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, LineString, Point
import glob
import numpy as np
from utils import polygon_functions as spf, geo_utils
import math
from tqdm import tqdm


HOME_DIR = '/local'  # /home/tim
RECON_DIR = HOME_DIR + '/Dropbox/NIWA_UC/January_2021/Station_3_Grid/'

SHOW_PLT = False
if SHOW_PLT:
    import matplotlib.pyplot as plt
    import cv2
NUM_POLYGON_PIX_SAMPLES = 2000


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
    dem_tif = rasterio.open(RECON_DIR + 'tiffs/dem_.tif')
    print(dem_tif.crs)

    shape_files = glob.glob(RECON_DIR + 'gpkg_files/*.gpkg')
    print(shape_files)
    for sf in shape_files:
        if not any(k in sf for k in ['Pred', 'Ann']) or any(k in sf for k in ['filt', 'proc', 'ref', '2D']):
            continue
        fn = sf.split('/')[-1].split('.')[0]
        print(fn)
        gdf = gpd.read_file(sf)
        print(gdf.crs)
        names = []
        confs = []
        polygons = []
        width_lines_local = []
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
                score = float(properties['conf']) * float(properties['cntr_cos'])
            else:
                score = 1.0
                properties = {'label': 'Pred'} if 'Pred' in fn else {'label': 'live_polygon_ann'}

            scallop_int_pnts_local = geo_utils.convert_gps2local(datum, scallop_int_pnts)

            eig_vecs, eig_vals, cntr = spf.pca(scallop_int_pnts_local)
            eig_vecs[:, 2] *= -1    # flip Eigen frame to right-handed
            x_extent_line = cloud_x_extent_line(scallop_int_pnts_local, eig_vecs, cntr)

            width_3D = np.linalg.norm(x_extent_line[1] - x_extent_line[0])
            properties['W_3D'] = round(width_3D, 4)

            if score > 0.85:
                polygons.append(scallop_int_pnts)
                width_lines_local.append(x_extent_line)
                names.append(str(properties))
                confs.append(score)

            if SHOW_PLT:
                fig = plt.figure(figsize=(10, 10))
                ax = plt.axes(projection='3d')
                ax.grid()
                poly_boundary_local = geo_utils.convert_gps2local(datum, poly_boundary)
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
        gdf_3D.to_file(RECON_DIR + 'gpkg_files/' + '_'.join(fn.split(' ')) + '_proc_3D' + '.gpkg')

        # Save raw 2D prediction lines
        linestrings_gps_2D = [LineString([xy[:2] for xy in list(line.coords)]) for line in linestrings_gps]
        gdf_2D = gpd.GeoDataFrame({'geometry': linestrings_gps_2D, 'NAME': names}, geometry='geometry', crs=gdf.crs)
        gdf_2D.to_file(RECON_DIR + 'gpkg_files/' + '_'.join(fn.split(' ')) + '_proc_2D' + '.gpkg')

        # Skip non prediction files as these won't have multiple overlapping predictions
        if 'Pred' in fn:
            continue

        predline_clusters, labels = spf.polygon_rnn_clustering(width_lines_local, names)
        filtered_lines = []
        filtered_labels = []
        for lines, labels in zip(predline_clusters, labels):
            lengths = []
            for label in labels:
                lengths.append(eval(label)['W_3D'])
            avg_length = round(np.array(lengths).mean(), 4)
            filtered_lines.append(np.array(lines[0], dtype=float))
            filtered_labels.append(str({'label': 'live', 'W_3D': avg_length}))

        linestrings_gps = [LineString(geo_utils.convert_local2gps(datum, l)) for l in filtered_lines]
        gdf_3D_filt = gpd.GeoDataFrame({'geometry': linestrings_gps, 'NAME': filtered_labels}, geometry='geometry', crs=gdf.crs)
        gdf_3D_filt.to_file(RECON_DIR + 'gpkg_files/' + '_'.join(fn.split(' ')) + '_proc_filt' + '.gpkg')

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

if __name__ == '__main__':
    main()