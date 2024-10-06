# import Metashape
# print("Metashape version {}".format(Metashape.version))
from utils import VTKPointCloud as PC, polygon_functions as spf
from utils import geo_utils
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import re
import glob
from shapely.geometry import Polygon
import geopandas as gpd

# TODO: make single filtered polygon per cluster
# TODO: display valid/invalid polygon examples
# TODO: weighted confidence function?
# TODO: outlier removal


VPZ_DIR = '/csse/research/CVlab/processed_bluerov_data/240714-140552/'

SAVE_PLT = True
DISPLAY = False
OUTPUT_SHAPES = True

SAVE_SHAPES_2D = True

SHAPEGROUP_KEY = "Pred"  # "Ann" None

SCALE_MUL = 1.0
CLUSTER_CNT_THRESH = 3

shape_file_3D = glob.glob(VPZ_DIR + '*3D.gpkg')
assert len(shape_file_3D) == 1
scallops_gpd = gpd.read_file(shape_file_3D[0])
scallop_polys_gps = []
for detection in scallops_gpd.itertuples():
    poly_lonlatz = np.array(detection.geometry.exterior.coords)
    label = detection.NAME
    conf = float(eval(label)['conf'])
    scallop_polys_gps.append([poly_lonlatz, conf])

datum = scallop_polys_gps[0][0][0][:2]
scallop_polys_local = [[geo_utils.convert_gps2local(datum, p), c] for p, c in scallop_polys_gps]

polygons_d = {"detections": scallop_polys_local}

for key, polygon_detections in polygons_d.items():
    print("Analysing {}...".format(key))

    print("Filtering {} polygons...".format(len(polygon_detections)))
    scallop_polygons, invalid_polygons = spf.filter_polygon_detections(polygon_detections)
    # scallop_polygons = [p[0] for p in polygon_detections]
    print("Filtered down to {} polygons".format(len(scallop_polygons)))

    print("RNN clustering polygons...")
    polygon_clusters, labels = spf.polygon_rnn_clustering(scallop_polygons, ["labels"]*len(scallop_polygons))
    print("Num clusters: {}".format(len(polygon_clusters)))

    print("Filtering clusters...")
    polygon_clusters, invalid_clusters = spf.filter_clusters(polygon_clusters)
    print("Filtered down to {} clusters".format(len(polygon_clusters)))

    print("Combining cluster masks...")
    filtered_polygons = [clust[0] for clust in polygon_clusters]

    print("Reducing filtered polygon complexity")
    filtered_polygons = [poly[::(1 + len(poly) // 40)] for poly in filtered_polygons]

    filtered_polygons = [geo_utils.convert_local2gps(datum, p) for p in filtered_polygons]

    print("Calculating cluster sizes...")
    scallop_sizes = spf.calc_cluster_widths(polygon_clusters, mode='max')

    if OUTPUT_SHAPES:
        shapes_fn = VPZ_DIR+key

        geometry = [Polygon(poly[:, :2]) for poly in filtered_polygons]
        names = [str(round(i, 4)) for i in scallop_sizes]
        polygons_2d_gpd = gpd.GeoDataFrame({'NAME': names, 'geometry': geometry}, geometry='geometry')
        polygons_2d_gpd.to_file(shapes_fn + '_Filtered_2D.gpkg')

    # fig = plt.figure(1)
    # ax = fig.gca()
    # plt.title("Scallop Detection Spatial Distribution [m] for " + key)
    # plt.xlabel("x [m]")
    # plt.ylabel("y [m]")
    # for cluster_id in range(len(centroids)):
    #     cluster_pnts = scallop_detection_pnts[detection_classes == cluster_id]
    #     plt.scatter(cluster_pnts[:, 0], cluster_pnts[:, 1], s=50, label=cluster_id)
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=10, color='black')
    # for valid_detection in valid_detection_clusters:
    #     circle = plt.Circle(valid_detection[0][:2], valid_detection[2] / 2, color='black', fill=False)
    #     ax.add_patch(circle)
    # plt.grid(True)
    # plt.axis('equal')
    # fig.set_size_inches(8, 6)
    # if SAVE_PLT:
    #     plt.savefig(RECON_DIR + "ScallopSpatialDistImg_{}.jpeg".format(key), dpi=600)

    # fig = plt.figure(2)
    # plt.title("Scallop Size Distribution (freq. vs size [cm]) for " + key)
    # plt.ylabel("Frequency")
    # plt.xlabel("Scallop Width [cm]")
    # plt.hist(scallop_sizes, bins=100)
    # plt.figtext(0.15, 0.85, "Total count: {}".format(len(scallop_sizes)))
    # plt.grid(True)
    # fig.set_size_inches(8, 6)
    # if SAVE_PLT:
    #     plt.savefig(RECON_DIR + "ScallopSizeDistImg_{}.jpeg".format(key), dpi=600)
    # if DISPLAY:
    #     plt.show()

#doc.save(RECON_DIR + 'recon.psx')

# with open("scallop_detections_l.pkl", "rb") as fp:
#     scallop_detections = pickle.load(fp)


# pnt_cld = PC.VtkPointCloud(pnt_size=4)
# ren = vtk.vtkRenderer()
# renWin = vtk.vtkRenderWindow()
# renWin.AddRenderer(ren)
# renWin.SetSize(1000, 1000)
# iren = vtk.vtkRenderWindowInteractor()
# iren.SetRenderWindow(renWin)
# ren.AddActor(pnt_cld.vtkActor)
# vtk_axes = vtk.vtkAxesActor()
# ren.AddActor(vtk_axes)
# iren.Initialize()
#
# len_pnts = scallop_pnts_wrld.shape[0]
# pnt_cld.setPoints(scallop_pnts_wrld, np.array(len_pnts*[[0, 1, 0]]))
# extr = vtk.vtkEuclideanClusterExtraction()
# extr.SetInputData(pnt_cld.vtkPolyData)
# extr.SetRadius(0.2)
# extr.SetExtractionModeToAllClusters()
# extr.SetColorClusters(True)
# extr.Update()
#
# #print(extr.GetOutput())
# subMapper = vtk.vtkPointGaussianMapper()
# subMapper.SetInputConnection(extr.GetOutputPort(0))
# subMapper.SetScaleFactor(0.05)
# subMapper.SetScalarRange(0, extr.GetNumberOfExtractedClusters())
# subActor = vtk.vtkActor()
# subActor.SetMapper(subMapper)
# #ren.AddActor(subActor)
# print(extr.GetNumberOfExtractedClusters())
#
# #confs_wrld = points_wrld[:, 7] * 255
# #confs_rgb = cv2.applyColorMap(confs_wrld.astype(np.uint8), cv2.COLORMAP_JET)[:, 0, :].astype(np.float32) / 255
#
# iren.Start()