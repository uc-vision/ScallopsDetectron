import Metashape
print("Metashape version {}".format(Metashape.version))
from utils import VTKPointCloud as PC, polygon_functions as spf
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

RECON_DIR = '/local/ScallopReconstructions/gopro_119_2/'

SAVE_PLT = True
DISPLAY = False
OUTPUT_SHAPES = True

SAVE_SHAPES_2D = True

SHAPEGROUP_KEY = "Pred"  # "Ann" None

SCALE_MUL = 1.0
CLUSTER_CNT_THRESH = 3

doc = Metashape.Document()
doc.open(RECON_DIR + "recon.psx")
doc.read_only = False

chunk = doc.chunks[0]

print("Extracting Shapes...")
if chunk.shapes is None:
    chunk.shapes = Metashape.Shapes()
    chunk.shapes.crs = Metashape.CoordinateSystem("EPSG::4326")
shapes_crs = chunk.shapes.crs
chunk_transform = chunk.transform.matrix

shape_files = glob.glob(RECON_DIR + '*.gpkg')
for shape_file in shape_files:
    chunk.importShapes(shape_file)
    lbl = shape_file.split('/')[-1].split('.')[0]
    chunk.shapes.groups[-1].label = lbl

chunk_polygons_d = spf.get_chunk_polygons_dict(chunk, key=SHAPEGROUP_KEY, world_crs=True)

for key, polygon_detections in chunk_polygons_d.items():
    print("Analysing {}...".format(key))

    print("Filtering {} polygons...".format(len(polygon_detections)))
    scallop_polygons, invalid_polygons = spf.filter_polygon_detections(polygon_detections)
    print("Filtered down to {} polygons".format(len(scallop_polygons)))

    print("RNN clustering polygons...")
    polygon_clusters = spf.polygon_rnn_clustering(scallop_polygons)
    print("Num clusters: {}".format(len(polygon_clusters)))

    print("Filtering clusters...")
    polygon_clusters, invalid_clusters = spf.filter_clusters(polygon_clusters)
    print("Filtered down to {} clusters".format(len(polygon_clusters)))

    print("Combining cluster masks...")
    filtered_polygons = [clust[0] for clust in polygon_clusters]

    print("Reducing filtered polygon complexity")
    filtered_polygons = [poly[::(1 + len(poly) // 40)] for poly in filtered_polygons]

    if OUTPUT_SHAPES:
        print("Saving filtered polygons back to recon...")
        filtered_shapegroup = chunk.shapes.addGroup()
        filtered_shapegroup.color = (100, 255, 100, 255)
        filtered_shapegroup.label = "Filtered_" + '_'.join(key.split('_')[1:])
        filtered_shapegroup.enabled = False
        for flt_poly in filtered_polygons:
            new_shape = chunk.shapes.addShape()
            new_shape.group = filtered_shapegroup
            new_shape.label = "scallop_flt" #_{}".format(round(0.01, 2))
            new_shape.geometry.type = Metashape.Geometry.Type.PolygonType
            polygon = [shapes_crs.project(Metashape.Vector(pnt)) for pnt in flt_poly]
            new_shape.geometry = Metashape.Geometry.Polygon(polygon)
        #doc.save()

        shapes_fn = RECON_DIR+key
        print(chunk.shapes.groups)
        chunk.exportShapes(shapes_fn + '_Filtered.gpkg', groups=[len(chunk.shapes.groups)-1])

        if SAVE_SHAPES_2D:
            # Convert shapes to 2D
            gdf = gpd.read_file(shapes_fn + '_Filtered.gpkg')
            new_geo = []
            for polygon in gdf.geometry:
                if polygon.has_z:
                    assert polygon.geom_type == 'Polygon'
                    lines = [xy[:2] for xy in list(polygon.exterior.coords)]
                    new_geo.append(Polygon(lines))
            gdf.geometry = new_geo
            gdf.to_file(shapes_fn + '_Filtered_2.gpkg')

    print("Calculating cluster sizes...")
    scallop_sizes = spf.calc_cluster_widths(polygon_clusters, mode='max')

    print("Plotting...")

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

    fig = plt.figure(2)
    plt.title("Scallop Size Distribution (freq. vs size [cm]) for " + key)
    plt.ylabel("Frequency")
    plt.xlabel("Scallop Width [cm]")
    plt.hist(scallop_sizes, bins=100)
    plt.figtext(0.15, 0.85, "Total count: {}".format(len(scallop_sizes)))
    plt.grid(True)
    fig.set_size_inches(8, 6)
    if SAVE_PLT:
        plt.savefig(RECON_DIR + "ScallopSizeDistImg_{}.jpeg".format(key), dpi=600)
    if DISPLAY:
        plt.show()

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