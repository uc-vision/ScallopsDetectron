import Metashape
print("Metashape version {}".format(Metashape.version))
from utils import VTKPointCloud as PC, scallop_poly_functions as spf, math_utils
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

RECON_DIR = '/local/ScallopReconstructions/gopro_119/'

SAVE_PLT = True
DISPLAY = True

SHAPEGROUP_KEY = "Pred"  # "Ann" None

SCALE_MUL = 1.0 #1.1236
CLUSTER_CNT_THRESH = 0 #3

PC_MUL = 1.9

doc = Metashape.Document()
doc.open(RECON_DIR + "recon.psx")

scallop_detections = {}

for chunk in doc.chunks:
    shapes_crs = chunk.shapes.crs
    chunk_transform = chunk.transform.matrix
    print("Extracting Shapes...")
    chunk_polygons_d = spf.get_chunk_polygons_dict(chunk, key=SHAPEGROUP_KEY, world_crs=True)
    for key, scallop_polys in chunk_polygons_d.items():
        print("Calculating PCA and sizes for {}...".format(key))
        key_detections = []
        for scallop_poly in tqdm(scallop_polys):
            scallop_poly = scallop_poly[::(1 + scallop_poly.shape[0] // 20)]
            pc_vecs, pc_lengths, center_pnt = spf.polyvert_pca(scallop_poly)
            pc_lengths = np.sqrt(pc_lengths) * PC_MUL
            scaled_pc_lengths = pc_lengths * 2
            pca_width = scaled_pc_lengths[0]
            scallop_vert_mat = np.repeat(scallop_poly[None], scallop_poly.shape[0], axis=0)
            scallop_vert_dists = np.linalg.norm(scallop_vert_mat - scallop_vert_mat.transpose([1, 0, 2]), axis=2)
            max_polygon_width = np.max(scallop_vert_dists)
            scallop_center = scallop_poly.mean(axis=0)
            key_detections.append([scallop_center, min(max_polygon_width * SCALE_MUL, 0.2)])
        scallop_detections[key] = key_detections

for key, detections in scallop_detections.items():
    # TODO: efficient clustering, confidence function, outlier removal
    print("Performing Clustering for {}...".format(key))
    scallop_detection_pnts = np.array([det[0] for det in detections])
    centroids, cluster_counts, detection_classes = math_utils.kmeans_clustering(scallop_detection_pnts, k_means=-1)
    valid_classes = np.where(cluster_counts >= CLUSTER_CNT_THRESH)[0]
    valid_detection_clusters = [[centroids[cls],
                                 cluster_counts[cls],
                                 np.array([det[1] for i, det in enumerate(detections) if detection_classes[i] == cls]).mean()] for cls in valid_classes]

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

    scallop_pnts = np.array([loc for loc, cnt, size in valid_detection_clusters])
    scallop_sizes = np.array([size for loc, cnt, size in valid_detection_clusters]) * 100

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