import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import Params as P
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import pathlib, os
from detectron2.config import get_cfg
from utils import VTKPointCloud as PC, scallop_poly_functions as spf
import vtk
import Metashape
import time


SHOW = True
VTK = False
EDGE_LIMIT_PIX = 200
OUTLIER_RADIUS = 0.1

def CamToWrld(pnts_cam, cam_quart):
    return np.matmul(cam_quart, np.vstack([pnts_cam, np.ones((1, pnts_cam.shape[1]))]))[:3, :]

def CamPixToWrldPnt(pixels_cam, cam_mtx):
    return np.matmul(np.linalg.inv(cam_mtx), np.vstack([pixels_cam, np.ones((1, pixels_cam.shape[1]))]))

def draw_scaled_axes(img, axis_vecs, axis_scales, origin, cam_mtx):
    points = np.concatenate([np.multiply(axis_vecs, np.repeat(axis_scales[:, None], 3, axis=1)) +
                             np.repeat(origin[None, :], 3, axis=0), origin[None, :]], axis=0)
    axis_points, _ = cv2.projectPoints(points, np.zeros((1, 3)), np.zeros((1, 3)), cam_mtx, None)
    axis_points = axis_points.astype(np.int)
    cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[2].ravel()), (255, 0, 0), 3)
    cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[1].ravel()), (0, 255, 0), 3)
    cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[0].ravel()), (0, 0, 255), 3)


doc = Metashape.Document()
doc.open("/local/ScallopReconstructions/gopro_116_0/recon.psx")
chunk = doc.chunks[0]
chunk_scale = chunk.transform.scale or 1
print("Chunk scale: {}".format(chunk_scale))
cameras = chunk.cameras
# print("Building depth maps...")
# st = time.time()
# chunk.buildDepthMaps()
# print("Depth build time: {}s".format(time.time()-st))

UD_ALPHA = 0

MODEL_PATH = "/local/ScallopMaskRCNNOutputs/HR+LR LP AUGS/"

cfg = get_cfg()
cfg.NUM_GPUS = 1
cfg.merge_from_file(MODEL_PATH + 'config.yml')
model_paths = [str(path) for path in pathlib.Path(MODEL_PATH).glob('*.pth')]
model_paths.sort()
cfg.MODEL.WEIGHTS = os.path.join(model_paths[-1])

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
cfg.TEST.AUG.ENABLED = False
cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
cfg.TEST.AUG.MAX_SIZE = 4000
cfg.TEST.AUG.FLIP = False
cfg.TEST.PRECISE_BN.ENABLED = False
cfg.TEST.PRECISE_BN.NUM_ITER = 200
predictor = DefaultPredictor(cfg)

if SHOW:
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Input image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Labelled sub image", cv2.WINDOW_NORMAL)
if VTK:
    pnt_cld = PC.VtkPointCloud(pnt_size=1)
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(1000, 1000)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    vtk_axes = vtk.vtkAxesActor()
    axes_transform_np = np.eye(4)
    axes_matrix = vtk.vtkMatrix4x4()
    vtk_axes.SetUserMatrix(axes_matrix)
    ren.AddActor(vtk_axes)
    ren.AddActor(pnt_cld.vtkActor)
    iren.Initialize()

scallop_detections = []
for cam in tqdm(cameras[3::1]):
    start_time = time.time()

    cam_img_m = cam.image()
    img_path = cam.photo.path
    img_fn = img_path.split('/')[-1]
    c = cam.calibration
    camMtx = np.array([[c.f + c.b1, c.b2, c.cx + c.width / 2],
                       [0, c.f, c.cy + c.height / 2],
                       [0, 0, 1]])
    camDist = np.array([[c.k1, c.k2, c.p2, c.p1, c.k3]])
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camMtx, camDist, (int(cam_img_m.width), int(cam_img_m.height)),
                                                      UD_ALPHA, (int(cam_img_m.width), int(cam_img_m.height)))
    x_ud, y_ud, w_ud, h_ud = roi
    img = np.frombuffer(cam_img_m.tostring(), dtype=np.uint8).reshape((int(cam_img_m.height), int(cam_img_m.width), -1))[:, :, ::-1]
    img_cam_ud = cv2.undistort(img, cameraMatrix=camMtx, distCoeffs=camDist, newCameraMatrix=newcameramtx)
    img_cam_und_roi = img_cam_ud[y_ud:y_ud+h_ud, x_ud:x_ud+w_ud].copy()
    cam_quart = np.array(cam.transform).reshape((4, 4))

    img_depth_ms = chunk.model.renderDepth(cam.transform, cam.sensor.calibration, add_alpha=False)
    img_shape = img_cam_ud.shape
    img_depth_np = np.frombuffer(img_depth_ms.tostring(), dtype=np.float32).reshape((img_shape[0], img_shape[1], 1))

    edge_box = (EDGE_LIMIT_PIX, EDGE_LIMIT_PIX, img_shape[1]-EDGE_LIMIT_PIX, img_shape[0]-EDGE_LIMIT_PIX)

    outputs = predictor(img_cam_ud)
    v = Visualizer(img_cam_ud[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    instances = outputs["instances"].to("cpu")
    v = v.draw_instance_predictions(instances)
    out_image = v.get_image()[:, :, ::-1].copy()
    masks = instances._fields['pred_masks']
    bboxes = instances._fields['pred_boxes']
    scores = instances._fields['scores']
    for mask, box, score in list(zip(masks, bboxes, scores)):
        mask_pnts = np.array(np.where(mask))[::-1].transpose()
        (x, y), radius = cv2.minEnclosingCircle(mask_pnts)
        if edge_box[0] <= x <= edge_box[2] and edge_box[1] <= y <= edge_box[3]:
            cv2.circle(out_image, (int(x), int(y)), int(radius), color=(0, 255, 0), thickness=2)

            mask_pnts_sub = mask_pnts[::2]
            vert_elavations = img_depth_np[mask_pnts_sub[:, 1], mask_pnts_sub[:, 0]]
            scallop_pnts_cam = CamPixToWrldPnt(mask_pnts_sub.T, camMtx)
            scallop_pnts_cam = scallop_pnts_cam * vert_elavations.T

            scallop_pnts_cam = spf.remove_outliers(scallop_pnts_cam, OUTLIER_RADIUS / chunk_scale)

            pc_vecs, pc_lengths, center_pnt = spf.polyvert_pca(scallop_pnts_cam.T)
            MUL = 1.9
            pc_lengths = np.sqrt(pc_lengths) * MUL
            scaled_pc_lengths = pc_lengths * chunk_scale * 2

            if spf.is_scallop(scaled_pc_lengths):
                scallop_center = mask_pnts_sub.mean(axis=0).astype(np.int)
                cv2.putText(out_image, str(round(scaled_pc_lengths[0], 3)), tuple(scallop_center + np.array([10, 10])),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 12, cv2.LINE_AA)
                draw_scaled_axes(out_image, pc_vecs, pc_lengths, center_pnt, camMtx)
                scallop_detections.append((np.mean(CamToWrld(scallop_pnts_cam, cam_quart), axis=1),
                                           scaled_pc_lengths[0],
                                           score.numpy()))

            if VTK:
                pnt_cld.setPoints(scallop_pnts_cam.T - center_pnt, np.array([1, 1, 1] * scallop_pnts_cam.shape[1]).T)
                axes_transform_np[:3, :3] = np.multiply(pc_vecs, np.repeat(pc_lengths[:, None], 3, axis=1)).T
                axes_matrix.DeepCopy(axes_transform_np.ravel())
                vtk_axes.Modified()
                iren.Render()
                iren.Start()

    if SHOW:
        #print("Image inference time: {}s".format(time.time()-start_time))
        cv2.rectangle(out_image, edge_box[:2], edge_box[2:], (0, 0, 255), thickness=1)
        cv2.imshow("Input image", img_cam_ud)
        cv2.imshow("Labelled sub image", out_image)
        depth_display = 255*(img_depth_np - np.min(img_depth_np)) / np.max(img_depth_np)
        cv2.imshow("Depth", depth_display.astype(np.uint8))
        key = cv2.waitKey(1)
        if key == ord('q'):
            exit(0)


pnt_cld = PC.VtkPointCloud(pnt_size=4)
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(1000, 1000)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
ren.AddActor(pnt_cld.vtkActor)
vtk_axes = vtk.vtkAxesActor()
ren.AddActor(vtk_axes)
iren.Initialize()

scallop_pnts_wrld = np.array([loc for loc, rad, score in scallop_detections])
scallop_sizes = np.array([size for loc, size, conf in scallop_detections])

len_pnts = scallop_pnts_wrld.shape[0]
pnt_cld.setPoints(scallop_pnts_wrld, np.array(len_pnts*[[0, 1, 0]]))
extr = vtk.vtkEuclideanClusterExtraction()
extr.SetInputData(pnt_cld.vtkPolyData)
extr.SetRadius(0.2)
extr.SetExtractionModeToAllClusters()
extr.SetColorClusters(True)
extr.Update()
#TODO: average cluster sizes w/ outlier rejection to get better sizing, fewer repeat detections

plt.figure(1)
plt.title("Scallop Spatial Distribution [m]")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.plot(scallop_pnts_wrld[:, 0], scallop_pnts_wrld[:, 1], 'ro')
plt.grid(True)
plt.savefig(P.INFERENCE_OUTPUT_DIR + "ScallopSpatialDistImg.jpeg")
plt.figure(2)
plt.title("Scallop Size Distribution (freq. vs size [m])")
plt.ylabel("Frequency")
plt.xlabel("Scallop Width [m]")
plt.hist(scallop_sizes, bins=100)
plt.figtext(0.15, 0.85, "Total count: {}".format(extr.GetNumberOfExtractedClusters()))
plt.grid(True)
plt.savefig(P.INFERENCE_OUTPUT_DIR + "ScallopSizeDistImg.jpeg")
plt.show()

#print(extr.GetOutput())
subMapper = vtk.vtkPointGaussianMapper()
subMapper.SetInputConnection(extr.GetOutputPort(0))
subMapper.SetScaleFactor(0.05)
subMapper.SetScalarRange(0, extr.GetNumberOfExtractedClusters())
subActor = vtk.vtkActor()
subActor.SetMapper(subMapper)
#ren.AddActor(subActor)
print(extr.GetNumberOfExtractedClusters())

#confs_wrld = points_wrld[:, 7] * 255
#confs_rgb = cv2.applyColorMap(confs_wrld.astype(np.uint8), cv2.COLORMAP_JET)[:, 0, :].astype(np.float32) / 255

iren.Start()