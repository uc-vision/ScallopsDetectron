import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import pathlib
import Params as P
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import VTKPointCloud as PC
import vtk
import Metashape
import time

SHOW = False
EDGE_LIMIT_PIX = int(0.05 / (P.PIXEL_SCALE * P.ORTHO_SCALE))
METASHAPE_INFERENCE_DIRECTORY = "/local/ScallopMaskDataset/Metashape_output/"


def CamToWrld(pnts_cam, cam_quart):
    return np.matmul(cam_quart, np.vstack([pnts_cam, np.ones((1, pnts_cam.shape[1]))]))[:3, :]

def CamPixToWrldPnt(pixels_cam, cam_mtx):
    return np.matmul(np.linalg.inv(cam_mtx), np.vstack([pixels_cam, np.ones((1, pixels_cam.shape[1]))]))


doc = Metashape.Document()
doc.open(P.METASHAPE_CHKPNT_PATH)
chunk = doc.chunk
cameras = chunk.cameras
# print("Building depth maps...")
# st = time.time()
# chunk.buildDepthMaps()
# print("Depth build time: {}s".format(time.time()-st))

c = chunk.sensors[0].calibration
camMtx = np.array([[c.f + c.b1,    c.b2,   c.cx + c.width / 2],
                   [0,             c.f,    c.cy + c.height / 2],
                   [0,             0,      1]])
camDist = np.array([[c.k1, c.k2, c.p2, c.p1, c.k3]])
x0, y0 = camMtx[:2, 2]
h,  w = (2160, 3840)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camMtx, camDist, (w,h), 0, (w,h))
w_ud, h_ud = roi[2:]
fx = newcameramtx[0, 0]
fy = newcameramtx[1, 1]
FOV = [math.degrees(2*math.atan(w_ud / (2*fx))),
       math.degrees(2*math.atan(h_ud / (2*fy)))]
#print(FOV)

cfg = get_cfg()
cfg.merge_from_file('config.yml')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, P.MODEL_PATH)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.92
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.DATASETS.TEST = ("/local/ScallopMaskDataset/val", )
predictor = DefaultPredictor(cfg)

scallop_detections = []
for cam in tqdm(cameras[::50]):
    start_time = time.time()
    cam_img_path = cam.photo.path
    cam_quart = np.array(cam.transform).reshape((4, 4))
    img_cam = cv2.imread(cam_img_path)
    img_cam = cv2.undistort(img_cam, cameraMatrix=camMtx, distCoeffs=camDist, newCameraMatrix=newcameramtx)
    img_shape = img_cam.shape
    edge_box = (EDGE_LIMIT_PIX, EDGE_LIMIT_PIX, img_shape[1]-EDGE_LIMIT_PIX, img_shape[0]-EDGE_LIMIT_PIX)
    img_depth_ms = chunk.model.renderDepth(cam.transform, cam.sensor.calibration, add_alpha=False)
    img_depth_np = np.frombuffer(img_depth_ms.tostring(), dtype=np.float32).reshape((img_cam.shape[0], img_cam.shape[1], 1))

    outputs = predictor(img_cam)
    v = Visualizer(img_cam[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
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
            #TODO: scallopness contour classification
            #TODO: work out if depth images are undistorted or what
            z_val = img_depth_np[int(y), int(x)] - cam_quart[2, 3]
            scallop_pnts_cam = CamPixToWrldPnt(np.array([[x, x-radius, x+radius], [y, y, y]]), camMtx)
            scallop_pnts_wrld = CamToWrld(scallop_pnts_cam, cam_quart) * z_val

            pnt_3D = scallop_pnts_wrld[:, 0] #cam.unproject(np.array([[x], [y]])) * z_val
            size_3D = np.linalg.norm(scallop_pnts_wrld[:, 1] - scallop_pnts_wrld[:, 2]) #cam.unproject(np.array([[x-radius], [y]])) - cam.unproject(np.array([[x+radius], [y]])))
            scallop_detections.append((pnt_3D, size_3D, score.numpy()))

    if SHOW:
        print("Image inference time: {}s".format(time.time()-start_time))
        RSZ_MOD = 2
        cv2.rectangle(out_image, edge_box[:2], edge_box[2:], (0, 0, 255), thickness=1)
        cv2.imshow("Labelled sub image", cv2.resize(out_image, (out_image.shape[1]//RSZ_MOD, out_image.shape[0]//RSZ_MOD)))
        depth_display = 2*255*(img_depth_np - np.min(img_depth_np)) / np.max(img_depth_np)
        cv2.imshow("Depth", cv2.resize(depth_display.astype(np.uint8), (depth_display.shape[1]//RSZ_MOD, depth_display.shape[0]//RSZ_MOD)))
        key = cv2.waitKey()
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

scallop_pnts_wrld = P.ORTHO_SCALE * np.array([loc for loc, rad, score in scallop_detections])
scallop_sizes = P.ORTHO_SCALE * np.array([size for loc, size, conf in scallop_detections])

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
plt.hist(scallop_sizes, bins=50)
plt.figtext(0.15, 0.85, "Total count: {}".format(scallop_pnts_wrld.shape[0]))
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
ren.AddActor(subActor)
print(extr.GetNumberOfExtractedClusters())

#confs_wrld = points_wrld[:, 7] * 255
#confs_rgb = cv2.applyColorMap(confs_wrld.astype(np.uint8), cv2.COLORMAP_JET)[:, 0, :].astype(np.float32) / 255

iren.Start()