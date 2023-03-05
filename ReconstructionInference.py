import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import pathlib, os
from detectron2.config import get_cfg
from utils import VTKPointCloud as PC, polygon_functions as spf
import vtk
import Metashape
print("Metashape version {}".format(Metashape.version))
import time
from datetime import datetime
import glob
from shapely.geometry import Polygon, Point
import geopandas as gpd

UD_ALPHA = 0
IMG_RS_MOD = 2
CAM_IDX_LIMIT = -1

SCALE_MUL = 1.0
LINE_DIST_THRESH = 1.0
EDGE_LIMIT_PIX = 200 // IMG_RS_MOD
OUTLIER_RADIUS = 0.1

IMSHOW = False
VTK = False
WAITKEY = 0

def CamToChunk(pnts_cam, cam_quart):
    return np.matmul(cam_quart, np.vstack([pnts_cam, np.ones((1, pnts_cam.shape[1]))]))[:3, :]

def CamPixToChunkPnt(pixels_cam, cam_mtx):
    return np.matmul(np.linalg.inv(cam_mtx), np.vstack([pixels_cam, np.ones((1, pixels_cam.shape[1]))]))

def draw_scaled_axes(img, axis_vecs, axis_scales, origin, cam_mtx):
    points = np.concatenate([np.multiply(axis_vecs, np.repeat(axis_scales[:, None], 3, axis=1)) +
                             np.repeat(origin[None, :], 3, axis=0), origin[None, :]], axis=0)
    axis_points, _ = cv2.projectPoints(points, np.zeros((1, 3)), np.zeros((1, 3)), cam_mtx, None)
    axis_points = axis_points.astype(np.int)
    cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[2].ravel()), (255, 0, 0), 3)
    cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[1].ravel()), (0, 255, 0), 3)
    cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[0].ravel()), (0, 0, 255), 3)

METASHAPE_OUTPUT_BASE = '/local/ScallopReconstructions/'
RECONSTRUCTION_DIRS = [METASHAPE_OUTPUT_BASE + 'gopro_128_nocrop/']
#                        METASHAPE_OUTPUT_BASE + 'gopro_118/',
#                        METASHAPE_OUTPUT_BASE + 'gopro_123/',
#                        METASHAPE_OUTPUT_BASE + 'gopro_124/',
#                        METASHAPE_OUTPUT_BASE + 'gopro_125/']

MODEL_PATH = "/local/ScallopMaskRCNNOutputs/HR+LR LP AUGS/"

cfg = get_cfg()
cfg.NUM_GPUS = 1
cfg.merge_from_file(MODEL_PATH + 'config.yml')
model_paths = [str(path) for path in pathlib.Path(MODEL_PATH).glob('*.pth')]
model_paths.sort()
cfg.MODEL.WEIGHTS = os.path.join(model_paths[-1])

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
cfg.TEST.AUG.ENABLED = False
cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
cfg.TEST.AUG.MAX_SIZE = 4000
cfg.TEST.AUG.FLIP = False
cfg.TEST.PRECISE_BN.ENABLED = False
cfg.TEST.PRECISE_BN.NUM_ITER = 200
predictor = DefaultPredictor(cfg)

if IMSHOW:
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

for RECON_DIR in RECONSTRUCTION_DIRS:
    doc = Metashape.Document()
    doc.open(RECON_DIR + "recon.psx")
    #doc.read_only = False
    chunk = doc.chunks[0]
    chunk_scale = SCALE_MUL * chunk.transform.scale or 1
    chunk_transform = chunk.transform.matrix
    chunk_T_inv = np.array(chunk_transform.inv()).reshape((4, 4))
    print("Chunk scale: {}".format(chunk_scale))
    cameras = chunk.cameras

    shape_files = glob.glob(RECON_DIR + '*.gpkg')
    for shape_file in shape_files:
        chunk.importShapes(shape_file)
        lbl = shape_file.split('/')[-1].split('.')[0]
        chunk.shapes.groups[-1].label = lbl

    if chunk.shapes is None:
        chunk.shapes = Metashape.Shapes()
        chunk.shapes.crs = Metashape.CoordinateSystem("EPSG::4326")
    shapes = chunk.shapes.shapes
    shapes_crs = chunk.shapes.crs
    # chunk_marker_dict = {v.key: v.position for v in chunk.markers}
    rope_lines = [shape for shape in shapes if shape.geometry.type == Metashape.Geometry.Type.LineStringType]
    has_roperefs = len(rope_lines) > 0
    rope_lines_chunk = [np.array(line.geometry.coordinates) for line in rope_lines] if len(rope_lines) else [None]
    ref_line = rope_lines_chunk[0]

    if chunk.model is None:
        chunk.buildModel()
        doc.save()

    prediction_geometries = []
    prediction_markers = []
    prediction_labels = []
    for cam in tqdm(cameras[3::1][:CAM_IDX_LIMIT]):
        if cam.transform is None:
            continue

        start_time = time.time()

        # Images from metashape are distorted including depth image
        cam_img_m = cam.image()
        img_path = cam.photo.path
        img_fn = img_path.split('/')[-1]

        c = cam.calibration
        camMtx = np.array([[c.f + c.b1, c.b2, c.cx + c.width / 2],
                           [0, c.f, c.cy + c.height / 2],
                           [0, 0, 1]])
        camMtx[:2, :] /= IMG_RS_MOD
        camDist = np.array([[c.k1, c.k2, c.p2, c.p1, c.k3]])
        img = np.frombuffer(cam_img_m.tostring(), dtype=np.uint8).reshape((int(cam_img_m.height), int(cam_img_m.width), -1))[:, :, ::-1]
        img_rs = cv2.resize(img, (img.shape[1] // IMG_RS_MOD, img.shape[0] // IMG_RS_MOD))
        # img_cam_ud = cv2.undistort(img_rs, cameraMatrix=camMtx, distCoeffs=camDist)
        rs_shape = img_rs.shape

        cam_quart = np.array(cam.transform).reshape((4, 4))

        img_depth_ms = chunk.model.renderDepth(cam.transform, cam.sensor.calibration, add_alpha=False)
        #img_depth_ms = chunk.depth_maps[cam].image()
        img_depth_np = np.frombuffer(img_depth_ms.tostring(), dtype=np.float32).reshape((img_depth_ms.height, img_depth_ms.width, 1))
        img_depth_np = cv2.resize(img_depth_np, (rs_shape[1], rs_shape[0]))
        img_depth_np = cv2.blur(img_depth_np, ksize=(51, 51))

        # img_depth_display = np.repeat(img_depth_np[:, :, None], 3, axis=2) - np.min(img_depth_np)
        # img_depth_display /= np.max(img_depth_display) + 1e-9
        # cv2.imshow("depth", img_depth_display)
        # cv2.imshow("img", img_rs)
        # cv2.imshow("overlap", img_rs.astype(np.float32) / 510 + img_depth_display)
        # cv2.imshow("overlap_ud", img_cam_ud.astype(np.float32) / 510 + img_depth_display)
        # img_from_file = cv2.imread(img_path)
        # cv2.imshow("img from file", cv2.resize(img_from_file, (img.shape[1] // IMG_RS_MOD, img.shape[0] // IMG_RS_MOD)))
        # cv2.imshow("img from metashape", img_rs)
        # cv2.waitKey()
        # exit(0)

        edge_box = (EDGE_LIMIT_PIX, EDGE_LIMIT_PIX, rs_shape[1]-EDGE_LIMIT_PIX, rs_shape[0]-EDGE_LIMIT_PIX)

        outputs = predictor(img_rs)
        instances = outputs["instances"].to("cpu")
        masks = instances._fields['pred_masks']
        bboxes = instances._fields['pred_boxes']
        scores = instances._fields['scores']
        if IMSHOW:
            v = Visualizer(img_rs[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
            v = v.draw_instance_predictions(instances)
            out_image = v.get_image()[:, :, ::-1].copy()
        for mask, box, score in list(zip(masks, bboxes, scores)):
            mask_pnts = np.array(np.where(mask))[::-1].transpose()
            scallop_centre, radius = cv2.minEnclosingCircle(mask_pnts)
            scallop_centre = np.array(scallop_centre, dtype=int)
            if edge_box[0] <= scallop_centre[0] <= edge_box[2] and edge_box[1] <= scallop_centre[1] <= edge_box[3]:
                mask_np = mask.numpy()[:, :, None].astype(np.uint8)
                contours, hierarchy = cv2.findContours(mask_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                scallop_polygon = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])][:, 0]
                # Clip number of vertices in polygon to 30->60
                scallop_polygon = scallop_polygon[::(1 + scallop_polygon.shape[0] // 60)]
                if IMSHOW:
                    cv2.circle(out_image, (scallop_centre[0], scallop_centre[1]), int(radius), color=(0, 255, 0), thickness=2)
                    cv2.drawContours(out_image, contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

                # Undistort polygon vertices
                vert_elavations = img_depth_np[scallop_polygon[:, 1], scallop_polygon[:, 0]]
                pxpoly_ud = spf.undistort_pixels(scallop_polygon, camMtx, camDist)
                scallop_poly_cam = CamPixToChunkPnt(pxpoly_ud.T, camMtx)
                scallop_poly_cam = scallop_poly_cam * vert_elavations.T

                mask_pnts_sub = mask_pnts[::10]
                vert_elavations = img_depth_np[mask_pnts_sub[:, 1], mask_pnts_sub[:, 0]]
                mask_pnts_ud = spf.undistort_pixels(mask_pnts_sub, camMtx, camDist)
                scallop_pnts_cam = CamPixToChunkPnt(mask_pnts_sub.T, camMtx)
                scallop_pnts_cam = scallop_pnts_cam * vert_elavations.T
                scallop_pnts_cam = spf.remove_outliers(scallop_pnts_cam, OUTLIER_RADIUS / chunk_scale)

                scallop_polygon_chunk = CamToChunk(scallop_poly_cam, cam_quart)
                scallop_center_chunk = CamToChunk(scallop_pnts_cam, cam_quart).mean(axis=1)
                if ref_line:
                    within_transect = spf.polyline_dist_thresh(scallop_center_chunk, ref_line, LINE_DIST_THRESH / chunk_scale)
                else:
                    within_transect = True
                if within_transect:
                    scallop_center_cam = scallop_poly_cam.mean(axis=1)
                    cam_center_offset_cos = scallop_center_cam[2] / np.linalg.norm(scallop_center_cam)
                    cam_center_offset_cos = -1 if np.isnan(cam_center_offset_cos) else cam_center_offset_cos
                    polygon = []
                    for pnt in scallop_polygon_chunk.T:
                        pnt_wrld = shapes_crs.project(chunk_transform.mulp(Metashape.Vector(pnt)))
                        polygon.append(pnt_wrld)
                    prediction_geometries.append(Polygon(polygon))
                    prediction_markers.append(Point(np.mean(polygon, axis=0)))
                    prediction_labels.append(str({"label": "live", "conf": round(score.item(), 2), "cntr_cos": round(cam_center_offset_cos, 2)}))

                if IMSHOW and scallop_pnts_cam.shape[1] > 1:
                    pc_vecs, pc_lengths, center_pnt = spf.pca(scallop_pnts_cam.T)
                    MUL = 1.9
                    pc_lengths = np.sqrt(pc_lengths) * MUL
                    scaled_pc_lengths = pc_lengths * chunk_scale * 2
                    width_scallop_circle = 2 * chunk_scale * scallop_pnts_cam[2, :].mean() * radius / camMtx[0, 0]

                    txt_col = (255, 255, 255) if within_transect else (0, 0, 255)
                    cv2.putText(out_image, str(round(scaled_pc_lengths[0], 3)), tuple(scallop_centre + np.array([20, -10])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, txt_col, 4, cv2.LINE_AA)
                    cv2.putText(out_image, str(round(width_scallop_circle, 3)), tuple(scallop_centre + np.array([20, 30])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, txt_col, 4, cv2.LINE_AA)
                    draw_scaled_axes(out_image, pc_vecs, pc_lengths, center_pnt, camMtx)

                if VTK:
                    pnt_cld.setPoints(scallop_pnts_cam.T - center_pnt, np.array([1, 1, 1] * scallop_pnts_cam.shape[1]).T)
                    axes_transform_np[:3, :3] = np.multiply(pc_vecs, np.repeat(pc_lengths[:, None], 3, axis=1)).T
                    axes_matrix.DeepCopy(axes_transform_np.ravel())
                    vtk_axes.Modified()
                    iren.Render()
                    iren.Start()

        if IMSHOW:
            #print("Image inference time: {}s".format(time.time()-start_time))
            cv2.rectangle(out_image, edge_box[:2], edge_box[2:], (0, 0, 255), thickness=1)
            cv2.imshow("Input image", img_rs)
            cv2.imshow("Labelled sub image", out_image)
            depth_display = 255*(img_depth_np - np.min(img_depth_np)) / np.max(img_depth_np)
            cv2.imshow("Depth", depth_display.astype(np.uint8))
            key = cv2.waitKey(WAITKEY)
            if key == ord('q'):
                exit(0)

    #doc.save()
    shapes_fn = RECON_DIR+'Pred_' + datetime.now().strftime("%d%m%y_%H%M")
    shapes_fn_3d = shapes_fn + '_3D.gpkg'
    gdf_3D = gpd.GeoDataFrame({'geometry': prediction_geometries, 'NAME': prediction_labels}, geometry='geometry', crs=shapes_crs.name)
    gdf_3D.to_file(shapes_fn_3d)
    markers_fn_3d = shapes_fn + '_3D_markers.gpkg'
    gdf_3D = gpd.GeoDataFrame({'geometry': prediction_markers, 'NAME': prediction_labels}, geometry='geometry', crs=shapes_crs.name)
    gdf_3D.to_file(markers_fn_3d)

    # Save shapes in 2D also
    gdf = gpd.read_file(shapes_fn_3d)
    new_geo = []
    for polygon in gdf.geometry:
        if polygon.has_z:
            assert polygon.geom_type == 'Polygon'
            lines = [xy[:2] for xy in list(polygon.exterior.coords)]
            new_geo.append(Polygon(lines))
    gdf.geometry = new_geo
    gdf.to_file(shapes_fn + '_2D.gpkg')
