import numpy as np
import pathlib as p
import cv2
import math
from tqdm import tqdm
import Params as P
import vtk
import VTKPointCloud as PC

SHOW_ORTHOS = False
SHOW = False


ortho_paths = [str(path) for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'ortho/').iterdir()]
dem_paths = [str(path) for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'dem/').iterdir()]
dem_paths.sort()
ortho_paths.sort()
assert len(ortho_paths) == len(dem_paths)
ortho_origin = np.load(P.METASHAPE_OUTPUT_DIR + 'ortho_origin.npy')

def LoadTiles(paths, dims=3, read_flags=None, d_type=np.uint8):
    tile_offsets_rc = np.array([[float(path.split('-')[-1][:-4]), float(path.split('-')[-2])] for path in paths])
    tile_offsets_rc *= P.TILE_SIZE
    tile_extents = tile_offsets_rc + P.TILE_SIZE
    max_extents = np.max(tile_extents, axis=0).astype(int)
    ortho_full = np.zeros((max_extents[0], max_extents[1], dims), dtype=d_type)
    full_shape = ortho_full.shape
    print("Ortho Shape: {}, loading tiles...".format(full_shape))
    for ortho_tile_path, offset in tqdm(list(zip(paths, tile_offsets_rc.astype(int)))):
        ortho_tile = cv2.imread(ortho_tile_path, read_flags)
        tile_shape = ortho_tile.shape
        ortho_full[offset[0]:(offset[0]+tile_shape[0]), offset[1]:(offset[1]+tile_shape[1])] = ortho_tile.reshape((tile_shape[0], tile_shape[1], -1))
    return ortho_full, full_shape

ortho_full, shape = LoadTiles(ortho_paths)
dem_full, shape = LoadTiles(dem_paths, dims=1, read_flags=cv2.IMREAD_ANYDEPTH, d_type=np.float32)

THRESHOLD = 0.8
def MorphThresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_norm = gray / np.max(gray)
    thresh = (gray_norm > THRESHOLD).astype(np.uint8)
    return thresh

print("Extracting points...")
thesh_ortho = MorphThresh(ortho_full)
proj_pnts = np.where(thesh_ortho.astype(bool))
elavations = np.clip(dem_full[proj_pnts], -10, 10)
proj_pnts_a = np.array(proj_pnts)[::-1, :].astype(np.float32)
pnts_wrld_2D = proj_pnts_a * P.PIXEL_SCALE
pnts_wrld_2D[1, :] *= -1
pnts_wrld = np.hstack([pnts_wrld_2D.transpose(), elavations]) + ortho_origin

pnt_cld = PC.VtkPointCloud(pnt_size=1)
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

len_pnts = pnts_wrld.shape[0]
pnt_cld.setPoints(pnts_wrld, np.array(len_pnts*[[0, 1, 0]]))
iren.Start()

print("Pnts array shape: {}".format(pnts_wrld.shape))

def worldPntToCamVec(pnts_wrld, cam_quart):
    ann_vecs_cam = np.matmul(np.linalg.inv(cam_quart), np.vstack([pnts_wrld.transpose(), np.ones((1, pnts_wrld.shape[0]))]))
    return ann_vecs_cam[:3, :]

def CamVecToPixCoord(pnts_cam, cam_mtx):
    pix_dash = np.matmul(cam_mtx, pnts_cam)
    return (pix_dash / pix_dash[2, :])[:2, :]


cam_coords = np.load(P.METASHAPE_OUTPUT_DIR + 'cam_coords.npy')
with open(P.METASHAPE_OUTPUT_DIR + 'cam_filenames.txt') as f:
    cam_filenames = f.read().splitlines()
f.close()
camMtx = np.load(P.METASHAPE_OUTPUT_DIR + 'camMtx.npy')
camDist = np.load(P.METASHAPE_OUTPUT_DIR + 'camDist.npy')
x0, y0 = camMtx[:2, 2]
h,  w = (2160, 3840)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camMtx, camDist, (w,h), 0, (w,h))
print(roi)
w_ud, h_ud = roi[2:]

print("Projecting into Cams...")
total_mismatch_score = 0
for idx, (cam_frame, cam_img_path) in tqdm(enumerate(list(zip(cam_coords, cam_filenames)))):
    img_cam = cv2.imread(cam_img_path)
    img_cam = cv2.undistort(img_cam, cameraMatrix=camMtx, distCoeffs=camDist, newCameraMatrix=newcameramtx)

    img_shape = img_cam.shape
    pnts_cam = worldPntToCamVec(pnts_wrld, cam_frame)
    valid_cam_pnts = pnts_cam[:, np.where((np.linalg.norm(pnts_cam, axis=0) < 100) * (np.linalg.norm(pnts_cam, axis=0) > 0.001))][:, 0, :]
    pix_coords = CamVecToPixCoord(valid_cam_pnts, newcameramtx)
    valid_pix_coords = pix_coords[:, np.where((pix_coords[0, :] >= 0) * (pix_coords[0, :] < w_ud) *
                                              (pix_coords[1, :] >= 0) * (pix_coords[1, :] < h_ud))][:, 0, :].astype(np.int)
    cam_thresh = MorphThresh(img_cam).astype(np.float32)
    proj_img = np.zeros_like(cam_thresh)
    proj_img[tuple(valid_pix_coords[::-1, :].tolist())] = 1
    total_proj_pixels = np.where(proj_img == 1)
    mismatched_pixels = np.where(cam_thresh != proj_img)
    matched_pixels = np.where(cam_thresh == proj_img)
    mismatch_score = len(mismatched_pixels[0]) / len(valid_pix_coords[0])
    total_mismatch_score += mismatch_score

    if SHOW:
        img_cam[tuple(valid_pix_coords[::-1, :].tolist())] = (0, 0, 255)
        img_cam[mismatched_pixels] = (0, 255, 0)
        display = np.zeros_like(cam_thresh)
        display[np.where(cam_thresh > 0)] += 0.65
        display[np.where(img_cam[:, :, 2] > 254)] += 0.35
        cv2.imshow("display", cv2.resize(display, (display.shape[1]//2, display.shape[0]//2)))

        cv2.imshow("Cam thresh", cv2.resize(cam_thresh*255, (img_cam.shape[1]//2, img_cam.shape[0]//2)))
        cv2.imshow("Display", cv2.resize(img_cam, (img_cam.shape[1]//2, img_cam.shape[0]//2)))
        key = cv2.waitKey()
        if key == ord('q'):
            break

mismatch_avg = total_mismatch_score / (idx + 1)
print("Mismatch score: {}".format(mismatch_avg))