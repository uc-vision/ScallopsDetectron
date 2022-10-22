import Metashape
import numpy as np
import pathlib as p
import os
import math

REBUILD_ORTHO = True
WRITE_TILES = True

TILE_SIZE = 5000

METASHAPE_OUTPUT_BASE = '/local/ScallopReconstructions/'
METASHAPE_OUTPUT_DIR = METASHAPE_OUTPUT_BASE + 'gopro_116_2/'
ORTHO_DIR_PATH = METASHAPE_OUTPUT_DIR + 'ortho/'
try:
    os.mkdir(ORTHO_DIR_PATH)
except OSError as error:
    print(error)
DEM_DIR_PATH = METASHAPE_OUTPUT_DIR + 'dem/'
try:
    os.mkdir(DEM_DIR_PATH)
except OSError as error:
    print(error)
if WRITE_TILES:
    [path.unlink() for path in p.Path(ORTHO_DIR_PATH).iterdir()]
    [path.unlink() for path in p.Path(DEM_DIR_PATH).iterdir()]

doc = Metashape.Document()
doc.open(METASHAPE_OUTPUT_DIR + 'recon.psx')
chunk = doc.chunk

# Camera transformations are in chunk frame
cameras = chunk.cameras
cam_coords = np.array([np.eye(4) if cam.transform == None else np.array(cam.transform).reshape((4, 4)) for cam in cameras])
cam_filenames = [cam.photo.path for cam in cameras]

chunk_scale = chunk.transform.scale or 1
chunk_translation = chunk.transform.translation or np.array([0, 0, 0])
chunk_rotation = np.array(chunk.transform.rotation or np.eye(3))
chunk_rotation = chunk_rotation.reshape((3, 3))
chunk_Q = np.eye(4)
chunk_Q[:3, :3] = chunk_rotation
chunk_Q[:3, 3] = chunk_translation
print("Chunk scale: {}\nchunk translation: {}\nchunk rotation: {}\n".format(chunk_scale, chunk_translation, chunk_rotation))
cam_coords[:, :3, 3] *= chunk_scale
cam_coords = np.matmul(chunk_Q, cam_coords)

c = chunk.sensors[0].calibration
cam_fov = np.array([math.degrees(2*math.atan(c.width / (2*(c.f + c.b1)))),
           math.degrees(2*math.atan(c.height / (2*c.f)))])
print("CAM FOV: {}".format(cam_fov))

cam_height_sum = 0
num_valid_cams = 0
min_height = 10
max_height = 0
dense_pnts = chunk.dense_cloud
for cam_frame in cam_coords:
    if np.array_equal(cam_frame, np.eye(4)) or cam_frame[2, 2] > 0:
        continue
    cam_frame_wrld = np.matmul(np.linalg.inv(chunk_Q), cam_frame)
    cam_frame_wrld[:3, 3] /= chunk_scale

    cam_origin = Metashape.Vector(cam_frame_wrld[:3, 3])
    cam_zray_pnt = cam_origin + Metashape.Vector(np.array([0, 0, -2]))
    pick_pnt = dense_pnts.pickPoint(cam_origin, cam_zray_pnt, endpoints=1)
    if pick_pnt is None:
        continue
    dist = (cam_origin - pick_pnt) * chunk_scale
    z_dist = np.abs(dist[2])
    #print("X, Y, Z distance: {}, {}, {}".format(dist[0], dist[1], dist[2]))
    # if z_dist > 4 or z_dist < 0.1:
    #     continue
    min_height = min(min_height, z_dist)
    max_height = max(max_height, z_dist)
    cam_height_sum += z_dist
    num_valid_cams += 1
print("AVG height: {}".format(cam_height_sum/num_valid_cams))
print("MIN height: {}".format(min_height))
print("MAX height: {}".format(max_height))
print("Input Img shape: {}".format(np.array([c.width, c.height])))
avg_pixel_scale = 2 * cam_height_sum/num_valid_cams * np.tan(np.deg2rad(cam_fov/2)) / np.array([c.width, c.height])
print("AVG Pixel Scale (mm): {}".format(1000*avg_pixel_scale))
pix_scale = np.max(avg_pixel_scale)

mean_cam_vec = np.mean(cam_coords[:, :3, 2], axis=0)
cam_min_xy = np.array([np.min(cam_coords[:, 0, 3]), np.min(cam_coords[:, 1, 3])])
cam_max_xy = np.array([np.max(cam_coords[:, 0, 3]), np.max(cam_coords[:, 1, 3])])
print("\nMean camera vector wrld: {}\nMin xy: {}\nMax xy: {}\n".format(mean_cam_vec, cam_min_xy, cam_max_xy))

region1 = Metashape.BBox()
cam_min_xy -= np.array([3, 3])
region_min = cam_min_xy
cam_max_xy += np.array([3, 3])
region_max = np.sign(cam_max_xy - cam_min_xy) * np.ceil(np.abs((cam_max_xy - cam_min_xy) / (TILE_SIZE * pix_scale))) * TILE_SIZE * pix_scale + cam_min_xy
print((region_max - region_min) / (TILE_SIZE * pix_scale))
region1.min = Metashape.Vector(region_min)
region1.max = Metashape.Vector(region_max)

print("Rebuilding DEM and Ortho...")
if WRITE_TILES and REBUILD_ORTHO:
    chunk.buildDem(region=region1, resolution=pix_scale)
    chunk.buildOrthomosaic(surface_data=Metashape.ElevationData, region=region1, resolution=pix_scale)
    doc.read_only = False
    doc.save()

if WRITE_TILES:
    chunk.exportRaster(DEM_DIR_PATH+"dem_"+".tif", source_data=Metashape.ElevationData, image_format=Metashape.ImageFormatTIFF,
                       format=Metashape.RasterFormatTiles, split_in_blocks=True, block_height=TILE_SIZE, block_width=TILE_SIZE, resolution=pix_scale,
                       white_background=False, region=region1)
    chunk.exportRaster(ORTHO_DIR_PATH+"ortho_"+".tif", source_data=Metashape.OrthomosaicData, image_format=Metashape.ImageFormatTIFF,
                       format=Metashape.RasterFormatTiles, split_in_blocks=True, block_height=TILE_SIZE, block_width=TILE_SIZE, resolution=pix_scale,
                       white_background=False, region=region1)

    # Delete non-intersecting DEM and Ortho tiles
    ortho_paths = [(path.name[:-4].split('_')[-1], str(path)) for path in p.Path(METASHAPE_OUTPUT_DIR + 'ortho/').iterdir()]
    dem_paths = [(path.name[:-4].split('_')[-1], str(path)) for path in p.Path(METASHAPE_OUTPUT_DIR + 'dem/').iterdir()]
    valid_inds = set(t[0] for t in ortho_paths).intersection(set(t[0] for t in dem_paths))
    [p.Path(path_t[1]).unlink() for path_t in ortho_paths if path_t[0] not in valid_inds]
    [p.Path(path_t[1]).unlink() for path_t in dem_paths if path_t[0] not in valid_inds]

# with open(METASHAPE_OUTPUT_DIR + 'cam_filenames.txt', 'w') as f:
#     f.write('\n'.join(cam_filenames))
# f.close()
#
# np.save(METASHAPE_OUTPUT_DIR + 'pix_scale.npy', pix_scale)
#
# np.save(METASHAPE_OUTPUT_DIR + 'cam_coords.npy', cam_coords)
# ortho_origin = np.array([region1.min[0], region1.max[1], 0])
# np.save(METASHAPE_OUTPUT_DIR + 'ortho_origin.npy', ortho_origin)
#
# np.save(METASHAPE_OUTPUT_DIR + 'chunk_quart.npy', chunk_Q)
# np.save(METASHAPE_OUTPUT_DIR + 'chunk_scale.npy', chunk_scale)
#
# # np.save("/local/ScallopMaskDataset/output/cam_fov.npy", cam_fov)
# K = np.array([
#     [c.f + c.b1,    c.b2,   c.cx + c.width / 2],
#     [0,             c.f,    c.cy + c.height / 2],
#     [0,             0,      1]
# ])
# dist = np.array([[c.k1, c.k2, c.p2, c.p1, c.k3]])
# np.save(METASHAPE_OUTPUT_DIR + 'camMtx.npy', K)
# np.save(METASHAPE_OUTPUT_DIR + 'camDist.npy', dist) # k1 k2 p1 p2
