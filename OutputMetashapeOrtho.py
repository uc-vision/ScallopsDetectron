import Metashape
import numpy as np
import pathlib as p
import Params as P
import os

WRITE_TILES = True

ORTHO_DIR_PATH = P.METASHAPE_OUTPUT_DIR + 'ortho/'
try:
    os.mkdir(ORTHO_DIR_PATH)
except OSError as error:
    print(error)
DEM_DIR_PATH = P.METASHAPE_OUTPUT_DIR + 'dem/'
try:
    os.mkdir(DEM_DIR_PATH)
except OSError as error:
    print(error)
if WRITE_TILES:
    [path.unlink() for path in p.Path(ORTHO_DIR_PATH).iterdir()]
    [path.unlink() for path in p.Path(DEM_DIR_PATH).iterdir()]

doc = Metashape.Document()
doc.open(P.METASHAPE_CHKPNT_PATH)
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

mean_cam_vec = np.mean(cam_coords[:, :3, 2], axis=0)
cam_min_xy = np.array([np.min(cam_coords[:, 0, 3]), np.min(cam_coords[:, 1, 3])])
cam_max_xy = np.array([np.max(cam_coords[:, 0, 3]), np.max(cam_coords[:, 1, 3])])
print("\nMean camera vector wrld: {}\nMin xy: {}\nMax xy: {}\n".format(mean_cam_vec, cam_min_xy, cam_max_xy))

region1 = Metashape.BBox()
cam_min_xy -= np.array([3, 3])
region_min = cam_min_xy
cam_max_xy += np.array([3, 3])
region_max = np.sign(cam_max_xy - cam_min_xy) * np.ceil(np.abs((cam_max_xy - cam_min_xy) / (P.TILE_SIZE * P.PIXEL_SCALE))) * P.TILE_SIZE * P.PIXEL_SCALE + cam_min_xy
print((region_max - region_min) / (P.TILE_SIZE * P.PIXEL_SCALE))
region1.min = Metashape.Vector(region_min)
region1.max = Metashape.Vector(region_max)

print("Rebuilding DEM and Ortho...")
if WRITE_TILES:
    chunk.buildDem(region=region1, resolution=P.PIXEL_SCALE)
    chunk.buildOrthomosaic(surface_data=Metashape.ElevationData, region=region1, resolution=P.PIXEL_SCALE)

doc.read_only = False
doc.save()

if WRITE_TILES:
    chunk.exportRaster(DEM_DIR_PATH+"dem_"+".tif", source_data=Metashape.ElevationData, image_format=Metashape.ImageFormatTIFF,
                       format=Metashape.RasterFormatTiles, split_in_blocks=True, block_height=P.TILE_SIZE, block_width=P.TILE_SIZE, resolution=P.PIXEL_SCALE,
                       white_background=False, region=region1)
    chunk.exportRaster(ORTHO_DIR_PATH+"ortho_"+".tif", source_data=Metashape.OrthomosaicData, image_format=Metashape.ImageFormatTIFF,
                       format=Metashape.RasterFormatTiles, split_in_blocks=True, block_height=P.TILE_SIZE, block_width=P.TILE_SIZE, resolution=P.PIXEL_SCALE,
                       white_background=False, region=region1)

    # Ensure ortho and DEM files line up
    ortho_paths = [(str(path), path.name) for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'ortho/').iterdir()]
    dem_paths = [(str(path), path.name) for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'dem/').iterdir()]
    dem_paths.sort()
    ortho_paths.sort()
    for idx, (ortho_path, ortho_fn) in enumerate(ortho_paths):
        if ortho_fn[6:] != dem_paths[idx][1][4:]:
            print("Found ortho dem tile mismatch! fn: {}".format(dem_paths[idx][1]))
            p.Path(dem_paths[idx][0]).unlink()
            dem_paths.pop(idx)
    while len(dem_paths) > len(ortho_paths):
        p.Path(dem_paths[-1][0]).unlink()
        dem_paths.pop()

with open(P.METASHAPE_OUTPUT_DIR + 'cam_filenames.txt', 'w') as f:
    f.write('\n'.join(cam_filenames))
f.close()

np.save(P.METASHAPE_OUTPUT_DIR + 'cam_coords.npy', cam_coords)
ortho_origin = np.array([region1.min[0], region1.max[1], 0])
np.save(P.METASHAPE_OUTPUT_DIR + 'ortho_origin.npy', ortho_origin)

# np.save(P.METASHAPE_OUTPUT_DIR + 'chunk_trans.npy', chunk_translation)
# np.save(P.METASHAPE_OUTPUT_DIR + 'ortho_rotmat.npy', chunk_rotation)
# np.save(P.METASHAPE_OUTPUT_DIR + 'chunk_rotmat.npy', chunk_rotation)

c = chunk.sensors[0].calibration
# cam_fov = [math.degrees(2*math.atan(c.width / (2*(c.f + c.b1)))),
#            math.degrees(2*math.atan(c.height / (2*c.f)))]
# np.save("/local/ScallopMaskDataset/output/cam_fov.npy", cam_fov)
K = np.array([
    [c.f + c.b1,    c.b2,   c.cx + c.width / 2],
    [0,             c.f,    c.cy + c.height / 2],
    [0,             0,      1]
])
dist = np.array([[c.k1, c.k2, c.p2, c.p1, c.k3]])
np.save(P.METASHAPE_OUTPUT_DIR + 'camMtx.npy', K)
np.save(P.METASHAPE_OUTPUT_DIR + 'camDist.npy', dist) # k1 k2 p1 p2