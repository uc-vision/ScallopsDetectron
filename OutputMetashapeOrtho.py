import Metashape
import numpy as np
import pathlib as p
import Params as P
import os

WRITE_TILES = True

ORTHO_DIR_PATH = P.METASHAPE_OUTPUT_DIR + 'ortho/'
DEM_DIR_PATH = P.METASHAPE_OUTPUT_DIR + 'dem/'
if WRITE_TILES:
    [path.unlink() for path in p.Path(ORTHO_DIR_PATH).iterdir()]
    [path.unlink() for path in p.Path(DEM_DIR_PATH).iterdir()]

doc = Metashape.Document()
doc.open(P.METASHAPE_CHKPNT_PATH)
chunk = doc.chunk

chunk.orthomosaic.crs = chunk.world_crs
chunk.elevation.crs = chunk.world_crs
chunk.orthomosaic.update()

cameras = chunk.cameras
cam_coords = [np.eye(4) if cam.transform == None else np.array(cam.transform).reshape((4, 4)) for cam in cameras]
cam_filenames = [cam.photo.path for cam in cameras]
chunk_scale = chunk.transform.scale
chunk_rotation = np.array(chunk.transform.rotation).reshape((3, 3))

region = chunk.region
r_center = region.center
r_rotate = np.array(region.rot).reshape((3, 3))
r_size = region.size

chunk_translation = chunk.transform.translation
cam_coords = np.array(cam_coords)
cam_coords[:, :3, :4] = np.matmul(chunk_rotation, cam_coords[:, :3, :4])
cam_coords[:, :3, 3] *= chunk_scale
cam_coords[:, :3, 3] += chunk_translation

ortho_chunk_rot = chunk_rotation#np.matmul(chunk_rotation, r_rotate)
print(ortho_chunk_rot)
#print(chunk_rotation)

# ortho_rot = chunk.orthomosaic.projection.matrix
# print(ortho_rot)
#ortho = chunk.orthomosaic
#ortho_origin = np.array([ortho.left, ortho.top, 0])
#print(ortho_origin)
#ortho_proj = ortho.projection.matrix
#print(ortho_proj)
print(region.center)
#print(region.size)
ortho_topleft = np.array([r_center[0]-r_size[0]/2, r_center[1]+r_size[1]/2, 0])
print(chunk_translation)
print(ortho_topleft)
#print(ortho_origin)
#exit(0)

ortho_ori = np.array([chunk.orthomosaic.left, chunk.orthomosaic.top, 0])
print(ortho_ori)

bbox_min = r_center - r_size/2
bbox_max = r_center + r_size/2
region1 = Metashape.BBox()
region1.min = Metashape.Vector([bbox_min[0], bbox_min[1]])
region1.max = Metashape.Vector([bbox_max[0], bbox_max[1]])
#print(region1.min)
#print(region1.max)
#

print("Rebuilding DEM and Ortho...")
chunk.buildDem()
chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)

doc.read_only = False
doc.save()
# exit(0)

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

with open(P.METASHAPE_OUTPUT_DIR + 'cam_filenames.txt', 'w') as f:
    f.write('\n'.join(cam_filenames))
f.close()

np.save(P.METASHAPE_OUTPUT_DIR + 'cam_coords.npy', cam_coords)
np.save(P.METASHAPE_OUTPUT_DIR + 'ortho_origin.npy', ortho_topleft)

np.save(P.METASHAPE_OUTPUT_DIR + 'chunk_trans.npy', chunk_translation)
np.save(P.METASHAPE_OUTPUT_DIR + 'ortho_rotmat.npy', r_rotate)
np.save(P.METASHAPE_OUTPUT_DIR + 'chunk_rotmat.npy', chunk_rotation)

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