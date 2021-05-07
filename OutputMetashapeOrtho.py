import Metashape
import numpy as np
import pathlib as p
import Params as P

ORTHO_DIR_PATH = P.METASHAPE_OUTPUT_DIR + 'ortho/'
DEM_DIR_PATH = P.METASHAPE_OUTPUT_DIR + 'dem/'
[path.unlink() for path in p.Path(ORTHO_DIR_PATH).iterdir()]
[path.unlink() for path in p.Path(DEM_DIR_PATH).iterdir()]

doc = Metashape.Document()
doc.open(P.METASHAPE_CHKPNT_PATH)
chunk = doc.chunk
cameras = chunk.cameras
cam_coords = [np.array(cam.transform).reshape((4, 4)) for cam in cameras]
cam_filenames = [cam.photo.path for cam in cameras]

region = chunk.region
r_center = region.center
r_rotate = region.rot
r_size = region.size
ortho_origin = np.array([r_center[0] - r_size[0]/2, r_center[1] + r_size[1]/2, 0])

bbox_min = r_center - r_size/2
bbox_max = r_center + r_size/2
region1 = Metashape.BBox()
region1.min = Metashape.Vector([bbox_min[0], bbox_min[1]])
region1.max = Metashape.Vector([bbox_max[0], bbox_max[1]])

chunk.exportRaster(ORTHO_DIR_PATH+"ortho_"+".tif", source_data=Metashape.OrthomosaicData, image_format=Metashape.ImageFormatTIFF,
                   format=Metashape.RasterFormatTiles, split_in_blocks=True, block_height=P.TILE_SIZE, block_width=P.TILE_SIZE, resolution=P.PIXEL_SCALE,
                   white_background=False, region=region1)
chunk.exportRaster(DEM_DIR_PATH+"dem_"+".tif", source_data=Metashape.ElevationData, image_format=Metashape.ImageFormatTIFF,
                   format=Metashape.RasterFormatTiles, split_in_blocks=True, block_height=P.TILE_SIZE, block_width=P.TILE_SIZE, resolution=P.PIXEL_SCALE,
                   white_background=False, region=region1)

with open(P.METASHAPE_OUTPUT_DIR + 'cam_filenames.txt', 'w') as f:
    f.write('\n'.join(cam_filenames))
f.close()

np.save(P.METASHAPE_OUTPUT_DIR + 'cam_coords.npy', cam_coords)
np.save(P.METASHAPE_OUTPUT_DIR + 'ortho_origin.npy', ortho_origin)

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