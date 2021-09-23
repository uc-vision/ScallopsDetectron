import Metashape
import numpy as np
import pathlib as p
import os

WORKSPACE = '/home/tim/Data/ROV_recons/'
PHOTO_DIR = '/media/tim/UData/2020ScallopSurveys/lowres_scan_210114_023202/left/'

RECON_FOLDER = 'recon_1/'

#TODO: add photos
#TODO: constrain sequential images / start/end transformation?
#TODO: align photos
#TODO: build dense cloud
#TODO: build mesh
#TODO: texturise mesh

doc = Metashape.Document()
#doc.open(P.METASHAPE_CHKPNT_PATH)
chunk = doc.addChunk()

img_paths = [str(pth) for pth in p.Path(PHOTO_DIR).rglob('*.png')]
img_paths.sort()
print("Number of photos: {}".format(len(img_paths)))

chunk.addPhotos(img_paths[:100])
chunk.matchPhotos()
chunk.alignCameras()
chunk.buildDepthMaps(downscale=4, filter_mode=Metashape.AggressiveFiltering)
chunk.buildDenseCloud()
chunk.buildModel(surface_type=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation)
chunk.buildUV(mapping_mode=Metashape.GenericMapping)
chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=4096)

region = chunk.region
r_center = region.center
r_rotate = np.array(region.rot).reshape((3, 3))
r_size = region.size
bbox_min = r_center - r_size/2
bbox_max = r_center + r_size/2
region1 = Metashape.BBox()
region1.min = Metashape.Vector([bbox_min[0], bbox_min[1]])
region1.max = Metashape.Vector([bbox_max[0], bbox_max[1]])

chunk.buildDem()
chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)

chunk.exportRaster("dem_"+".tif", source_data=Metashape.ElevationData, image_format=Metashape.ImageFormatTIFF,
                   format=Metashape.RasterFormatTiles, split_in_blocks=True, block_height=2000, block_width=2000, resolution=0.01,
                   white_background=False, region=region1)
chunk.exportRaster("ortho_"+".tif", source_data=Metashape.OrthomosaicData, image_format=Metashape.ImageFormatTIFF,
                   format=Metashape.RasterFormatTiles, split_in_blocks=True, block_height=2000, block_width=2000, resolution=0.01,
                   white_background=False, region=region1)
