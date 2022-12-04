import Metashape
import numpy as np
import pathlib as p
import os
import math
import time
import logging
from datetime import datetime
from utils import file_utils, metashape_utils

DEM_RES = 0.005
ORTHO_RES = 0.0005

METASHAPE_OUTPUT_BASE = '/local/ScallopReconstructions/' #'/scratch/data/tkr25/Reconstructions/' #
RECONSTRUCTION_DIRS = [METASHAPE_OUTPUT_BASE + 'gopro_121/']
                       #]
                       # METASHAPE_OUTPUT_BASE + 'gopro_119_2/',
                       # METASHAPE_OUTPUT_BASE + 'gopro_121/',
                       # METASHAPE_OUTPUT_BASE + 'gopro_125/']

for RECON_DIR in RECONSTRUCTION_DIRS:
    logging.basicConfig(filename=RECON_DIR+'recon.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger('Recon')
    consoleHandler = logging.StreamHandler()
    logger.addHandler(consoleHandler)
    logger.info("Recon start time: {}".format(datetime.now().strftime("%H:%M %d/%m/%Y")))

    # TODO: Metashape logging
    Metashape.app.settings.log_path = RECON_DIR+"metashape.log"
    Metashape.app.Settings(log_enable=True)

    logger.info("Creating new doc or opening existing...")
    doc = Metashape.Document()
    doc.open(RECON_DIR+"recon.psx")
    doc.read_only = False

    for chunk in doc.chunks:

        # logger.info("Building DepthMaps...")
        # start_ts = time.time()
        # chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.AggressiveFiltering)
        # logger.info("Depthmap build time: {} mins".format((time.time() - start_ts) / 60))
        # doc.save()
        #
        # logger.info("Building Dense cloud...")
        # start_ts = time.time()
        # chunk.buildDenseCloud()
        # logger.info("Dense cloud build time: {} mins".format((time.time() - start_ts) / 60))
        # doc.save()

        # logger.info("Building Model...")
        # start_ts = time.time()
        # chunk.buildModel()
        # logger.info("Model build time: {} mins".format((time.time() - start_ts) / 60))
        # doc.save()

        logger.info("Building DEM...")
        start_ts = time.time()
        chunk.buildDem(source_data=Metashape.DenseCloudData, resolution=DEM_RES,
                       interpolation=Metashape.EnabledInterpolation)
        logger.info("DEM build time: {} mins".format((time.time() - start_ts) / 60))
        doc.save()

        logger.info("Building Ortho...")
        start_ts = time.time()
        chunk.buildOrthomosaic(surface_data=Metashape.ElevationData, resolution=ORTHO_RES,
                               blending_mode=Metashape.AverageBlending, fill_holes=True,
                               ghosting_filter=False, cull_faces=False, refine_seamlines=False)
        logger.info("ORTHO build time: {} mins".format((time.time() - start_ts) / 60))
        doc.save()

        logger.info("Exporting Rasters...")
        start_ts = time.time()
        chunk.exportRaster(RECON_DIR + "dem_" + ".tif", source_data=Metashape.ElevationData,
                           image_format=Metashape.ImageFormatTIFF,
                           format=Metashape.RasterFormatTiles, split_in_blocks=True, resolution=DEM_RES,
                           white_background=False)

        chunk.exportRaster(RECON_DIR + "demcol_" + ".tif", source_data=Metashape.ElevationData,
                           image_format=Metashape.ImageFormatTIFF,
                           raster_transform=Metashape.RasterTransformPalette,
                           format=Metashape.RasterFormatTiles, split_in_blocks=True, resolution=DEM_RES,
                           white_background=False)

        chunk.exportRaster(RECON_DIR+"ortho_"+".tif", source_data=Metashape.OrthomosaicData,
                           image_format=Metashape.ImageFormatTIFF,
                           format=Metashape.RasterFormatTiles, split_in_blocks=True, resolution=ORTHO_RES,
                           white_background=False)
        logger.info("Raster export time: {} mins".format((time.time() - start_ts) / 60))

    logging.shutdown()