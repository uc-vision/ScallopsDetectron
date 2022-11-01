import Metashape
print("Metashape version {}".format(Metashape.version))
import numpy as np
import cv2
from tqdm import tqdm
import math
from utils import file_utils, metashape_utils
import time
from datetime import datetime
import logging
import os

#TODO: DVL to camera transform not correct

USE_INITIAL = False
USE_APPROX_REF = False
INIT_CAM_APPROX = True

REPROCESS = True

MAX_PHOTOS_PER_CHUNK = 4000
FILE_MOD = 3

METASHAPE_OUTPUT_BASE = '/local/ScallopReconstructions/' #'/scratch/data/tkr25/Reconstructions/' #
RECONSTRUCTION_DIRS = [METASHAPE_OUTPUT_BASE + 'gopro_128/',]
                       #METASHAPE_OUTPUT_BASE + 'gopro_116_2/']
#                        METASHAPE_OUTPUT_BASE + 'gopro_124/',
#                        METASHAPE_OUTPUT_BASE + 'gopro_125/']

cam_offsets = {'000F315DAB37': -0.310, '000F315DB084': 0.0, '000F315DAB68': 0.310}
gps_origin = np.array([174.53350, -35.845, 40]) #np.array([174.856321, -37.241498, 112.4])
crs = "EPSG::4326"  # "EPSG::2193"

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

    # ROV data processing if pkl file found
    pkl_telem = file_utils.try_load_pkl(RECON_DIR)

    logger.info("Creating new doc or opening existing...")
    doc = Metashape.Document()
    if os.path.isfile(RECON_DIR+"recon.psx"):
        logger.info("Recon file already exists, loading...")
        doc.open(RECON_DIR+"recon.psx")
    if doc.read_only:
        logger.error("Doc is read only")
        exit(0)
    doc.save(RECON_DIR + 'recon.psx')

    if REPROCESS or len(doc.chunks) == 0:
        start_ts = time.time()
        img_path_tuples, sensor_ids = file_utils.get_impath_tuples(RECON_DIR)
        num_sensors = len(sensor_ids)
        img_path_tuples = img_path_tuples[::FILE_MOD]

        logger.info("Total number of photo sets: {}".format(len(img_path_tuples)))
        logger.info("Total number of sensors: {}".format(num_sensors))
        num_chunks = math.ceil(len(img_path_tuples) / MAX_PHOTOS_PER_CHUNK)
        logger.info("Number of chunks: {}".format(num_chunks))

        for chunk_i in range(num_chunks):
            img_paths_chunk = img_path_tuples[chunk_i * MAX_PHOTOS_PER_CHUNK:(chunk_i + 1) * MAX_PHOTOS_PER_CHUNK]
            logger.info("Chunk IDX {}, num photos: {}".format(chunk_i, len(img_paths_chunk)))
            chunk = doc.addChunk()
            chunk.crs = Metashape.CoordinateSystem(crs)
            logger.info(chunk.crs)

            logger.info("Adding multisensor frames...")
            filegroups = [num_sensors]*len(img_paths_chunk)
            filenames = list(sum(img_paths_chunk, ()))
            chunk.addPhotos(filenames=filenames, filegroups=filegroups, layout=Metashape.MultiplaneLayout)
            # Set sensor labels
            for key, cam in zip(sensor_ids, chunk.cameras[:num_sensors]):
                sensor = cam.sensor
                sensor.type = Metashape.Sensor.Type.Frame
                sensor.label = key
                sensor.fixed_rotation = False
                sensor.fixed_location = False
                sensor.fixed_calibration = False
                #sensor.master = sensors['b']

            # Initialise camera positions if ROV data
            if pkl_telem:
                metashape_utils.init_cam_poses_pkl(chunk.cameras, pkl_telem, gps_origin, cam_offsets)
            if USE_APPROX_REF:
                metashape_utils.init_cam_poses_line(chunk.cameras, gps_origin, len(img_path_tuples))
            if INIT_CAM_APPROX:
                metashape_utils.init_cam_pose_approx(chunk.cameras, gps_origin)

            doc.save()
        logger.info("Photo add time: {} mins".format((time.time() - start_ts) / 60))

    if REPROCESS or not any(cam.transform is not None for cam in doc.chunks[0].cameras):
        start_ts = time.time()
        for chunk in doc.chunks:
            logger.info("LR camera alignment...")
            chunk.matchPhotos(downscale=1, generic_preselection=True, guided_matching=False,
                              reference_preselection=False,
                              # reference_preselection_mode=Metashape.ReferencePreselectionSource,
                              keypoint_limit=100e3, tiepoint_limit=20e3, keep_keypoints=False)
            chunk.alignCameras()
            metashape_utils.print_alignment_stats(chunk.cameras, logger=logger)

            # ReferencePreselectionSource, ReferencePreselectionEstimated, ReferencePreselectionSequential
            logger.info("HR camera alignment...")
            chunk.matchPhotos(downscale=1, generic_preselection=False, guided_matching=False,
                              reference_preselection=True,
                              reference_preselection_mode=Metashape.ReferencePreselectionEstimated,
                              keypoint_limit=40e3, tiepoint_limit=4e3, keep_keypoints=False)
            chunk.alignCameras()
            chunk.optimizeCameras()
            metashape_utils.print_alignment_stats(chunk.cameras, logger=logger)
            doc.save()
        logger.info("Camera alignment time: {} mins".format((time.time() - start_ts) / 60))

    # Print Sensor Intrinsics
    for chunk in doc.chunks:
        logger.info("Chunk frames: {}".format(chunk.frames))
        logger.info(len(chunk.cameras))
        for i, cam in enumerate(chunk.cameras[:len(chunk.sensors)]):
            s = cam.sensor
            photo_label = cam.photo.path.split('/')[-2][-1]
            logger.info("Cam type: {}".format(cam.type))
            logger.info("Cam photo label: {}".format(photo_label))
            c = s.calibration
            cam_fov = np.array([math.degrees(2*math.atan(c.width / (2*(c.f + c.b1)))),
                    math.degrees(2*math.atan(c.height / (2*c.f)))])
            logger.info("Sensor FOV: {}".format(cam_fov))
            logger.info("Sensor layer idx: {}".format(s.layer_index))
            logger.info("Sensor label: {}".format(s.label))
            logger.info("Sensor master: {}".format(s.master))
            logger.info("Fixed calibration: {}".format(s.fixed_calibration))
            logger.info("Fixed pose: {}".format(s.fixed))
            logger.info("Fixed rot: {}".format(s.fixed_rotation))
            logger.info("Fixed pos: {}".format(s.fixed_location))

    if len(doc.chunks) > 1:
        logger.info("Aligning chunks...")
        doc.alignChunks(method=0)
        #doc.mergeChunks()

    for chunk in doc.chunks:
        if REPROCESS or chunk.depth_maps is None:
            logger.info("Building DepthMaps...")
            start_ts = time.time()
            chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.AggressiveFiltering)
            logger.info("Depthmap build time: {} mins".format((time.time() - start_ts) / 60))
            doc.save()

        if REPROCESS or chunk.dense_cloud is None:
            logger.info("Building Dense cloud...")
            start_ts = time.time()
            chunk.buildDenseCloud()
            logger.info("Dense cloud build time: {} mins".format((time.time() - start_ts) / 60))
            doc.save()

        if REPROCESS or chunk.elevation is None:
            logger.info("Building DEM...")
            start_ts = time.time()
            chunk.buildDem(source_data=Metashape.DenseCloudData, resolution=0.005)
            logger.info("DEM build time: {} mins".format((time.time() - start_ts) / 60))
            doc.save()

        if REPROCESS or chunk.orthomosaic is None:
            logger.info("Building Ortho...")
            start_ts = time.time()
            chunk.buildOrthomosaic(surface_data=Metashape.ElevationData, resolution=0.0005)
            logger.info("ORTHO build time: {} mins".format((time.time() - start_ts) / 60))
            doc.save()

        logger.info("Exporting Rasters...")
        chunk.exportRaster(RECON_DIR + "dem_" + ".tif", source_data=Metashape.ElevationData,
                           image_format=Metashape.ImageFormatTIFF,
                           format=Metashape.RasterFormatTiles, split_in_blocks=False, resolution=0.001,
                           white_background=False)

        chunk.exportRaster(RECON_DIR+"ortho_"+".tif", source_data=Metashape.OrthomosaicData,
                           image_format=Metashape.ImageFormatTIFF,
                           format=Metashape.RasterFormatTiles, split_in_blocks=False, resolution=0.001,
                           white_background=False)

    logger.info("Recon finish time: {}".format(datetime.now().strftime("%H:%M %d/%m/%Y")))