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
USE_APPROX_REF = True
ADD_GPS_MARKER = False

MAX_PHOTOS_PER_CHUNK = 3000
FILE_MOD = 1

METASHAPE_OUTPUT_BASE = '/local/ScallopReconstructions/' #'/scratch/data/tkr25/Reconstructions/' #
RECONSTRUCTION_DIRS = [METASHAPE_OUTPUT_BASE + 'gopro_124/',
                       ]

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
        logger.info("Recon file already exists, replacing...")
    doc.read_only = False
    doc.save(RECON_DIR + 'recon.psx')

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
            metashape_utils.init_cam_poses_pkl(chunk, pkl_telem, gps_origin, cam_offsets)
        if ADD_GPS_MARKER:
            metashape_utils.add_gps_marker(chunk, gps_origin)

        doc.save()
    logger.info("Photo add time: {} mins".format((time.time() - start_ts) / 60))

    start_ts = time.time()
    for chunk in doc.chunks:
        logger.info("LR camera alignment...")
        chunk.matchPhotos(downscale=2, generic_preselection=True, guided_matching=False,
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
        logger.info("Camera alignment time: {} mins".format((time.time() - start_ts) / 60))

        if USE_APPROX_REF:
            chunk.transform = Metashape.ChunkTransform()
            chunk.transform.translation = chunk.crs.unproject(Metashape.Vector(gps_origin))
            chunk.transform.rotation = Metashape.Matrix(np.eye(3))
            chunk.transform.scale = 1.0
            # metashape_utils.add_gps_approx(chunk, gps_origin)

        logger.info("Building DepthMaps...")
        start_ts = time.time()
        chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.AggressiveFiltering)
        logger.info("Depthmap build time: {} mins".format((time.time() - start_ts) / 60))
        doc.save()

        logger.info("Building Dense cloud...")
        start_ts = time.time()
        chunk.buildDenseCloud()
        logger.info("Dense cloud build time: {} mins".format((time.time() - start_ts) / 60))
        doc.save()

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

    # if len(doc.chunks) > 1:
    #     logger.info("Aligning chunks...")
    #     doc.alignChunks(method=0)
    #     #doc.mergeChunks()

    logger.info("Recon finish time: {}".format(datetime.now().strftime("%H:%M %d/%m/%Y")))