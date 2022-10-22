import Metashape
print("Metashape version {}".format(Metashape.version))
import numpy as np
import cv2
from tqdm import tqdm
import math
from utils import file_utils, metashape_utils
from scipy.spatial.transform import Rotation as R
import time

#TODO: DVL to camera transform not correct

USE_INITIAL = False
USE_APPROX_REF = False

MAX_PHOTOS_PER_CHUNK = 2000
FILE_MOD = 2

METASHAPE_OUTPUT_BASE = '/local/ScallopReconstructions/'
RECONSTRUCTION_DIRS = [METASHAPE_OUTPUT_BASE + 'gopro_116_2/'] #'/local/ScallopReconstructions/221008-122856/'] # METASHAPE_OUTPUT_BASE + 'gopro_116_1/',
#                        METASHAPE_OUTPUT_BASE + 'gopro_118/',
#                        METASHAPE_OUTPUT_BASE + 'gopro_123/',
#                        METASHAPE_OUTPUT_BASE + 'gopro_124/',
#                        METASHAPE_OUTPUT_BASE + 'gopro_125/']

cam_offsets = {'000F315DAB37': -0.310, '000F315DB084': 0.0, '000F315DAB68': 0.310}
gps_origin = np.array([174.53350, -35.845, 40]) #np.array([174.856321, -37.241498, 112.4])
crs = "EPSG::4326"  # "EPSG::2193"

for RECON_DIR in RECONSTRUCTION_DIRS:
    # ROV data processing if pkl file found
    pkl_telem = file_utils.try_load_pkl(RECON_DIR)

    print("Creating new doc")
    doc = Metashape.Document()

    img_path_tuples, sensor_ids = file_utils.get_impath_tuples(RECON_DIR)
    num_sensors = len(sensor_ids)
    img_path_tuples = img_path_tuples[::FILE_MOD]

    print("Total number of photo sets: {}".format(len(img_path_tuples)))
    print("Total number of sensors: {}".format(num_sensors))
    num_chunks = math.ceil(len(img_path_tuples) / MAX_PHOTOS_PER_CHUNK)
    print("Number of chunks: {}".format(num_chunks))

    recon_start_ts = time.time()
    for chunk_i in range(num_chunks):
        img_paths_chunk = img_path_tuples[chunk_i * MAX_PHOTOS_PER_CHUNK:(chunk_i + 1) * MAX_PHOTOS_PER_CHUNK]
        print("Chunk IDX {}, num photos: {}".format(chunk_i, len(img_paths_chunk)))
        chunk = doc.addChunk()
        chunk.crs = Metashape.CoordinateSystem(crs)
        print(chunk.crs)

        print("Adding multisensor frames...")
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

        doc.save(RECON_DIR + 'recon.psx')

        print("LR camera alignment...")
        chunk.matchPhotos(downscale=1, generic_preselection=True, guided_matching=False,
                          reference_preselection=False,
                          # reference_preselection_mode=Metashape.ReferencePreselectionSource,
                          keypoint_limit=100e3, tiepoint_limit=20e3, keep_keypoints=False)
        chunk.alignCameras()
        doc.save(RECON_DIR + 'recon_lr.psx')
        # exit(0)

        # ReferencePreselectionSource, ReferencePreselectionEstimated, ReferencePreselectionSequential
        print("HR camera alignment...")
        chunk.matchPhotos(downscale=1, generic_preselection=False, guided_matching=False,
                          reference_preselection=True,
                          reference_preselection_mode=Metashape.ReferencePreselectionEstimated,
                          keypoint_limit=40e3, tiepoint_limit=4e3, keep_keypoints=True)
        chunk.alignCameras()

        chunk.optimizeCameras()
        doc.save(RECON_DIR + 'recon.psx')

    print("Reconstruction time: {}s".format(time.time() - recon_start_ts))

    for chunk in doc.chunks:
        print("Chunk frames: {}".format(chunk.frames))
        print(len(chunk.cameras))
        for i, cam in enumerate(chunk.cameras[:num_sensors]):
            s = cam.sensor
            photo_label = cam.photo.path.split('/')[-2][-1]
            print()
            print("Cam type: {}".format(cam.type))
            print("Cam photo label: {}".format(photo_label))
            c = s.calibration
            cam_fov = np.array([math.degrees(2*math.atan(c.width / (2*(c.f + c.b1)))),
                    math.degrees(2*math.atan(c.height / (2*c.f)))])
            print("Sensor FOV: {}".format(cam_fov))
            print("Sensor layer idx: {}".format(s.layer_index))
            print("Sensor label: {}".format(s.label))
            print("Sensor master: {}".format(s.master))
            print("Fixed calibration: {}".format(s.fixed_calibration))
            print("Fixed pose: {}".format(s.fixed))
            print("Fixed rot: {}".format(s.fixed_rotation))
            print("Fixed pos: {}".format(s.fixed_location))

    # print("Aligning multisensor chunks...")
    # if len(doc.chunks) > 1:
    #     doc.alignChunks(method=0)
    #     doc.mergeChunks()

    doc.save(RECON_DIR + 'recon.psx')

    #     chunk.buildDepthMaps(downscale=4, filter_mode=Metashape.AggressiveFiltering)
    #     chunk.buildDenseCloud()
    #
    #     doc.save(RECON_DIR + 'recon_channels.psx')
    #
    #     chunk.buildModel(surface_type=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation)
    #     chunk.buildUV(mapping_mode=Metashape.GenericMapping)
    #     chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=4096)
    #
    #     cameras = chunk.cameras
    #     cam_coords = np.array([np.array(cam.transform).reshape((4, 4)) for cam in cameras if cam.transform != None])
    #     mean_cam_frame = cam_coords.mean(axis=0)
    #     covariance_m = np.cov(cam_coords[:, :3, 3].T)
    #     eig_vals, eig_vecs = np.linalg.eig(covariance_m)
    #     print("Camera origin points Eigen vectors: {}".format(eig_vecs))
    #     chunk.transform.translation = mean_cam_frame[:3, 3]
    #     avg_cam_z = -mean_cam_frame[:3, 2]
    #     avg_x = np.cross(eig_vecs[0], avg_cam_z)
    #     avg_y = np.cross(avg_cam_z, avg_x)
    #     rot = np.eye(3)
    #     rot[:3, 0] = avg_x
    #     rot[:3, 1] = avg_y
    #     rot[:3, 2] = avg_cam_z
    #     chunk.transform.rotation = rot
    #
    #     doc.save(RECON_DIR + 'recon_channels.psx')
    #
    # if len(doc.chunks) > 1:
    #     doc.alignChunks()
    #     doc.mergeChunks()
    #
    # doc.save(RECON_DIR + 'recon_channels.psx')