import Metashape
import numpy as np
import pathlib as p
import os
import pickle
from matplotlib import pyplot as plt
import cv2
import pathlib
from tqdm import tqdm
import math

#TODO: constrain sequential images / start/end transformation?
#TODO: Do lowres reconstruction and save camera positions, reload and do high res

USE_INITIAL = False
ROV_DATA = False

MAX_PHOTOS_PER_CHUNK = 1000

# RECONSTRUCTION_DIRS = [P.METASHAPE_OUTPUT_BASE + 'gopro_116_1/',
#                        P.METASHAPE_OUTPUT_BASE + 'gopro_118/',
#                        P.METASHAPE_OUTPUT_BASE + 'gopro_123/',
#                        P.METASHAPE_OUTPUT_BASE + 'gopro_124/',
#                        P.METASHAPE_OUTPUT_BASE + 'gopro_125/']

METASHAPE_OUTPUT_BASE = '/local/ScallopReconstructions/'
RECONSTRUCTION_DIRS = [METASHAPE_OUTPUT_BASE + 'gopro_115_colcal/']

for RECON_DIR in RECONSTRUCTION_DIRS:
    if ROV_DATA:
        [path.unlink() for path in p.Path(RECON_DIR + 'left/').iterdir()]
        print("Converting images to png...")
        folder_root = pathlib.Path(RECON_DIR)
        file_list = [str(item) for item in folder_root.iterdir()]
        file_list.sort()
        left_imgs = [i for i in file_list if 'l0' in i]
        right_imgs = [i for i in file_list if 'r0' in i]
        img_pairs = list(zip(left_imgs, right_imgs))
        for idx, (l_path, r_path) in tqdm(enumerate(img_pairs)):
            img_l_bay = cv2.imread(l_path)
            img_l = cv2.cvtColor(img_l_bay[:, :, 0][:, :, None], cv2.COLOR_BAYER_BG2BGR)
            cv2.imwrite(RECON_DIR + 'left/' + str(idx) + '.png', img_l)
    # def loadTelem(folder_path, telem_buff):
    #     telem_pkl_file = open(folder_path + "viewer_data.pkl", "rb")
    #     while True:
    #         try:
    #             telem_buff.append(pickle.load(telem_pkl_file))
    #         except EOFError:
    #             break
    #         except:
    #             break
    #     telem_pkl_file.close()
    # telem = []
    # loadTelem(TELEM_PATH, telem)
    # imu_data = []
    # depth_data = []
    # sonar_data = []
    # viewer_data = []
    # gps_data = []
    # for item in telem:
    #     if item[0] == b'topic_stereo_camera_calib':
    #         print("Has Calib data!")
    #     if item[0] == b'topic_gps':
    #         gps_data.append(item)
    #     if item[0] == b'topic_viewer_data':
    #         viewer_data.append(item)
    #     if item[0] == b'topic_imu':
    #         imu_data.append(item)
    #     if item[0] == b'topic_depth':
    #         depth_data.append(item)
    #     if item[0] == b'topic_sonar':
    #         sonar_data.append(item)
    # fig, ax = plt.subplots()
    # s_time = depth_data[0][1]['ts']
    # e_time = depth_data[-1][1]['ts']
    # depth = np.array([item[1]['depth'] for item in depth_data])
    # if 'temp' in depth_data[0][1]:
    #     temp = np.array([item[1]['temp'] for item in depth_data])
    #     ax.plot(np.linspace(s_time, e_time, len(temp)), temp, label='Temp (deg C)', color='magenta')
    # if 'sonar' in sonar_data[0][1]:
    #     sonar = np.array([item[1]['sonar'][0] for item in sonar_data]) / 1000

    doc = Metashape.Document()
    #doc.open(P.METASHAPE_OUTPUT_BASE + 'gopro_115_colcal/recon_channels.psx')  # 'test/test.psx')

    dirs = list(p.Path(RECON_DIR).glob('**'))
    sub_folders = [pth for pth in dirs if pth.is_dir() and pth.name.__contains__('in_imgs') and not pth.name[-1] in []]
    img_paths = {folder.name: [str(pth) for pth in p.Path(str(folder) + '/').rglob('*.png')] for folder in sub_folders}
    num_planes = len(img_paths)
    for key, folder in img_paths.items():
        img_paths[key] = sorted(folder, key=lambda pth: int(pth.split('/')[-1][:-4]))
    img_path_tuples = list(zip(*img_paths.values()))  # [:200]
    assert all(all(x.split('/')[-1] == pth_tuple[0].split('/')[-1] for x in pth_tuple) for pth_tuple in img_path_tuples)

    print("Total number of photos: {}".format(len(img_path_tuples)))
    num_chunks = math.ceil(len(img_path_tuples) / MAX_PHOTOS_PER_CHUNK)
    print("Number of chunks: {}".format(num_chunks))

    for chunk_i in range(num_chunks):
        img_paths_chunk = img_path_tuples[chunk_i * MAX_PHOTOS_PER_CHUNK:(chunk_i + 1) * MAX_PHOTOS_PER_CHUNK]
        print("Chunk IDX {}, num photos: {}".format(chunk_i, len(img_paths_chunk)))
        chunk = doc.addChunk()

        print("Adding multisensor frames...")
        filegroups = [num_planes]*len(img_paths_chunk)
        filenames = list(sum(img_paths_chunk, ()))
        chunk.addPhotos(filenames=filenames, filegroups=filegroups, layout=Metashape.MultiplaneLayout)
        sensors = {cam.photo.path.split('/')[-2][-1]: cam.sensor for cam in chunk.cameras[:num_planes]}
        for key, sensor in sensors.items():
            sensor.type = Metashape.Sensor.Type.Frame
            sensor.label = key
            sensor.fixed_calibration = False
            sensor.master = sensors['b']
        assert all(cam.photo.path.split('/')[-2][-1] == cam.sensor.label for cam in chunk.cameras)

        print("Aligning cameras for multisensor...")
        chunk.matchPhotos(downscale=2, reference_preselection_mode=Metashape.ReferencePreselectionSequential)
        chunk.alignCameras()
        chunk.optimizeCameras()

    for chunk in doc.chunks:
        print("Chunk frames: {}".format(chunk.frames))
        print(len(chunk.cameras))
        for i, cam in enumerate(chunk.cameras[:3]):
            s = cam.sensor
            photo_label = cam.photo.path.split('/')[-2][-1]
            if i < 3:
                s.label = photo_label
            print("Cam type: {}".format(cam.type))
            print("Cam photo label: {}".format(photo_label))
            print("Sensor layer idx: {}".format(s.layer_index))
            print("Sensor label: {}".format(s.label))
            print("Sensor master: {}".format(s.master))
            print("Fixed calibration: {}".format(s.fixed_calibration))
            print("Fixed pose: {}".format(s.fixed))
            print("Fixed rot: {}".format(s.fixed_rotation))
            print("Fixed pos: {}".format(s.fixed_location))
            print()

    print("Aligning multisensor chunks...")
    if len(doc.chunks) > 1:
        doc.alignChunks()
        doc.mergeChunks()

    doc.save(RECON_DIR + 'recon_channels.psx')

    doc.open(RECON_DIR + 'recon_channels.psx')

    print("Undistorting multisensor input images...")
    if not os.path.isdir(RECON_DIR + 'out_imgs_u'):
        os.mkdir(RECON_DIR + 'out_imgs_u', 0o777)
    else:
        [pth.unlink() for pth in p.Path(RECON_DIR + 'out_imgs_u').iterdir()]
    undistort_maps = {}
    master_mtx = None
    for idx in range(num_planes):
        sensor = doc.chunks[0].sensors[idx]
        c = sensor.calibration
        lbl = sensor.label
        c.save(RECON_DIR + lbl + '_calibCV.calib')
        cam_mtx = np.array([[c.f + c.b1, c.b2, c.cx + c.width / 2],
                           [0, c.f, c.cy + c.height / 2],
                           [0, 0, 1]])
        if master_mtx is None:
            master_mtx = cam_mtx
        dist = np.array([[c.k1, c.k2, c.p2, c.p1, c.k3]])
        print(lbl)
        print(cam_mtx)
        print(dist)
        print()
        undistort_maps[lbl] = cv2.initUndistortRectifyMap(cam_mtx, dist, None, master_mtx, (2880, 1620), cv2.CV_8UC1)
    for pths in img_path_tuples:
        ud_imgs = {}
        for idx, pth in enumerate(pths):
            pth_ch = pth.split('/')[-2][-1]
            img = cv2.imread(pth)
            img_ud = cv2.remap(img, undistort_maps[pth_ch][0], undistort_maps[pth_ch][1], interpolation=cv2.INTER_LINEAR)
            ud_imgs[pth.split('/')[-2][-1]] = img_ud
        imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in [ud_imgs[k] for k in ['b', 'g', 'r']]]
        col_img = np.dstack(imgs_gray)
        cv2.imwrite(RECON_DIR + 'out_imgs_u/' + pths[0].split('/')[-1], col_img)

    # -------------------- Colour reconstruction ----------------------------

    col_doc = Metashape.Document()
    col_imgs = [str(pth) for pth in p.Path(str(RECON_DIR + '/out_imgs_u/')).rglob('*.png')]

    for chunk_i in range(num_chunks):
        img_paths_chunk = col_imgs[chunk_i * MAX_PHOTOS_PER_CHUNK:(chunk_i + 1) * MAX_PHOTOS_PER_CHUNK]
        print("Chunk IDX {}, num photos: {}".format(chunk_i, len(img_paths_chunk)))
        chunk = col_doc.addChunk()
        print(img_paths_chunk)

        chunk.addPhotos(img_paths_chunk)
        chunk.matchPhotos(downscale=2, reference_preselection_mode=Metashape.ReferencePreselectionSequential)
        chunk.alignCameras()
        chunk.optimizeCameras(fit_k4=True)
        chunk.buildDepthMaps()
        chunk.buildDenseCloud()

    print("Aligning multisensor chunks...")
    if len(col_doc.chunks) > 1:
        col_doc.alignChunks()
        col_doc.mergeChunks()

    col_doc.save(RECON_DIR + 'recon_col.psx')


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