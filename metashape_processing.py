import Metashape
import numpy as np
import pathlib as p
import os
import pickle
from matplotlib import pyplot as plt
import Params as P
import cv2
import pathlib
from tqdm import tqdm

#TODO: constrain sequential images / start/end transformation?

if P.ROV_DATA:
    [path.unlink() for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'left/').iterdir()]
    print("Converting images to png...")
    folder_root = pathlib.Path(P.ROV_DATA_PATH)
    file_list = [str(item) for item in folder_root.iterdir()]
    file_list.sort()
    left_imgs = [i for i in file_list if 'l0' in i]
    right_imgs = [i for i in file_list if 'r0' in i]
    img_pairs = list(zip(left_imgs, right_imgs))
    for idx, (l_path, r_path) in tqdm(enumerate(img_pairs)):
        img_l_bay = cv2.imread(l_path)
        img_l = cv2.cvtColor(img_l_bay[:, :, 0][:, :, None], cv2.COLOR_BAYER_BG2BGR)
        cv2.imwrite(P.METASHAPE_OUTPUT_DIR + 'left/' + str(idx) + '.png', img_l)

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

#doc.open(RECON_FOLDER+RECON_CHKPNT_NAME)
#chunk = doc.chunk

chunk = doc.addChunk()

img_paths = [str(pth) for pth in p.Path(P.METASHAPE_OUTPUT_DIR + 'left/').rglob('*.png')]
img_paths.sort()
print("Number of photos: {}".format(len(img_paths)))

chunk.addPhotos(img_paths)
#chunk.sensors
chunk.matchPhotos()
chunk.alignCameras()
chunk.buildDepthMaps(downscale=4, filter_mode=Metashape.AggressiveFiltering)
chunk.buildDenseCloud()
chunk.buildModel(surface_type=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation)
chunk.buildUV(mapping_mode=Metashape.GenericMapping)
chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=4096)

doc.save(P.METASHAPE_OUTPUT_DIR+'recon.psx')