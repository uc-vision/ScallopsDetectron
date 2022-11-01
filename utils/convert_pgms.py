import cv2
import pathlib
import os
import glob
from tqdm import tqdm
import shutil

SRC_DIR = '/csse/research/CVlab/bluerov_data/221008-122856/'

DEST_DIR = '/local/ScallopReconstructions/' + SRC_DIR.split('/')[-2] + '/'

if not pathlib.Path(DEST_DIR).exists():
    os.mkdir(DEST_DIR)

pgm_image_paths = glob.glob(SRC_DIR+'*.pgm')
pkl_path = glob.glob(SRC_DIR+'*.pkl')
assert len(pkl_path) == 1
pkl_path = pkl_path[0]
shutil.copyfile(pkl_path, DEST_DIR+pkl_path.split('/')[-1])

image_paths_cams = {}
for pth in pgm_image_paths:
    cam_id = pth.split('_')[-1][:-4]
    if not cam_id in image_paths_cams:
        image_paths_cams[cam_id] = []
    image_paths_cams[cam_id].append(pth)

cam_keys = list(image_paths_cams.keys())
for key in cam_keys:
    print("Converting cam "+key)
    assert len(image_paths_cams[key]) == len(image_paths_cams[cam_keys[0]])
    image_paths_cams[key].sort()
    CAM_RGB_DIR = DEST_DIR+'imgs_'+key
    if not pathlib.Path(CAM_RGB_DIR).exists():
        os.mkdir(CAM_RGB_DIR)
    else:
        [path.unlink() for path in pathlib.Path(CAM_RGB_DIR).iterdir()]
    for pgm_path in tqdm(image_paths_cams[key]):
        img_bayer = cv2.imread(pgm_path, 0)
        img_rgb = cv2.cvtColor(img_bayer, cv2.COLOR_BAYER_BG2BGR)
        write_fn = pgm_path.split('/')[-1].split('_')[0]+'.png'
        img_rgb_rs = cv2.resize(img_rgb, (img_rgb.shape[1]//2, img_rgb.shape[0]//2))
        cv2.imwrite(CAM_RGB_DIR + '/' + write_fn, img_rgb_rs)
