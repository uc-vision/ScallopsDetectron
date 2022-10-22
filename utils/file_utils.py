import os
import pickle
import pathlib
from utils import dvl_data_utils
from matplotlib import pyplot as plt
import numpy as np

def get_impath_tuples(dir):
    dirs = list(pathlib.Path(dir).glob('**'))
    sub_folders = [pth for pth in dirs if pth.is_dir() and pth.name.__contains__('imgs')] # and not pth.name[-1] in ['7', '8']]
    sensor_ids = [pth.name.split('/')[-1].split('_')[-1] for pth in sub_folders]
    img_paths = {folder.name: [str(pth) for pth in pathlib.Path(str(folder) + '/').rglob('*.png')] for folder in sub_folders}
    for key, folder in img_paths.items():
        img_paths[key] = sorted(folder, key=lambda pth: int(pth.split('/')[-1][:-4]))
    img_path_tuples = list(zip(*img_paths.values()))
    assert all(all(x.split('/')[-1] == pth_tuple[0].split('/')[-1] for x in pth_tuple) for pth_tuple in img_path_tuples)
    return img_path_tuples, sensor_ids

def has_pickle(dir):
    return any(fn.endswith('.pkl') for fn in os.listdir(dir))

def try_load_pkl(dir):
    return PickleTelemetry(dir) if has_pickle(dir) else None

class PickleTelemetry():
    def __init__(self, RECON_DIR):
        self.telem_pkl_file = open(RECON_DIR + "viewer_data.pkl", "rb")
        self.telem_dict = {}
        while True:
            try:
                item = pickle.load(self.telem_pkl_file)
            except EOFError:
                break
            if not item[0] in self.telem_dict:
                self.telem_dict[item[0]] = []
            self.telem_dict[item[0]].append(item[1])
        print("Telem topics: "+str(list(self.telem_dict.keys())))

        #print(telem_dict[b'topic_camera_telem'])
        #exit(0)

        depth_data = self.telem_dict[b'topic_sonar'] # b'topic_depth']
        self.depth_data = [(d['ts'], d['sonar'][0]) for d in depth_data] #depth

        img_keys = [ts_t[0] for ts_t in self.telem_dict[b'topic_stereo_camera_ts']]
        img_ts = [ts_t[1] for ts_t in self.telem_dict[b'topic_stereo_camera_ts']]
        self.img_timestamps = dict(zip(img_keys, img_ts)) #{d['frame_cnt'][1]: d['ts'] + zero_ts_diff for d in telem_dict[b'topic_viewer_data']}

        if b'topic_dvl_raw' in self.telem_dict:
            self.has_dvl = True
            dvl_data_proc = [(raw_data['ts'], dvl_data_utils.parse_line(raw_data['dvl_raw'])) for raw_data in self.telem_dict[b'topic_dvl_raw']]
            self.dvl_data_dr = [(ts, d) for ts, d in dvl_data_proc if d and d['type'] == 'deadreacon']

            dvl_points_txyzXYZ = np.array([[ts, d['x'], d['y'], d['z'], d['roll']+180, d['pitch'], d['yaw']+90] for ts, d in dvl_data_dr])
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot3D(dvl_points_txyzXYZ[:, 1], dvl_points_txyzXYZ[:, 2], dvl_points_txyzXYZ[:, 3], 'gray')
            ax.scatter3D(dvl_points_txyzXYZ[0, 1], dvl_points_txyzXYZ[0, 2], dvl_points_txyzXYZ[0, 3], c='red')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            fig2 = plt.figure()
            ax2 = plt.axes()
            ax2.plot(dvl_points_txyzXYZ[:, 4], label='roll')
            ax2.plot(dvl_points_txyzXYZ[:, 5], label='pitch')
            ax2.plot(dvl_points_txyzXYZ[:, 6], label='yaw')
            ax2.legend()
            plt.show()