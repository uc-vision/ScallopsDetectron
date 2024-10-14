import os, shutil
import pickle
import pathlib
from deprecated_files import dvl_data_utils
from matplotlib import pyplot as plt
import numpy as np
import zipfile
import glob
import xml.etree.ElementTree as ET


def del_if_exists(path):
    if os.path.exists(path):
        os.remove(path)

def open_archive(dir, mode="r"):
    vpz_files = glob.glob(dir + '*.vpz')
    assert len(vpz_files) == 1
    return zipfile.ZipFile(vpz_files[0], mode)

def open_vpz_r(dir, mode="r"):
    archive = open_archive(dir, mode)
    xml_file = [f for f in archive.namelist() if ".xml" in f]
    assert len(xml_file) == 1
    root = ET.fromstring(archive.read(xml_file[0]))
    return root, archive

def get_xml_layers(xml_root):
    sites = xml_root.find('sites')
    assert len(sites) == 1
    return sites[0].find('layers')

def append_vpz_shapes(vpz_dir, shapes_fp_l, col_rgb_l=None):
    xml_root, archive_r = open_vpz_r(vpz_dir, "r")
    labels = [fp.split('/')[-1].split('.')[0] for fp in shapes_fp_l]
    tmp_archive_fp = vpz_dir+'.tmp.vpz'
    archive_w = zipfile.ZipFile(tmp_archive_fp, 'w')
    layers = get_xml_layers(xml_root)
    # Copy all files which are not xml or gpkg to new archive
    for file in archive_r.filelist:
        if not any(k in file.filename for k in ['.xml', '.gpkg']):
            archive_w.writestr(file.filename, archive_r.read(file.filename))
    # copy over all shape files not being overwritten to new archive,
    # delete xml entries of files which are being overwritten
    sf_idx = 0
    for layer in list(layers):
        if not layer.get('type') == 'shapes':
            continue
        shp_fp = layer.find('data').get('path')
        if layer.get('label') in labels:
            layers.remove(layer)
        else:
            new_fn = 'shapes{}.gpkg'.format(sf_idx)
            archive_w.writestr(new_fn, archive_r.read(shp_fp))
            layer.find('data').set('path', new_fn)
            sf_idx += 1
    # Append updated shapes to vpz archive and xml file
    cols_hex = ['#ffffff'] * len(shapes_fp_l) if col_rgb_l is None else ['#%02x%02x%02x' % c for c in col_rgb_l]
    for label, fp, col in list(zip(labels, shapes_fp_l, cols_hex)):
        new_fn = 'shapes{}.gpkg'.format(sf_idx)
        archive_w.write(fp, new_fn)
        content='''\
                <layer type="shapes" label="{}" enabled="true">
                    <data path="{}"/>
                    <meta>
                        <property name="style/color" value="{}"/>
                    </meta>
                </layer>
                '''.format(label, new_fn, col)
        layers.append(ET.XML(content))
        sf_idx += 1
    archive_w.writestr('doc.xml', ET.tostring(xml_root))
    old_archive_fp = archive_r.fp.name
    archive_w.close()
    archive_r.close()
    shutil.move(tmp_archive_fp, old_archive_fp)

def extract_vpz_shapes(vpz_dir):
    xml_root, archive = open_vpz_r(vpz_dir)
    tmpshps_dir = vpz_dir+'gpkg_files/.tmpshps/'
    os.mkdir(tmpshps_dir)
    archive.extractall(tmpshps_dir, members=[f for f in archive.namelist() if ".gpkg" in f])
    layers = get_xml_layers(xml_root)
    xml_shapes_layers = [layer for layer in layers if layer.get('type') == 'shapes' and layer.find('data') is not None]
    for layer in xml_shapes_layers:
        fn_nospace = '_'.join(layer.get('label').split(' '))
        shutil.move(tmpshps_dir+layer.find('data').get('path'), vpz_dir+'gpkg_files/'+fn_nospace+'.gpkg')
    shutil.rmtree(tmpshps_dir)

def get_vpz_dataset_paths(vpz_dir):
    vpz_files = glob.glob(vpz_dir + '*.vpz')
    assert len(vpz_files) == 1
    archive = zipfile.ZipFile(vpz_files[0])
    zipped_shape_files = [f for f in archive.namelist() if ".gpkg" in f]
    file_paths = []
    for zipped_fn in zipped_shape_files:
        file_paths.append('zip://' + 'Station_3_grid.vpz' + '!' + zipped_fn) # vpz_files[0]
        print(file_paths[-1])
    return file_paths

def get_impath_tuples(dir):
    dirs = list(pathlib.Path(dir).glob('**'))
    sub_folders = [pth for pth in dirs if pth.is_dir() and pth.name.__contains__('imgs')] # and not pth.name[-1] in ['7', '8']]
    sensor_ids = [pth.name.split('/')[-1].split('_')[-1] for pth in sub_folders]
    img_paths = {}
    for folder in sub_folders:
        glob_paths = list(pathlib.Path(str(folder) + '/').rglob('*.png')) + \
                     list(pathlib.Path(str(folder) + '/').rglob('*.jpg'))
        img_paths[folder.name] = [str(pth) for pth in glob_paths]
    for key, folder in img_paths.items():
        img_paths[key] = sorted(folder, key=lambda pth: int(''.join(filter(str.isdigit, pth.split('/')[-1][:-4]))))
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

            dvl_points_txyzXYZ = np.array([[ts, d['x'], d['y'], d['z'], d['roll']+180, d['pitch'], d['yaw']+90] for ts, d in self.dvl_data_dr])
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

# def main():
#     pkl_telem = try_load_pkl("/csse/research/CVlab/bluerov_data/221008-101944/")
#
# if __name__ == '__main__':
#     main()