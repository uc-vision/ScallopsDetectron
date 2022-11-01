import Metashape
import numpy as np
from utils import dvl_data_utils
import cv2
from matplotlib import pyplot as plt

gps2m_scale = np.array([111e3, 111e3, 1])

def print_alignment_stats(cams, logger=None):
    total_cams = len(cams)
    aligned_cams = 0
    loc_covariances = []
    for cam in cams:
        if cam.transform:
            aligned_cams += 1
        if cam.location_covariance:
            loc_covariances.append(np.array(cam.location_covariance).reshape((3, 3)))
    if len(loc_covariances):
        diag_vecs = np.array(loc_covariances)[:, (0, 1, 2), (0, 1, 2)].transpose()
        mean_covs = np.round(np.mean(diag_vecs, axis=1), 3)
    else:
        mean_covs = [0, 0, 0]
    print_f = logger.info if logger else print
    print_f('================================================')
    print_f("Cameras aligned: {} / {}".format(aligned_cams, total_cams))
    print_f('MEAN location covariance x {} y {} z {}'.format(mean_covs[0], mean_covs[1], mean_covs[2]))
    print_f('================================================')
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(diag_vecs[0], diag_vecs[1], diag_vecs[2])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    return aligned_cams / len(cams), np.mean(mean_covs)

def init_cam_pose_approx(cams, origin):
    for cam in cams:
        #cam.reference.rotation = Metashape.Vector([0, 0, 0])
        cam.reference.location = Metashape.Vector(origin)
        #cam.reference.rotation_accuracy = Metashape.Vector([20, 20, 20])
        cam.reference.location_accuracy = Metashape.Vector(np.array([100.0, 100.0, 20.0]))

def init_cam_poses_line(cams, origin, num_frames):
    global_cam_idx = 0
    pos_offset = origin.copy()
    halfway_idx = num_frames // 2
    for cam in cams:
        if global_cam_idx < halfway_idx:
            cam.reference.rotation = Metashape.Vector([0, 0, 0])
            cam.reference.location = Metashape.Vector(pos_offset)
            pos_offset[0] += (200 / num_frames) / gps2m_scale[0]
        else:
            if global_cam_idx == halfway_idx:
                pos_offset[1] += 1 / gps2m_scale[1]
            cam.reference.rotation = Metashape.Vector([0, 0, 180])
            cam.reference.location = Metashape.Vector(pos_offset)
            pos_offset[0] -= (200 / num_frames) / gps2m_scale[0]
        cam.reference.rotation_accuracy = Metashape.Vector([50, 50, 50])
        cam.reference.location_accuracy = Metashape.Vector(np.array([20.0, 20.0, 10.0]) / gps2m_scale)
        global_cam_idx += 1

def init_cam_poses_pkl(cams, pkl_telem, origin, cam_offsets):
    cam_quarts_dvl = []
    pos_offset = origin.copy()
    for cam in cams:
        if not int(cam.label) in pkl_telem.img_timestamps:
            continue
        img_ts = pkl_telem.img_timestamps[int(cam.label)]

        if pkl_telem.has_dvl:
            closest_dvl_d = np.eye(4)
            closest_dvl_ts_err = 1e9
            for ts, d in pkl_telem.dvl_data_dr:
                ts_err = abs(img_ts - ts)
                if ts_err < closest_dvl_ts_err:
                    closest_dvl_d = d
                    closest_dvl_ts_err = ts_err
            cam.reference.rotation = Metashape.Vector([closest_dvl_d['yaw'],
                                                       closest_dvl_d['pitch'],
                                                       closest_dvl_d['roll']])
            cam.reference.rotation_accuracy = Metashape.Vector([5, 5, 5])

            rot = dvl_data_utils.YPRToRot33(closest_dvl_d['yaw'],
                                            closest_dvl_d['pitch'],
                                            closest_dvl_d['roll'])
            cam_offset_cam = np.array([[cam_offsets[cam.sensor.label]], [0.0], [0.0]])
            cam_offset_wrld = np.matmul(rot, cam_offset_cam)
            cam_pos_wrld = np.array([closest_dvl_d['x'], closest_dvl_d['y'], closest_dvl_d['z']], dtype=np.float32) / gps2m_scale \
                           + origin \
                           + cam_offset_wrld[:, 0] / gps2m_scale  # closest_dvl_d['z']
            cam_quarts_dvl.append([cam.sensor.label, np.eye(4, dtype=np.float32)])
            cam_quarts_dvl[-1][1][:3, :3] = rot
            cam_quarts_dvl[-1][1][:3, 3] = cam_pos_wrld + pos_offset
            cam.reference.location = Metashape.Vector(cam_pos_wrld)
            cam.reference.location_accuracy = Metashape.Vector([0.1, 0.1, 0.1])
        else:
            closest_depth = 0.0
            closest_d_ts_err = 1e9
            for ts, d in pkl_telem.depth_data:
                ts_err = abs(img_ts - ts)
                if ts_err < closest_d_ts_err:
                    closest_depth = d
                    closest_d_ts_err = ts_err
            cam.reference.rotation = None # Metashape.Vector([0, 0, 0])
            cam.reference.rotation_accuracy = None # Metashape.Vector([5, 5, 5])

            rot = dvl_data_utils.YPRToRot33(0, 0, 0)
            cam_offset_cam = np.array([[cam_offsets[cam.sensor.label]], [0.0], [0.0]])
            cam_offset_wrld = np.matmul(rot, cam_offset_cam)
            cam_pos_wrld = np.array([0, 0, closest_depth], dtype=np.float32) + cam_offset_wrld[:, 0]
            cam_quarts_dvl.append([cam.sensor.label, np.eye(4, dtype=np.float32)])
            cam_quarts_dvl[-1][1][:3, :3] = rot
            cam_quarts_dvl[-1][1][:3, 3] = cam_pos_wrld + pos_offset
            cam.reference.location = Metashape.Vector(cam_pos_wrld)
            cam.reference.location_accuracy = Metashape.Vector([100.0, 100.0, 0.01])


# if False:
#     import vtk
# CAM_DISPLAY_LIM = 200
# ren = vtk.vtkRenderer()
# renWin = vtk.vtkRenderWindow()
# renWin.AddRenderer(ren)
# renWin.SetSize(1000, 1000)
# iren = vtk.vtkRenderWindowInteractor()
# iren.SetRenderWindow(renWin)
# iren.Initialize()
# vtk_axes_origin = vtk.vtkAxesActor()
# ren.AddActor(vtk_axes_origin)
# for sensor_key, cam_q in cam_quarts_dvl[:CAM_DISPLAY_LIM]:
#     vtk_axes = vtk.vtkAxesActor()
#     vtk_axes.AxisLabelsOff()
#     vtk_axes.SetConeRadius(0.05)
#     vtk_axes.SetTotalLength(0.2, 0.2, 0.2)
#     axes_matrix = vtk.vtkMatrix4x4()
#     axes_matrix.DeepCopy(cam_q.ravel())
#     vtk_axes.SetUserMatrix(axes_matrix)
#     ren.AddActor(vtk_axes)
#     text_actor = vtk.vtkTextActor3D()
#     text_actor.SetInput(sensor_key)
#     text_actor.SetPosition(cam_q[:3, 3])
#     text_actor.SetScale(0.005)
#     ren.AddActor(text_actor)
#
# doc.open(RECON_DIR + 'recon.psx')
# chunk0 = doc.chunks[0]
#
# cam_coords = np.array([np.array(c.transform).reshape((4, 4)) for c in chunk0.cameras if c.transform is not None])
# cam_sensor_labels = [c.sensor.label for c in chunk0.cameras if c.transform is not None]
# chunk_scale = chunk0.transform.scale or 1
# chunk_translation = chunk0.transform.translation or np.array([0, 0, 0])
# chunk_rotation = np.array(chunk0.transform.rotation or np.eye(3))
# chunk_rotation = chunk_rotation.reshape((3, 3))
# chunk_Q = np.eye(4)
# chunk_Q[:3, :3] = chunk_rotation
# chunk_Q[:3, 3] = chunk_translation
# print("Chunk scale: {}\nchunk translation: {}\nchunk rotation: {}\n".format(chunk_scale, chunk_translation, chunk_rotation))
# cam_coords[:, :3, 3] *= chunk_scale
# cam_coords = np.matmul(chunk_Q, cam_coords)
#
# for cam_sensor_key, cam_Q in list(zip(cam_sensor_labels, cam_coords))[:CAM_DISPLAY_LIM]:
#     cam_Q[:3, 3] += 15
#     vtk_axes = vtk.vtkAxesActor()
#     vtk_axes.AxisLabelsOff()
#     vtk_axes.SetConeRadius(0.1)
#     vtk_axes.SetTotalLength(0.3, 0.3, 0.3)
#     axes_matrix = vtk.vtkMatrix4x4()
#     axes_matrix.DeepCopy(cam_Q.ravel())
#     vtk_axes.SetUserMatrix(axes_matrix)
#     ren.AddActor(vtk_axes)
#     text_actor = vtk.vtkTextActor3D()
#     text_actor.SetInput(cam_sensor_key)
#     text_actor.SetPosition(cam_Q[:3, 3])
#     text_actor.SetScale(0.005)
#     ren.AddActor(text_actor)
#
# iren.Start()
# exit(0)
