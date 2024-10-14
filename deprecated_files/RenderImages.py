import Metashape
import numpy as np
import cv2
import math


RECORD = True
REC_FRAMES = 50

CAM_SHAPE = (2464, 2056) # W, H

METASHAPE_OUTPUT_DIR = '/local/ScallopReconstructions/220606-152021/'

doc = Metashape.Document()
doc.open(METASHAPE_OUTPUT_DIR + 'recon.psx')

chunk = doc.chunks[0]
chunk_scale = chunk.transform.scale if chunk.transform.scale else 1.0
cam_name = '2382'
photo_names = [c.photo.path.split('/')[-1][:-4] for c in chunk.cameras]
cam_idx = photo_names.index(cam_name)
cam = chunk.cameras[cam_idx]
assert cam.transform is not None

cam_q = cam.transform + Metashape.Matrix(np.array([[0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, -0.5],
                                                  [0, 0, 0, 0]]))
print(cam_q)
cam_0_Q = np.eye(4)
cam_1_Q = cam_0_Q + np.array([[0, 0, 0, 0.9],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]])
cam_2_Q = cam_0_Q + np.array([[0, 0, 0, 1.8],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]])
#ref_cam_Q[2, 3] = 2.0
multicam_positions = [
    Metashape.Matrix(cam_0_Q),
    Metashape.Matrix(cam_1_Q),
    Metashape.Matrix(cam_2_Q),
]
c = cam.sensor.calibration
# c.k1 = 0
# c.k2 = 0
# c.k3 = 0
# c.k4 = 0
# c.p1 = 0
# c.p2 = 0
# c.height = CAM_SHAPE[1]
# c.width = CAM_SHAPE[0]
# c.f = 2300
# c.b2 = 0
# c.b1 = 0
camMtx = np.array([[c.f + c.b1,    c.b2,   c.cx + c.width / 2],
                   [0,             c.f,    c.cy + c.height / 2],
                   [0,             0,      1]
])
cam_fov = np.array([math.degrees(2*math.atan(c.width / (2*(c.f + c.b1)))),
                    math.degrees(2*math.atan(c.height / (2*c.f)))])
print(cam_fov)
np.save("../data/CamMtx.npy", camMtx)
np.save("../data/Extrinsics.npy", np.stack([cam_0_Q, cam_1_Q, cam_2_Q]))
exit(0)

if RECORD:
    col_writers = []
    dep_writers = []
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    for i in range(3):
        col_writers.append(cv2.VideoWriter('data/colFrameOverlap_' + str(i) + '.avi', fourcc, 10, CAM_SHAPE))
        dep_writers.append(cv2.VideoWriter('data/depFrameOverlap_' + str(i) + '.avi', fourcc, 10, CAM_SHAPE))

frame_cnt = 0
key = ''
while key != ord('q'):
    col_imgs = []
    dep_imgs = []
    for cam_rel_trans in multicam_positions:
        cam_t = cam_q * cam_rel_trans
        img_render = chunk.model.renderImage(cam_t, c, add_alpha=False)
        depth_render = chunk.model.renderDepth(cam_t, c, add_alpha=False)
        # img_render.save("test_render.jpg")
        # depth_render.save("depth_render.jpg")
        img_np = np.frombuffer(img_render.tostring(), dtype=np.uint8).reshape((int(img_render.height), int(img_render.width), -1))[:, :, :3][:, :, ::-1]
        col_imgs.append(img_np)
        depth_np = np.frombuffer(depth_render.tostring(), dtype=np.float32).reshape((int(depth_render.height), int(depth_render.width), -1))[:, :, ::-1]
        dep_imgs.append(depth_np)
        avg_depth = np.mean(depth_np[np.where((depth_np > -10) * (depth_np < 10))])

    col_img = np.hstack(col_imgs)
    depth_img = np.hstack(dep_imgs)

    if RECORD:
        for i in range(3):
            col_writers[i].write(col_imgs[i])
            depth_img = dep_imgs[i]
            depth_u8 = np.clip(0.3 * 255 * depth_img * (depth_img > -10) * (depth_img < 10), 0, 255).astype(np.uint8)
            depth_u8c3 = np.repeat(depth_u8, 3, axis=2)
            dep_writers[i].write(depth_u8c3)
            cam_q[1, 3] += 0.03 + np.random.normal(scale=0.01)
            cam_q[0, 3] += np.random.normal(scale=0.01)
            cam_q[2, 3] += np.random.normal(scale=0.01)
    else:
        cv2.namedWindow("IMG", cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow("DEPTH", cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("IMG", col_img)
        depth_norm = depth_img - np.min(depth_img[np.where(depth_img > 0)])
        depth_norm /= max(1e-6, np.max(depth_norm))
        cv2.imshow("DEPTH", depth_norm)
        key = cv2.waitKey()

        DISTANCE_INC = 0.5
        if key == ord('w'):
            cam_q[1, 3] += DISTANCE_INC
        elif key == ord('s'):
            cam_q[1, 3] -= DISTANCE_INC
        elif key == ord('a'):
            cam_q[0, 3] -= DISTANCE_INC
        elif key == ord('d'):
            cam_q[0, 3] += DISTANCE_INC

    frame_cnt += 1
    print(frame_cnt, end='\r')
    if RECORD and frame_cnt >= REC_FRAMES:
        break

if RECORD:
    for i in range(3):
        col_writers[i].release()
        dep_writers[i].release()
