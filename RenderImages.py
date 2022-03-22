import Metashape
import numpy as np
import cv2


METASHAPE_OUTPUT_BASE = '/local/ScallopReconstructions/'
METASHAPE_OUTPUT_DIR = METASHAPE_OUTPUT_BASE + 'gopro_124/'

doc = Metashape.Document()
doc.open(METASHAPE_OUTPUT_DIR + 'recon.psx')

chunk = doc.chunks[1]
cam = None
cam_idx = 730
while cam_idx < 1000:
    cam = chunk.cameras[cam_idx]
    if cam.transform is not None:
        break
    cam_idx += 1

key = ''
cam_q = cam.transform
while key != ord('q'):
    img_render = chunk.model.renderImage(cam_q, cam.sensor.calibration, add_alpha=False)
    depth_render = chunk.model.renderDepth(cam_q, cam.sensor.calibration, add_alpha=False)
    # img_render.save("test_render.jpg")
    # depth_render.save("depth_render.jpg")

    img_np = np.frombuffer(img_render.tostring(), dtype=np.uint8).reshape((int(img_render.height), int(img_render.width), -1))[:, :, ::-1]
    depth_np = np.frombuffer(depth_render.tostring(), dtype=np.float32).reshape((int(depth_render.height), int(depth_render.width), -1))[:, :, ::-1]

    cv2.namedWindow("IMG", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("DEPTH", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("IMG", img_np)
    depth_norm = depth_np - np.min(depth_np[np.where(depth_np > 0)])
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