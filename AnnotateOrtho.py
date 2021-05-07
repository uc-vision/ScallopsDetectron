import numpy as np
import pathlib as p
import cv2
from tqdm import tqdm
import Params as P

ORTHO_DIR_PATH = P.METASHAPE_OUTPUT_DIR + 'ortho/'
ORTHOANN_DIR_PATH = P.METASHAPE_OUTPUT_DIR + 'ortho_ann/'
#[path.unlink() for path in p.Path(ORTHOANN_DIR_PATH).iterdir()]
ortho_paths = [[str(path), path.name] for path in p.Path(ORTHO_DIR_PATH).iterdir()]

key = ''
mouse_held = False
circle_size = 100
def draw_ann(event, x, y, flags, param):
    global mouse_held, display, ann_img, circle_size
    if event==cv2.EVENT_RBUTTONDOWN:
        mouse_held = True
    elif event==cv2.EVENT_RBUTTONUP:
        mouse_held = False
    if mouse_held:
        cv2.circle(ann_img, (x, y), circle_size, (0, 0, 255), -1)
    elif event==cv2.EVENT_MOUSEMOVE:
        display[:, :, :] = 0
        cv2.circle(display, (x, y), circle_size, (0, 0, 255), -1)


cv2.namedWindow("ortho_ann", cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback("ortho_ann", draw_ann)
for path, fn in tqdm(ortho_paths):
    ortho_img = cv2.imread(path)
    IMG_SIZE = ortho_img.shape
    ann_img = np.zeros_like(ortho_img)
    display = np.zeros_like(ortho_img)

    while True:
        cv2.imshow("ortho_ann", (ortho_img + ann_img/2 + display/2)/255)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)
        elif key == ord('s'):
            print(ORTHOANN_DIR_PATH+"ortho_ann_"+fn[6:])
            cv2.imwrite(ORTHOANN_DIR_PATH+"ortho_ann_"+fn[6:], cv2.cvtColor(ann_img, cv2.COLOR_BGR2GRAY))
            break

cv2.destroyAllWindows()