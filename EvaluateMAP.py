from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, transforms
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch
import os
import pathlib
import json
from tqdm import tqdm
import warnings
from utils import augmentations as A, eval_net

warnings.filterwarnings("ignore")

# img = cv2.imread("../../Pictures/labelledscallopscropped.png")
# img_shape = img.shape
# img_points = np.vstack([np.indices(img_shape[:2])[::-1], np.zeros(img_shape[:2])[None], np.ones(img_shape[:2])[None]]).reshape((4, -1))
#
# key = ''
# while key != ord('q'):
#     Q = np.eye(4)
#     Q[:3, 3] = (np.random.random(size=(3,)) - 0.5) * 100
#     Q[:3, :3] += (np.random.random(size=(3, 3)) - 0.5) / 2
#     print(Q)
#     #Q[0, 1] = 0.2
#     img_pnts_T = np.matmul(Q, img_points).astype(np.int32)
#     pixel_coords = img_pnts_T[:2][::-1]
#     # min_x, min_y, max_x, max_y
#     extents = np.array((np.min(pixel_coords[0]), np.max(pixel_coords[0]), np.min(pixel_coords[1]), np.max(pixel_coords[1])))
#     #pixel_coords -= extents[::2, None]
#     rows = pixel_coords[0].clip(0, img_shape[0]-1)
#     cols = pixel_coords[1].clip(0, img_shape[1]-1)
#     img_T = img[(rows, cols)].reshape(img_shape)
#
#     cv2.imshow("original", img)
#     cv2.imshow("transformed", img_T)
#     key = cv2.waitKey()
# exit(0)


SHOW = True
TEST_IMG_DIR = "/local/ScallopMaskDataset/gopro_119_prop"# "/home/cosc/research/CVlab/bluerov_data/210113-065012/"#"/local/ScallopMaskDataset/train/"#
MODEL_PATH = "/local/ScallopMaskRCNNOutputs/HR+LR LP AUGS/"


cfg = get_cfg()
cfg.NUM_GPUS = 1
cfg.merge_from_file(MODEL_PATH + 'config.yml')
model_paths = [str(path) for path in pathlib.Path(MODEL_PATH).glob('*.pth')]
model_paths.sort()
cfg.MODEL.WEIGHTS = os.path.join(model_paths[-1])

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
cfg.TEST.AUG.ENABLED = False
cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
cfg.TEST.AUG.MAX_SIZE = 4000
cfg.TEST.AUG.FLIP = False
cfg.TEST.PRECISE_BN.ENABLED = False
cfg.TEST.PRECISE_BN.NUM_ITER = 200

model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()
print(cfg)

def getDatasetDict(dataset_dir):
    with open(dataset_dir + "/labels.json", 'r') as fp:
        dataset_dict = json.load(fp)
        for data_entry in dataset_dict:
            data_entry["file_name"] = dataset_dir + '/' + data_entry["file_name"].split('/')[-1]
    return dataset_dict
DatasetCatalog.clear()
DatasetCatalog.register(TEST_IMG_DIR, lambda dataset_dir=TEST_IMG_DIR: getDatasetDict(dataset_dir))
MetadataCatalog.get(TEST_IMG_DIR).set(thing_classes=["scallop"])

CNN_INPUT_SHAPE = (800, 1333)
augs = [A.GeometricTransform()] #transforms.Resize(CNN_INPUT_SHAPE)]
        #transforms.RandomCrop(crop_type="absolute", crop_size=CNN_INPUT_SHAPE)] #A.GeometricTransform()
print(A.GeometricTransform().__dir__())
mapper = A.CustomMapper(cfg, is_train=True, augmentations=augs)
data_loader = build_detection_test_loader(cfg, dataset_name=TEST_IMG_DIR, mapper=mapper)
evaluator = eval_net.mAPEvaluator(dataset_name=TEST_IMG_DIR, display=SHOW)
evaluator.reset()

with torch.no_grad():
    print("Calculating AP@[0.5:0.95]...")
    for inputs in tqdm(data_loader):
        outputs = model(inputs)
        evaluator.process(inputs, outputs)
    results = evaluator.evaluate(display=True)
    #print(results)
