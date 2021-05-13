DATASET_DIR = "/local/ScallopMaskDataset/"
MODEL_PATH = "model_final.pth"
CNN_INPUT_SHAPE = (1080, 1920) #(3840, 2160)

METASHAPE_CHKPNT_PATH = '/home/cosc/research/CVlab/GoPro Ortho TEMP/checkpoint.psx'
METASHAPE_OUTPUT_DIR = "/local/ScallopMaskDataset/Metashape_output/"

INFERENCE_OUTPUT_DIR = "/local/ScallopInferenceOutput/Test/"
POLY_ANN_LIST_FN = "PolyAnnList.csv"
TILE_SIZE = 5000
PIXEL_SCALE = 0.001 # meters per pixel
ORTHO_SCALE = 0.2674 # actual size / ortho scale #TODO: double check scale measurement