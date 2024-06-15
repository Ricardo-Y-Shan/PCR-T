import os

PROJECT_NAME = 'PCR-T'


def project_path():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = cur_dir[:cur_dir.find("{}/".format(PROJECT_NAME)) + len("{}/".format(PROJECT_NAME))]
    # print('当前项目名称：{}\r\n当前项目根路径：{}'.format(PROJECT_NAME, root_path))
    return root_path


ROOT_PATH = project_path()

DATASET_ROOT = "/home/sy2021/Datasets"
SHAPENET_PATH = {
    "image_dir": "/home/sy2021/Datasets/ShapeNetImages/ShapeNetRendering",
    "gt_dir": "/home/sy2021/Datasets/ShapeNetModels",
    "resample_gt_dir": "/home/sy2021/Datasets/ShapeNetPoints",
}

PRETRAINED_WEIGHTS_PATH = {
    "resnet50": os.path.join(ROOT_PATH, "pretrained/resnet50-19c8e357.pth"),
    "psgn": os.path.join(ROOT_PATH, "pretrained/psgn.pth"),
}

PREDICT_DIR = os.path.join(ROOT_PATH, 'prediction')

Pix3D_ROOT = '/home/sy2021/pix3d'
