from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn import preprocessing
import json
from PIL import Image
import numpy as np
from torchvision import transforms
from easydict import EasyDict as edict
import mediapipe as mp
from pathlib import Path
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import scipy.sparse as sp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

KEYPOINTS_COLOR_PANEL = [
    (128,0,0),
    (139,0,0),
    (165,42,42),
    (178,34,34),
    (220,20,60),
    (255,0,0),
    (255,99,71),
    (255,127,80),
    (205,92,92),
    (240,128,128),
    (233,150,122),
    (250,128,114),
    (255,160,122),
    (255,69,0),
    (255,140,0),
    (255,165,0),
    (255,215,0),
    (184,134,11),
    (218,165,32),
    (238,232,170),
    (189,183,107),
    (240,230,140),
    (128,128,0),
    (255,255,0),
    (154,205,50),
    (85,107,47),
    (107,142,35),
    (124,252,0),
    (127,255,0),
    (173,255,47),
    (0,100,0),
    (0,128,0),
    (34,139,34),
]


def addGaussian(keypoints, scale=0.02):
    return (keypoints + np.random.normal(loc=0.0, scale=scale, size=keypoints.shape)).astype(np.float32)


def loadKeypoints(anno_file, keypoint_features):
    with open(anno_file, "r") as f:
        anno = json.load(f)
    keypoints = []
    for point_id in range(YogaDataset.KEYPOINTS_NUM):
        point = anno[str(point_id)]
        keypoints.append([point[key] for key in keypoint_features])
    return np.transpose(np.array(keypoints, dtype=np.float32), [1, 0])


class YogaDataset(Dataset):
    KEYPOINTS_NUM = 33
    _THICKNESS_POSE_LANDMARKS = 2

    def __init__(self, csv_file, need_img=False, img_size=(220, 144), color_landmark=False,
                 keypoint_features=["x", "y"], gaussian_scale=0.02, only_points=False):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.df["label"])
        self.df["class"] = self.le.transform(self.df["label"])
        self.need_img = need_img
        self.img_size = img_size
        self.class_num = len(self.le.classes_)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(degrees=30, translate=(0.2, 0.2)),
            transforms.RandomResizedCrop(size=(img_size[1], img_size[0]), scale=(0.8, 1.0)),
            transforms.ToTensor()
        ])
        self.color_landmark = color_landmark
        self.cache = dict()
        self.keypoint_features = keypoint_features
        self.gaussion_scale = gaussian_scale
        self.only_points = only_points

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        cached_item = self.cache.get(index, None)
        if cached_item is not None:
            item_result = cached_item.copy()
            item_result["keypoints"] = addGaussian(cached_item["keypoints"], self.gaussion_scale)
            item_result["keypoints"] = self.randomHorizontalFlip(item_result["keypoints"])
            if not self.only_points:
                skeleton = self.convertKeypoints2Img(item_result["keypoints"], self.img_size, 0)
                item_result["skeleton"] = self.transform(Image.fromarray(skeleton))

            return item_result

        row = self.df.iloc[index]
        item_result = {}
        if self.need_img:
            img = Image.open(row["img"])
            item_result["img"] = img.resize(self.img_size)

        # # load keypoints
        # with open(row["anno"], "r") as f:
        #     anno = json.load(f)
        # keypoints = []
        # for point_id in range(self.KEYPOINTS_NUM):
        #     point = anno[str(point_id)]
        #     keypoints.append([point[key] for key in self.keypoint_features])
        # item_result["keypoints"] = np.transpose(np.array(keypoints, dtype=np.float32), [1, 0])
        item_result["keypoints"] = loadKeypoints(row["anno"], self.keypoint_features)

        # load class
        item_result["class"] = row["class"]

        # one_hot
        one_hot = np.zeros([self.class_num], dtype=np.long)
        one_hot[row["class"]] = 1
        item_result["one_hot"] = one_hot

        if not self.only_points:
            # skeleton
            skeleton = self.convertKeypoints2Img(item_result["keypoints"], self.img_size)
            item_result["skeleton"] = skeleton
            self.cache[index] = item_result.copy()
            item_result["skeleton"] = self.transform(Image.fromarray(item_result["skeleton"]))
        item_result["keypoints"] = addGaussian(item_result["keypoints"], self.gaussion_scale)
        return item_result

    def convertKeypoints2Img(self, keypoints, img_size=(220, 144), sigma=0.05):
        landmark_list = edict({
            "landmark": []
        })
        for [x, y] in np.transpose(keypoints, [1, 0]):
            mark = edict({
                "x": x + np.random.normal(scale=sigma),
                "y": y + np.random.normal(scale=sigma),
                "HasField": lambda a: False
            })
            landmark_list.landmark.append(mark)
        canvas = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        drawing_spec = YogaDataset.get_default_pose_landmarks_style() \
            if self.color_landmark else mp_drawing_styles.get_default_pose_landmarks_style()
        mp_drawing.draw_landmarks(
            canvas,
            landmark_list,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec
        )
        return canvas

    @staticmethod
    def randomHorizontalFlip(keypoints, rate=0.5):
        if np.random.rand() < rate:
            keypoints[0, :] = 1 - keypoints[0, :]
            return keypoints
        else:
            return keypoints

    @staticmethod
    def get_default_pose_landmarks_style():
        """Returns the default pose landmarks drawing style.

        Returns:
            A mapping from each pose landmark to its default drawing spec.
        """
        pose_landmark_style = {}
        for landmark in mp_pose.PoseLandmark:
            pose_landmark_style[landmark] = DrawingSpec(
                color=KEYPOINTS_COLOR_PANEL[landmark.value], thickness=YogaDataset._THICKNESS_POSE_LANDMARKS)
        return pose_landmark_style


class YogaDatasetTriple(YogaDataset):
    def __init__(self, csv_file, need_img=False, img_size=(220, 144), **kwargs):
        super().__init__(csv_file, need_img, img_size, **kwargs)

    def __getitem__(self, index):
        base_item = super().__getitem__(index)
        base_class = base_item["class"]

        exclude_index = ~self.df.index.isin([index])

        # get another item with same class
        same_class_index = self.df[exclude_index & (self.df["class"] == base_class)].sample(1).index.values[0]
        same_class_item = super().__getitem__(same_class_index)

        # get another item with different class
        diff_class_index = self.df[exclude_index & (self.df["class"] != base_class)].sample(1).index.values[0]
        diff_class_item = super().__getitem__(diff_class_index)

        result = {
            "base": base_item,
            "same": same_class_item,
            "diff": diff_class_item,
        }
        return result


class Score3ItemDataset(Dataset):
    COL_PATTERNS = ["keys_{}", "score_{}"]
    RAW_COL_PATTERNS = "keys_{}_anno"
    WEIGHT_COL = "weight"
    NUM_ITEMS = 3

    def __init__(self, score_csv, raw_anno=False, keypoint_features=["x", "y"], gaussian_scale=0.02, **kwargs):
        super().__init__()
        self.file = score_csv
        self.gaussian_scale = gaussian_scale
        self.df = pd.read_csv(score_csv)
        self.raw_anno = raw_anno
        self.keypoint_features = keypoint_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        result = {}
        for i in range(self.NUM_ITEMS):
            if self.raw_anno is False:
                keys_col_name = self.COL_PATTERNS[0].format(i)
                keys = np.frombuffer(eval(row[keys_col_name]), dtype=np.float32).reshape((-1, 2))
                keys = np.transpose(keys, (1, 0))
            else:
                keys_col_name = self.RAW_COL_PATTERNS.format(i)
                keys = loadKeypoints(row[keys_col_name], self.keypoint_features)
                
            score_col_name = self.COL_PATTERNS[1].format(i)
            score = row[score_col_name]
            result[score] = torch.from_numpy(addGaussian(np.array(keys), scale=self.gaussian_scale))
        return result




