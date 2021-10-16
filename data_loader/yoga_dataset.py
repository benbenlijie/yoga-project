from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
import json
from PIL import Image
import numpy as np
from torchvision import transforms
from easydict import EasyDict as edict
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles


class YogaDataset(Dataset):
    KEYPOINTS_NUM = 33
    def __init__(self, csv_file, need_img=False, img_size=(220, 144)):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.df["label"])
        self.df["class"] = self.le.transform(self.df["label"])
        self.need_img = need_img
        self.img_size = img_size
        self.class_num = len(self.le.classes_)
        self.transform = transforms.Compose([
            transforms.RandomRotation(10,),
            transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomResizedCrop(size=(img_size[1], img_size[0])),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        item_result = {}
        if self.need_img:
            img = Image.open(row["img"])
            item_result["img"] = img.resize(self.img_size)
        
        #load keypoints
        with open(row["anno"], "r") as f:
            anno = json.load(f)
        keypoints = []
        for point_id in range(self.KEYPOINTS_NUM):
            point = anno[str(point_id)]
            keypoints.append([point["x"], point["y"]])
        item_result["keypoints"] = np.array(keypoints, dtype=np.float)

        # load class
        item_result["class"] = row["class"]

        # one_hot
        one_hot = np.zeros([self.class_num], dtype=np.long)
        one_hot[row["class"]] = 1
        item_result["one_hot"] = one_hot

        # skeleton
        skeleton = self.convertKeypoints2Img(item_result["keypoints"], self.img_size)
        item_result["skeleton"] = self.transform(Image.fromarray(skeleton))
        return item_result
    
    @staticmethod
    def convertKeypoints2Img(keypoints, img_size=(220, 144)):
        landmark_list = edict({
            "landmark": []
        })
        for [x, y] in keypoints:
            mark = edict({
                "x": x,
                "y": y,
                "HasField": lambda a: False
            })
            landmark_list.landmark.append(mark)
        canvas = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        mp_drawing.draw_landmarks(
            canvas, 
            landmark_list,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return canvas


class YogaDatasetTriple(YogaDataset):
    def __init__(self, csv_file, need_img=False, img_size=(220, 144)):
        super().__init__(csv_file, need_img, img_size)
    

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
        

    