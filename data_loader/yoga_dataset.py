from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
import json
from PIL import Image
import numpy as np

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

        return item_result

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
        

    