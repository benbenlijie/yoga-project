import pandas as pd
from pathlib import Path
import numpy as np
import json
import os
from prepare_data import getBodyKeypoints
import cv2


root_folder = Path("/hpctmp/e0703350_yoga/datas/score_3_items")
name_map_file = root_folder / "name_map.txt"
anno_root = (root_folder / "../datasets_1/yoga_anno").resolve()

with open(str(name_map_file), "r") as f:
    name_map_dict = json.load(f)


def translateImageName(class_name, new_name):
    if class_name not in name_map_dict:
        raise KeyError(f"{class_name} not in dict")
    name_dict = name_map_dict[class_name]
    if new_name not in name_dict:
        print(new_name, "cannot be found in name_map!")
        return "error"
    raw_name = name_dict[new_name]
    anno_name = Path(raw_name).stem + ".txt"
    return os.path.join(class_name, anno_name)


def decodeAnno(file, keypoint_features=['x', 'y'], key_num=33):
    if not Path(file).exists():
        # anno file not exists? create it
        # get image
        img_file = file.replace("yoga_anno", "yoga_images").replace(".txt", ".png")
        print(img_file)
        images = {}
        img = cv2.imread(str(img_file), 0)
        images[str(img_file)] = img
        print(img.shape)
        getBodyKeypoints(images, str(Path(file).parent))

    with open(file, "r") as f:
        anno = json.load(f)
    keypoints = []
    for point_id in range(key_num):
        point = anno[str(point_id)]
        keypoints.append([point[key] for key in keypoint_features])
    return keypoints


def readScore(csv_file, output_cols):
    print("Operating", csv_file)
    class_name = Path(csv_file).stem.replace("_", " ")
    df = pd.read_csv(csv_file)
    columns = df.columns[2:-1]
    df = df[columns]

    def dfLambda(name, bytes=True, anno_path=False):
        anno_file = translateImageName(class_name, name)
        anno_file = anno_root / anno_file
        if not anno_file.exists():
            return np.nan
        # print(anno_file, anno_file.exists())
        keypoints = decodeAnno(str(anno_file))
        keypoints = np.array(keypoints)
        if bytes:
            return keypoints.tobytes()
        elif anno_path is False:
            return keypoints
        else:
            return anno_file

    for i, col in enumerate(columns):
        if "image" in col.lower():
            df[output_cols[i]] = df[col].apply(func=dfLambda)
            df[output_cols[i] + "_mark"] = df[col].apply(func=dfLambda, bytes=False)
            df[output_cols[i] + "_anno"] = df[col].apply(func=dfLambda, bytes=False, anno_path=True)
        else:
            df[output_cols[i]] = df[col].replace(["High", "Medium", "Low"], [0, 1, 2])
    # print(df.head())
    df.dropna(inplace=True)

    # df.to_csv(f"{class_name}.csv")
    return df


if __name__ == '__main__':
    columns = []
    for i in range(3):
        columns.extend([f"keys_{i}", f"score_{i}"])
    output_df = pd.DataFrame(columns=columns)
    df_list = []
    for csv_file in root_folder.glob("*.csv"):
        result_df = readScore(csv_file=csv_file, output_cols=columns)
        df_list.append(result_df)

    output_df = pd.concat(df_list)
    output_df[columns].to_csv("score_data.csv", index=False)
    output_df.to_csv("score_data_raw.csv", index=False)
