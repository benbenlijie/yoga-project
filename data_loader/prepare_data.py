from pathlib import Path
import pandas as pd
import cv2
import mediapipe as mp
import json
import os

mp_pose = mp.solutions.pose


def save_landmark_result(result, save_file):
    result_dict = {}
    for mark in mp_pose.PoseLandmark:
        mark_dict = {}
        keys = ["x", "y", "z", "visibility"]
        for k in keys:
            mark_dict[k] = eval(f"result.pose_landmarks.landmark[mark].{k}")
        result_dict[mark.value] = mark_dict
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, "w") as f:
        json.dump(result_dict, f)


def replaceSuffix(origin, new_suffix):
    dot_idx = origin.rindex(".")
    return origin[:dot_idx] + new_suffix


def getBodyKeypoints(images, save_folder):
    with mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
        for name, image in images.items():

            if image is None:
                continue
            name = replaceSuffix(name, ".txt")
            name = "/".join(name.split("/")[-2:])
            name = os.path.join(save_folder, name)
            if os.path.exists(name):
                continue

            print('\r', name, end="")
            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            try:
                if len(image.shape) < 3:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(image.shape)
                results = pose.process(image)
            except Exception as e:
                print(e)
                continue
            # Print nose landmark.
            print("results", results.pose_landmarks)

            if not results.pose_landmarks:
                continue
            try:
                save_landmark_result(results, name)
            except:
                pass


def prepareAnnos(dataset_folder):
    dataset_folder = Path(dataset_folder)
    image_folder = dataset_folder / "yoga_images"
    anno_folder = dataset_folder / "yoga_anno"

    # search for all images
    img_suffix = ["*.jpg", "*.jpeg", "*.png"]
    image_list = []
    for suffix in img_suffix:
        image_list.extend(image_folder.glob(f"*/{suffix}"))
    images = {}
    for image_path in image_list:
        img = cv2.imread(str(image_path), 0)
        if img is not None:
            images[str(image_path)] = img
    print("image amount:", len(images))

    getBodyKeypoints(images, str(anno_folder))


def loopFolder(image_folder, annotation_folder, labels=None, anno_suffix=".txt", img_suffix=".png"):
    image_folder = Path(image_folder) if isinstance(image_folder, str) else image_folder
    annotation_folder = Path(annotation_folder) if isinstance(annotation_folder, str) else annotation_folder

    data_info_list = []
    for sub_anno_folder in annotation_folder.iterdir():
        if sub_anno_folder.is_dir():
            folder_name = sub_anno_folder.name
            if labels is not None and folder_name not in labels:
                continue
            print("operating label", folder_name)
            sub_img_folder = image_folder / folder_name
            for anno_file in sub_anno_folder.glob("*" + anno_suffix):
                img_files = list(sub_img_folder.glob(anno_file.stem + img_suffix))
                if len(img_files) != 1:
                    continue
                img_file = img_files[0]
                data_info_list.append({
                    "anno": anno_file,
                    "img": img_file,
                    "label": folder_name
                })
    return data_info_list


if __name__ == '__main__':
    dataset_root = Path("/hpctmp/e0703350_yoga/datas/")
    # prepareAnnos(str(dataset_root / "datasets_2"))

    # read select labels
    select_labels_file = dataset_root / "filtered_labels.txt"
    with open(str(select_labels_file), "r") as f:
        select_labels = [l.strip() for l in f.readlines()]
    print(f"number of filtered classes: {len(select_labels)}")

    image_folder = "yoga_images"
    anno_folder = "yoga_anno"
    merged_data = []
    for sub_folder in dataset_root.iterdir():
        if sub_folder.is_dir() and "dataset" in sub_folder.name:
            data_info_list = loopFolder(sub_folder / image_folder, sub_folder / anno_folder, labels=select_labels)
            merged_data.extend(data_info_list)
            df = pd.DataFrame(data_info_list)
            df.to_csv(str(sub_folder / (sub_folder.name + "_filtered.csv")), index=False)
    df = pd.DataFrame(merged_data)
    df.to_csv(str(dataset_root / "merged_data.csv"), index=False)
