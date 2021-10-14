from pathlib import Path
import pandas as pd

def loopFolder(image_folder, annotation_folder, anno_suffix=".txt", img_suffix=".png"):
    image_folder = Path(image_folder) if isinstance(image_folder, str) else image_folder
    annotation_folder = Path(annotation_folder) if isinstance(annotation_folder, str) else annotation_folder

    data_info_list = []
    for sub_anno_folder in annotation_folder.iterdir():
        if sub_anno_folder.is_dir():
            folder_name = sub_anno_folder.name
            sub_img_folder = image_folder / folder_name
            for anno_file in sub_anno_folder.glob("*"+anno_suffix):
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
    dataset_folder = Path("/hpctmp/e0703350_yoga/datas/datasets_1")
    image_folder = "yoga_images"
    anno_folder = "yoga_anno"
    data_info_list = loopFolder(dataset_folder / image_folder, dataset_folder / anno_folder)
    df = pd.DataFrame(data_info_list)
    df.to_csv("dataset1.csv", index=False)
    