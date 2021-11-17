import pandas as pd
from pathlib import Path
import random

score_file = "score_data_raw.csv"
output_file = "score_data_combine.csv"
category_file = "datasets_1_filtered.csv"
column_pattern = ["keys_{}_anno", "score_{}"]
weight_column = "weight"

target_columns = [
    s.format(i) for s in column_pattern for i in range(3)
]

if __name__ == '__main__':
    score_df = pd.read_csv(score_file)
    score_df = score_df[target_columns]
    score_df[weight_column] = 5.0
    # print(score_df.head())

    cat_df = pd.read_csv(category_file)
    # print(cat_df.head())

    # from cat_df select 3 items:
    # first two items belong to same label
    # the third one belongs to another category
    for idx, row in cat_df.iterrows():
        pool = cat_df[(cat_df["label"] == row["label"]) & (cat_df.index != idx)]
        same_idx = random.randint(0, len(pool)-1)
        same_item = pool.iloc[same_idx]

        diff_pool = cat_df[cat_df["label"] != row["label"]]
        diff_idx = random.randint(0, len(diff_pool)-1)
        diff_item = diff_pool.iloc[diff_idx]

        new_row = {"weight": 1.0}

        for i, item in enumerate([row, same_item, diff_item]):
            new_row[column_pattern[0].format(i)] = item["anno"]
            new_row[column_pattern[1].format(i)] = i

        score_df = score_df.append(new_row, ignore_index=True)

    score_df.to_csv(output_file, index=False)

    