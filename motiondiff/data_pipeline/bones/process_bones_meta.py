import argparse
import os

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="dataset/bones/bones_full_raw_v14/")
parser.add_argument(
    "--feature_dir", type=str, default="dataset/bones/bones_full353_v2.0"
)
parser.add_argument("--initial_csv_file", type=str, default="metadata_240527_v014.csv")
parser.add_argument("--new_csv_file", type=str, default="metadata_240527_v014.csv")
parser.add_argument("--initial_meta_file", type=str, default="meta_240527_v014.csv")
parser.add_argument("--new_meta_file", type=str, default="meta_240527_v014.csv")
parser.add_argument("--update", action="store_true", default=False)
args = parser.parse_args()


if __name__ == "__main__":
    new_meta_path = os.path.join(args.feature_dir, args.new_meta_file)

    if not args.update:
        initial_csv = pd.read_csv(os.path.join(args.data_dir, args.initial_csv_file))
        new_meta_dict = {
            "feature_path": [
                f"{i:06d}.npy" for i in range(len(initial_csv.move_bvh_path))
            ],
            "natural_desc_1": initial_csv.content_natural_desc_1.copy(),
            "natural_desc_2": initial_csv.content_natural_desc_2.copy(),
            "natural_desc_3": initial_csv.content_natural_desc_3.copy(),
            "technical_description": initial_csv.content_technical_description.copy(),
            "short_description": initial_csv.content_short_description.copy(),
            "short_description_2": initial_csv.content_short_description_2.copy(),
            "content": initial_csv.content_name,
        }

    else:
        initial_csv = pd.read_csv(os.path.join(args.data_dir, args.initial_csv_file))
        new_csv = pd.read_csv(os.path.join(args.data_dir, args.new_csv_file))
        initial_meta = pd.read_csv(
            os.path.join(args.feature_dir, args.initial_meta_file)
        )

        new_meta_dict = {
            "feature_path": initial_meta.feature_path,
            "natural_desc_1": initial_csv.content_natural_desc_1.copy(),
            "natural_desc_2": initial_csv.content_natural_desc_2.copy(),
            "natural_desc_3": initial_csv.content_natural_desc_3.copy(),
            "technical_description": initial_csv.content_technical_description.copy(),
            "short_description": initial_csv.content_short_description.copy(),
            "content": initial_csv.content_name,
        }

        num_updates = 0
        f_change = open(
            os.path.join(args.feature_dir, "meta_change_log_240416_v3.txt"), "w"
        )
        initial_bvh_path = list(initial_csv.move_bvh_path)
        for i, bvh_path in enumerate(tqdm(new_csv.move_bvh_path)):
            if bvh_path in initial_bvh_path:
                j = initial_bvh_path.index(bvh_path)
                for col in [
                    "content_natural_desc_1",
                    "content_natural_desc_2",
                    "content_natural_desc_3",
                    "content_technical_description",
                    "content_short_description",
                ]:
                    initial_val = initial_csv[col][j]
                    new_val = new_csv[col][i]

                    # if isinstance(initial_val, str) and initial_val[:-1] == new_val: continue
                    # if isinstance(initial_val, str) and initial_val == new_val[:-1]: continue

                    if (
                        (isinstance(initial_val, str) and "talent" in initial_val)
                        or (isinstance(initial_val, str) and " lose" in initial_val)
                        or initial_val != new_val
                    ):
                        if "lose" in new_val:
                            new_val = new_val.replace(" lose", " loose")
                        if "talent" in new_val:
                            new_val = new_val.replace("talent", "person")

                        num_updates += 1
                        new_meta_dict[col.replace("content_", "")][j] = new_val
                        f_change.write(
                            f"{bvh_path}-{col}:\n{initial_val}\n{new_val}\n\n"
                        )

        f_change.close()
        print(f"Find {num_updates} text updates")

    df = pd.DataFrame(new_meta_dict)
    df.to_csv(new_meta_path, index=False)
