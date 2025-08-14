"""
Disclaimer: A substantial portion of the code contained in this script is directly taken from
this public GitHub repository: https://github.com/kumo-ai/ContextGNN/tree/xinwei_add_static_data_and_model_v1 and
adapted to work within the framework Elliot (https://elliot.readthedocs.io/en/latest/).
Please refer to the above cited GitHub repository and to the original paper
of ContextGNN (https://arxiv.org/abs/2411.19513) for further details.
"""

import argparse
import os.path as osp
import pandas as pd
from torch_geometric.seed import seed_everything

PSEUDO_TIME = "pseudo_time"
TRAIN_SET_TIMESTAMP = pd.Timestamp("1970-01-01")
SRC_ENTITY_TABLE = "user_table"
DST_ENTITY_TABLE = "item_table"
TRANSACTION_TABLE = "transaction_table"
SRC_ENTITY_COL = "user_id"
DST_ENTITY_COL = "item_id"

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='gowalla')
args = parser.parse_args()

dataset = args.dataset
input_data_dir = f'./data/{dataset}/'
seed = 42

seed_everything(seed)

# Load user data
user_path = osp.join(input_data_dir, "user_list.txt")
src_df = pd.read_csv(user_path, delim_whitespace=True)
# Drop `org_id` and rename `remap_id` to `user_id`
src_df = src_df.drop(columns=['org_id']).rename(
    columns={'remap_id': SRC_ENTITY_COL})
src_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP
src_df[PSEUDO_TIME] = src_df[PSEUDO_TIME].astype('int64')
NUM_SRC_NODES = len(src_df)

# Load item data
item_path = osp.join(input_data_dir, "item_list.txt")
dst_df = pd.read_csv(item_path, delim_whitespace=True)
# Drop `org_id` and rename `remap_id` to `item_id`
dst_df = dst_df.drop(columns=['org_id']).rename(
    columns={'remap_id': DST_ENTITY_COL})
dst_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP
dst_df[PSEUDO_TIME] = dst_df[PSEUDO_TIME].astype('int64')
NUM_DST_NODES = len(dst_df)

# Load user item link for train data
train_path = osp.join(input_data_dir, "train.txt")
user_ids = []
item_ids = []
with open(train_path, 'r') as file:
    for line in file:
        values = list(map(int, line.split()))
        user_id = values[0]
        item_ids_for_user = values[1:]
        user_ids.append(user_id)
        item_ids.append(item_ids_for_user)
train_df = pd.DataFrame({SRC_ENTITY_COL: user_ids, DST_ENTITY_COL: item_ids})
# Shuffle train data
train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
# Add pseudo time column
train_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP
train_df[PSEUDO_TIME] = train_df[PSEUDO_TIME].astype('int64')

# load user item link for test data
test_path = osp.join(input_data_dir, "test.txt")
user_ids = []
item_ids = []
with open(test_path, 'r') as file:
    for line in file:
        values = list(map(int, line.split()))
        user_id = values[0]
        item_ids_for_user = values[1:]
        user_ids.append(user_id)
        item_ids.append(item_ids_for_user)
test_df = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids})
# Shuffle train data
test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
# Add pseudo time column
test_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP + pd.Timedelta(days=1)

train_df_explode = train_df.explode(DST_ENTITY_COL).reset_index(drop=True)
target_df = train_df_explode

target_df[['user_id', 'item_id']].to_csv(input_data_dir + 'train_elliot.tsv', sep='\t', index=False, header=False)

with open(input_data_dir + 'test_elliot.tsv', 'w') as file:
    for idx, row in test_df.iterrows():
        for it in row['item_id']:
            file.write(str(row['user_id'])+'\t'+str(it)+'\n')

src_df.to_parquet(input_data_dir + 'src_df.tsv', engine='pyarrow', index=False)
dst_df.to_parquet(input_data_dir + 'dst_df.tsv', engine='pyarrow', index=False)
target_df.to_parquet(input_data_dir + 'target_df.tsv', engine='pyarrow', index=False)
test_df.to_parquet(input_data_dir + 'test_df.tsv', engine='pyarrow', index=False)
train_df.to_parquet(input_data_dir + 'train_df.tsv', engine='pyarrow', index=False)