"""
Disclaimer: A substantial portion of the code contained in this script is directly taken from
this public GitHub repository: https://github.com/kumo-ai/ContextGNN/tree/xinwei_add_static_data_and_model_v1 and
adapted to work within the framework Elliot (https://elliot.readthedocs.io/en/latest/).
Please refer to the above cited GitHub repository and to the original paper
of ContextGNN (https://arxiv.org/abs/2411.19513) for further details.
"""

import argparse
import os
import os.path as osp
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from torch_geometric.seed import seed_everything

PSEUDO_TIME = "pseudo_time"
TRAIN_SET_TIMESTAMP = pd.Timestamp("1970-01-01")
SRC_ENTITY_COL = "user_id"
DST_ENTITY_COL = "item_id"

def _read_mapping_robust(path: str) -> pd.DataFrame:
    """
    Read a two-column ID mapping file robustly.
    The function expects a whitespace-delimited file with either columns remap_id and org_id or at least two columns (the first two are used).
    Output columns are normalized to strings.

    Parameters
    ----------
    path : str
        Path to the mapping file (e.g. user_list.txt or item_list.txt).

    Returns
    -------
    pandas.DataFrame
        DataFrame with exactly two string columns:
        - remap_id : str
        - org_id : str

    Raises
    ------
    FileNotFoundError
        If the file at path does not exist.
    """

    if not osp.exists(path):
        raise FileNotFoundError(f"Missing mapping at {path}")
    
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python")

        if df.shape[1] < 2:
            raise ValueError
        
        cols = list(df.columns)

        if "remap_id" in cols and "org_id" in cols:
            df = df[["remap_id", "org_id"]].copy()
        else:
            df = df.iloc[:, :2].copy()
            df.columns = ["remap_id", "org_id"]
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=["remap_id", "org_id"], engine="python")
    
    df["remap_id"] = df["remap_id"].astype(str)
    df["org_id"] = df["org_id"].astype(str)

    return df

def _read_ctx_file(path: str) -> pd.DataFrame:
    """
    Read a ContextGNN interaction list file.
    Each non-empty line is parsed as <user> <item_1> <item_2> ...
    Lines with users but no items are skipped.

    Parameters
    ----------
    path : str
        Path to the interactions file (e.g. train.txt or test.txt).

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:
        - user : str
        - items : list of str (list of item identifiers interacted by the user on that line)

    Raises
    ------
    FileNotFoundError
        If the file at path does not exist.
    """

    if not osp.exists(path):
        raise FileNotFoundError(f"Missing interactions file at {path}")
    
    users: List[str] = []
    items_list: List[List[str]] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            u, its = parts[0], parts[1:]

            if len(its) == 0:
                continue

            users.append(u)
            items_list.append(its)

    return pd.DataFrame({"user": users, "items": items_list})

def _explode_pairs(df_ui_list: pd.DataFrame) -> pd.DataFrame:
    """
    Convert (user, [items...]) rows into (user, item) rows.

    Parameters
    ----------
    df_ui_list : pandas.DataFrame
        DataFrame with columns:
        - user : str
        - items : list of str

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - user : str
        - item : str

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    
    df = df_ui_list.copy()
    df = df.explode("items").reset_index(drop=True)
    df.rename(columns={"items": "item"}, inplace=True)

    return df[["user", "item"]]

def _log2_bins(deg: pd.Series) -> pd.Series:
    """
    Bucket degrees into floor(log2(degree)) bins.
    Degrees are clipped at 1 before applying log2 so that zeros map to bin 0.

    Parameters
    ----------
    deg : pandas.Series
        Non-negative degree counts indexed by entity ID.

    Returns
    -------
    pandas.Series
        Integer bin index per entity (same index as deg).
    """
    
    d = deg.clip(lower=1).astype(float)
    return np.floor(np.log2(d)).astype(int)

def _sample_stratified(ids: List[str], deg: pd.Series, target: int, rng: np.random.RandomState) -> Set[str]:
    """
    Stratified sampling over log2-degree bins.
    Preserves the popularity/long-tail proportions by allocating the sample size across degree bins, with final rounding adjustments and uniform top-up if some bins are too small.

    Parameters
    ----------
    ids : list of str
        Candidate IDs to sample from.
    deg : pandas.Series
        Degree series indexed by ID; must contain entries for all ids.
    target : int
        Desired number of samples. If target >= len(ids), returns all ids.
    rng : numpy.random.RandomState
        Random generator used for reproducible sampling without replacement.

    Returns
    -------
    set of str
        The sampled ID set (size <= target).

    Raises
    ------
    KeyError
        If deg does not contain degrees for some IDs in ids.
    ValueError
        If target is negative.
    """
    
    if target >= len(ids):
        return set(ids)

    bins = _log2_bins(deg.loc[ids])
    by_bin: Dict[int, List[str]] = {}

    for i, b in zip(ids, bins):
        by_bin.setdefault(int(b), []).append(i)

    total = float(len(ids))
    desired = {b: int(round(len(v) / total * target)) for b, v in by_bin.items()}

    diff = target - sum(desired.values())

    for b in sorted(by_bin.keys(), key=lambda x: -len(by_bin[x])):
        if diff == 0:
            break

        if diff > 0:
            desired[b] += 1
            diff -= 1
        elif desired[b] > 0:
            desired[b] -= 1
            diff += 1

    sampled: Set[str] = set()

    for b, candidates in by_bin.items():
        k = min(desired[b], len(candidates))

        if k > 0:
            sampled |= set(rng.choice(candidates, size=k, replace=False).tolist())

    remain = [i for i in ids if i not in sampled]
    
    if len(sampled) < target and remain:
        extra = min(target - len(sampled), len(remain))
        sampled |= set(rng.choice(remain, size=extra, replace=False).tolist())

    return sampled

def _build_subset(in_dir: str, ratio: float, seed: int = 42, refill_rounds: int = 3):
    """
    Build a degree-stratified subgraph for train/test and filtered mappings.

    The function:
        1) Reads mappings and ContextGNN train/test lists;
        2) Samples users and items with log2-degree stratification on the full train;
        3) Induces the train subgraph and prunes isolated nodes;
        4) Refills up to the targets for a few rounds;
        5) Constructs a consistent test subset (users must appear in both train and test; items must appear in train);
        6) Filters mapping files to the active IDs.

    Parameters
    ----------
    in_dir : str
        Dataset directory containing user_list.txt, item_list.txt, train.txt and test.txt.
    ratio : float
        Fraction of users and items to retain (0 < ratio <= 1).
    seed : int, default=42
        Random seed for sampling.
    refill_rounds : int, default=3
        Number of refill iterations to approach the target counts.

    Returns
    -------
    tuple
        - sub_train_pairs : pandas.DataFrame
            Columns ['user', 'item'] for the train subset.
        - sub_test_pairs : pandas.DataFrame
            Columns ['user', 'item'] for the test subset (filtered).
        - src_map_sub : pandas.DataFrame
            Filtered user mapping with columns ['remap_id', 'org_id'].
        - dst_map_sub : pandas.DataFrame
            Filtered item mapping with columns ['remap_id', 'org_id'].

    Raises
    ------
    FileNotFoundError
        If any of the required input files are missing.
    KeyError
        If expected columns are missing in the input files.
    ValueError
        If ratio is invalid (<= 0 or > 1) or if intermediate inputs are malformed.
    """
    
    src_map = _read_mapping_robust(osp.join(in_dir, "user_list.txt"))
    dst_map = _read_mapping_robust(osp.join(in_dir, "item_list.txt"))

    train_list = _read_ctx_file(osp.join(in_dir, "train.txt"))
    test_list = _read_ctx_file(osp.join(in_dir, "test.txt"))

    train_pairs = _explode_pairs(train_list)
    u_deg_full = train_pairs.groupby("user").size()
    i_deg_full = train_pairs.groupby("item").size()

    users = u_deg_full.index.tolist()
    items = i_deg_full.index.tolist()

    rng = np.random.RandomState(seed)
    target_u = max(1, int(round(len(users) * ratio)))
    target_i = max(1, int(round(len(items) * ratio)))

    sel_users = _sample_stratified(users, u_deg_full, target_u, rng)
    sel_items = _sample_stratified(items, i_deg_full, target_i, rng)

    sub_train_pairs = train_pairs[
        train_pairs["user"].isin(sel_users) & train_pairs["item"].isin(sel_items)
    ].copy()

    def prune(sel_u: Set[str], sel_i: Set[str], df: pd.DataFrame):
        active_u = set(df["user"].unique())
        active_i = set(df["item"].unique())
        sel_u &= active_u
        sel_i &= active_i
        return sel_u, sel_i, df

    sel_users, sel_items, sub_train_pairs = prune(sel_users, sel_items, sub_train_pairs)

    for _ in range(refill_rounds):
        need_u = max(0, target_u - len(sel_users))
        need_i = max(0, target_i - len(sel_items))

        if need_u == 0 and need_i == 0:
            break

        if need_u > 0:
            remaining_u = [u for u in users if u not in sel_users]

            if remaining_u:
                add_u = _sample_stratified(remaining_u, u_deg_full.loc[remaining_u], need_u, rng)
                sel_users |= set(add_u)

        if need_i > 0:
            remaining_i = [i for i in items if i not in sel_items]

            if remaining_i:
                add_i = _sample_stratified(remaining_i, i_deg_full.loc[remaining_i], need_i, rng)
                sel_items |= set(add_i)

        sub_train_pairs = train_pairs[
            train_pairs["user"].isin(sel_users) & train_pairs["item"].isin(sel_items)
        ].copy()

        sel_users, sel_items, sub_train_pairs = prune(sel_users, sel_items, sub_train_pairs)

    test_pairs = _explode_pairs(test_list)

    sub_test_pairs = test_pairs[
        test_pairs["user"].isin(sel_users) & test_pairs["item"].isin(sel_items)
    ].copy()

    train_u_deg = sub_train_pairs.groupby("user").size()
    test_u_deg = sub_test_pairs.groupby("user").size()
    valid_users = set(train_u_deg.index) & set(test_u_deg.index)
    sub_train_pairs = sub_train_pairs[sub_train_pairs["user"].isin(valid_users)].copy()
    sub_test_pairs = sub_test_pairs[sub_test_pairs["user"].isin(valid_users)].copy()

    train_i_deg = sub_train_pairs.groupby("item").size()
    active_items = set(train_i_deg.index)
    sub_train_pairs = sub_train_pairs[sub_train_pairs["item"].isin(active_items)].copy()
    sub_test_pairs = sub_test_pairs[sub_test_pairs["item"].isin(active_items)].copy()

    def _as_int(x):
        try:
            return int(x)
        except Exception:
            return x

    src_map_sub = src_map[src_map["remap_id"].isin(sorted(valid_users, key=_as_int))].copy()
    dst_map_sub = dst_map[dst_map["remap_id"].isin(sorted(active_items, key=_as_int))].copy()

    return sub_train_pairs, sub_test_pairs, src_map_sub, dst_map_sub

def _write_outputs(
    out_dir: str,
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    src_map: pd.DataFrame,
    dst_map: pd.DataFrame,
    seed: int
):
    """
    Write Elliot-compatible TSVs and parquet artifacts for the subset.

    Produces:
        - train_elliot.tsv: two-column TSV (user_id, item_id), one row per (user,item).
        - test_elliot.tsv: two-column TSV (user_id, item_id) expanded from per-user lists.
        - src_df.tsv / dst_df.tsv / target_df.tsv / test_df.tsv / train_df.tsv: parquet files (note: saved with .tsv extension, matching the original script).

    Parameters
    ----------
    out_dir : str
        Output directory to create (if absent) and write into.
    train_pairs : pandas.DataFrame
        Train pairs with columns ['user', 'item'].
    test_pairs : pandas.DataFrame
        Test pairs with columns ['user', 'item'].
    src_map : pandas.DataFrame
        User mapping with columns ['remap_id', 'org_id'].
    dst_map : pandas.DataFrame
        Item mapping with columns ['remap_id', 'org_id'].
    seed : int
        Random seed used for shuffling rows deterministically.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the output directory cannot be created or files cannot be written.
    KeyError
        If required columns are missing from the provided DataFrames.
    ImportError
        If a parquet engine (e.g., pyarrow) is not installed.
    ValueError
        If inputs are malformed (e.g., non-list items when grouping).
    """

    os.makedirs(out_dir, exist_ok=True)

    train_grouped = train_pairs.groupby("user")["item"].apply(list).reset_index()
    train_df = train_grouped.rename(columns={"user": SRC_ENTITY_COL, "item": DST_ENTITY_COL}).copy()
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP
    train_df[PSEUDO_TIME] = train_df[PSEUDO_TIME].astype("int64")

    test_grouped = test_pairs.groupby("user")["item"].apply(list).reset_index()
    test_df = test_grouped.rename(columns={"user": SRC_ENTITY_COL, "item": DST_ENTITY_COL}).copy()
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP + pd.Timedelta(days=1)

    train_df_explode = train_df.explode(DST_ENTITY_COL).reset_index(drop=True)
    target_df = train_df_explode

    target_df[[SRC_ENTITY_COL, DST_ENTITY_COL]].to_csv(
        osp.join(out_dir, "train_elliot.tsv"), sep="\t", index=False, header=False
    )

    with open(osp.join(out_dir, "test_elliot.tsv"), "w") as f:
        for _, row in test_df.iterrows():
            for it in row[DST_ENTITY_COL]:
                f.write(f"{row[SRC_ENTITY_COL]}\t{it}\n")

    src_out = src_map.copy()
    src_out[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP
    src_out[PSEUDO_TIME] = src_out[PSEUDO_TIME].astype("int64")

    dst_out = dst_map.copy()
    dst_out[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP
    dst_out[PSEUDO_TIME] = dst_out[PSEUDO_TIME].astype("int64")

    src_out.to_parquet(osp.join(out_dir, "src_df.tsv"), engine="pyarrow", index=False)
    dst_out.to_parquet(osp.join(out_dir, "dst_df.tsv"), engine="pyarrow", index=False)
    target_df.to_parquet(osp.join(out_dir, "target_df.tsv"), engine="pyarrow", index=False)
    test_df.to_parquet(osp.join(out_dir, "test_df.tsv"), engine="pyarrow", index=False)
    train_df.to_parquet(osp.join(out_dir, "train_df.tsv"), engine="pyarrow", index=False)

def main():
    parser = argparse.ArgumentParser(description="map_rel_bench with --ratio for degree-stratified subsets.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (folder under data_root)")
    parser.add_argument("--ratio", type=float, required=True, help="Fraction to keep (e.g., 0.10 for 10%)")
    parser.add_argument("--data_root", type=str, default="./data", help="Root data folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if not (0 < args.ratio <= 1.0):
        raise ValueError("--ratio must be in (0, 1].")

    dataset = args.dataset
    in_dir = osp.join(args.data_root, dataset)

    if not osp.isdir(in_dir):
        raise FileNotFoundError(f"Dataset folder not found: {in_dir}")

    seed_everything(args.seed)

    if args.ratio >= 0.9999:
        user_path = osp.join(in_dir, "user_list.txt")

        src_df = pd.read_csv(user_path, delim_whitespace=True)
        src_df = src_df.drop(columns=["org_id"]).rename(columns={"remap_id": SRC_ENTITY_COL})
        src_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP
        src_df[PSEUDO_TIME] = src_df[PSEUDO_TIME].astype("int64")

        item_path = osp.join(in_dir, "item_list.txt")

        dst_df = pd.read_csv(item_path, delim_whitespace=True)
        dst_df = dst_df.drop(columns=["org_id"]).rename(columns={"remap_id": DST_ENTITY_COL})
        dst_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP
        dst_df[PSEUDO_TIME] = dst_df[PSEUDO_TIME].astype("int64")

        train_path = osp.join(in_dir, "train.txt")
        user_ids, item_ids = [], []

        with open(train_path, "r") as file:
            for line in file:
                values = list(map(int, line.split()))
                user_id = values[0]
                item_ids_for_user = values[1:]
                user_ids.append(user_id)
                item_ids.append(item_ids_for_user)

        train_df = pd.DataFrame({SRC_ENTITY_COL: user_ids, DST_ENTITY_COL: item_ids}).sample(
            frac=1, random_state=args.seed
        ).reset_index(drop=True)

        train_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP
        train_df[PSEUDO_TIME] = train_df[PSEUDO_TIME].astype("int64")

        test_path = osp.join(in_dir, "test.txt")
        user_ids, item_ids = [], []

        with open(test_path, "r") as file:
            for line in file:
                values = list(map(int, line.split()))
                user_id = values[0]
                item_ids_for_user = values[1:]
                user_ids.append(user_id)
                item_ids.append(item_ids_for_user)

        test_df = pd.DataFrame({SRC_ENTITY_COL: user_ids, DST_ENTITY_COL: item_ids}).sample(
            frac=1, random_state=args.seed
        ).reset_index(drop=True)

        test_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP + pd.Timedelta(days=1)

        train_df_explode = train_df.explode(DST_ENTITY_COL).reset_index(drop=True)
        target_df = train_df_explode

        target_df[[SRC_ENTITY_COL, DST_ENTITY_COL]].to_csv(
            osp.join(in_dir, "train_elliot.tsv"), sep="\t", index=False, header=False
        )

        with open(osp.join(in_dir, "test_elliot.tsv"), "w") as file:
            for _, row in test_df.iterrows():
                for it in row[DST_ENTITY_COL]:
                    file.write(str(row[SRC_ENTITY_COL]) + "\t" + str(it) + "\n")

        src_df.to_parquet(osp.join(in_dir, "src_df.tsv"), engine="pyarrow", index=False)
        dst_df.to_parquet(osp.join(in_dir, "dst_df.tsv"), engine="pyarrow", index=False)
        target_df.to_parquet(osp.join(in_dir, "target_df.tsv"), engine="pyarrow", index=False)
        test_df.to_parquet(osp.join(in_dir, "test_df.tsv"), engine="pyarrow", index=False)
        train_df.to_parquet(osp.join(in_dir, "train_df.tsv"), engine="pyarrow", index=False)

        print("Full dataset processed (ratio=1). Outputs written to:", in_dir)

        return
    
    sub_pct = int(round(args.ratio * 100))
    out_dir = osp.join(args.data_root, f"{dataset}_sub{sub_pct}")

    sub_train_pairs, sub_test_pairs, src_map_sub, dst_map_sub = _build_subset(
        in_dir, args.ratio, seed=args.seed
    )

    _write_outputs(out_dir, sub_train_pairs, sub_test_pairs, src_map_sub, dst_map_sub, seed=args.seed)

    print(f"Subset processed (ratio={args.ratio}). Outputs written to: {out_dir}")

if __name__ == "__main__":
    main()