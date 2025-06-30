"""
Pipeline for preprocessing session based recommender benchmark datasets
The preprocessing logic in the code of NARM implementation (see reference) is refactored using pandas
Date: created on 05/09/2022
References:
    paper: Neural Attentive Session-based Recommendation
    url: http://arxiv.org/abs/1711.04725
    code: https://github.com/lijingsdu/sessionRec_NARM
Authors: Bo Kang, klinux@live.com
"""

import argparse
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

pd.options.mode.chained_assignment = None

SESSION_ID_FIELD, ITEM_ID_FILED, TIME_FIELD, INTERNAL_ITEM_ID_FIELD = "session_id", "item_id", "time", "iid"
TRAIN_DATA_PREFIX, TEST_DATA_PREFIX = "train_sessions", "test_sessions"

DATA_CONFIGS = {
    "yoochoose": {
        "delimiter": ",",
        "columns": [SESSION_ID_FIELD,
                    TIME_FIELD,
                    ITEM_ID_FILED,
                    "category"],
        "rename": None,
        "time_format": "%Y-%m-%dT%H:%M:%S.%fZ",
    },
    "diginetica": {
        "delimiter": ";",
        "columns": None,
        "rename": {
            "sessionId": SESSION_ID_FIELD,
            "itemId": ITEM_ID_FILED,
            "eventdate": TIME_FIELD
        },
        "time_format": "%Y-%m-%d"
    },
}


def filter_by_session_len(df, min_session_len=2):
    df_session_item_count = (df.groupby(SESSION_ID_FIELD, as_index=False)[ITEM_ID_FILED].count().rename(columns={ITEM_ID_FILED: "counts"}).query(f"counts >= {min_session_len}").drop(columns=["counts"]))
    return df.merge(df_session_item_count, on=SESSION_ID_FIELD)


def filter_by_min_item_freq(df, min_item_freq=5):
    df_item_counts = (pd.DataFrame().from_dict(dict(Counter(df.explode(ITEM_ID_FILED)[ITEM_ID_FILED])), orient='index', columns=["counts"]).reset_index().rename(columns={"index": ITEM_ID_FILED}))
    return df.merge(df_item_counts.query(f"counts >= {min_item_freq}").drop(columns=["counts"]), on=ITEM_ID_FILED)


def split_train_test(df, test_days=7, format="%Y-%m-%d"):
    df[TIME_FIELD] = pd.to_datetime(df[TIME_FIELD], format=format)
    tr_mask = df[TIME_FIELD] <= (df[TIME_FIELD].max() - timedelta(days=test_days))
    df_train, df_test = df[tr_mask], df[~tr_mask]
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def encode_item_id(df_train, df_test):
    label_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=int)
    # label_encoder.fit(np.hstack((df_train[ITEM_ID_FILED].values, df_test[ITEM_ID_FILED].values))[:,None])
    label_encoder.fit(df_train[ITEM_ID_FILED].values[:, None])

    df_train[INTERNAL_ITEM_ID_FIELD] = label_encoder.transform(df_train[ITEM_ID_FILED].values[:, None]).squeeze(1) + 1
    df_train = filter_by_session_len(df_train).reset_index(drop=True)
    df_test[INTERNAL_ITEM_ID_FIELD] = label_encoder.transform(df_test[ITEM_ID_FILED].values[:, None]).squeeze(1) + 1
    df_test = filter_by_session_len(df_test.query(f"{INTERNAL_ITEM_ID_FIELD} != 0")).reset_index(drop=True)
    return df_train, df_test


def groupby_session(df):
    return (df[[SESSION_ID_FIELD, TIME_FIELD, INTERNAL_ITEM_ID_FIELD]].sort_values([SESSION_ID_FIELD, TIME_FIELD]).groupby(SESSION_ID_FIELD, as_index=False)[[INTERNAL_ITEM_ID_FIELD]].agg(list))


def save_data(df, config, prefix):
    output_folder = Path(config.path).parent
    if config.fraction is not None:
        output_folder = output_folder / f"{config.name}_{config.fraction}"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_folder / f"{prefix}.pkl")


def main(config):
    print("read raw data")
    df = pd.read_csv(config.path, delimiter=config.delimiter, names=config.columns)
    if config.rename is not None:
        df.rename(columns=config.rename, inplace=True)

    print("filter session by length")
    df = filter_by_session_len(df, min_session_len=config.min_session_len)

    print("filter item by frequency")
    df = filter_by_min_item_freq(df, min_item_freq=config.min_item_freq)

    print("filter session by length again")
    df = filter_by_session_len(df, min_session_len=config.min_session_len)

    print("split train-test")
    df_train, df_test = split_train_test(df, test_days=config.test_days, format=config.time_format)

    print("encode item id")
    df_train, df_test = encode_item_id(df_train, df_test)

    print("group by session")
    df_train, df_test = groupby_session(df_train), groupby_session(df_test)

    print("compute a fraction of the train data, if applicable")
    if config.fraction is not None:
        df_train = df_train.iloc[-int(len(df_train) / config.fraction):]

    print("compute statistics")
    n_items = pd.concat([df_train, df_test])[INTERNAL_ITEM_ID_FIELD].explode().max() + 1
    avg_seq_len = (df_train[INTERNAL_ITEM_ID_FIELD].apply(len).sum() + df_test[INTERNAL_ITEM_ID_FIELD].apply(len).sum()) / (len(df_train) + len(df_test))

    print(f"\ttrain sessions: {len(df_train):,}")
    print(f"\ttest sessions: {len(df_test):,}")
    print(f"\tall items: {n_items:,}")
    print(f"\tavg length: {avg_seq_len:.2f}")

    print("save data")
    save_data(df_train, config, TRAIN_DATA_PREFIX)
    save_data(df_test, config, TEST_DATA_PREFIX)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="diginetica")
    parser.add_argument("--path", default="./diginetica/train-item-views-sample.csv")
    parser.add_argument("--test_days", default=7, type=int)
    parser.add_argument("--fraction", default=None, type=int)
    parser.add_argument("--min_session_len", default=2, type=int)
    parser.add_argument("--min_item_freq", default=5, type=int)

    args = parser.parse_args()
    args.__dict__.update(DATA_CONFIGS[args.name])
    main(args)
"""
sample datasets:
python preprocess_session_based.py --name diginetica --path ./diginetica/train-item-views-sample.csv
python preprocess_session_based.py --name yoochoose --path ./yoochoose/yoochoose-clicks-sample.dat --test_days 2

full datasets:
python preprocess_session_based.py --name diginetica --path ./diginetica/train-item-views.csv
python preprocess_session_based.py --name yoochoose --path ./yoochoose/yoochoose-clicks.dat --fraction 4 --test_days 1
python preprocess_session_based.py --name yoochoose --path ./yoochoose/yoochoose-clicks.dat --fraction 64 --test_days 1
"""
