---
title: Quick start
description: Train a click predictor on Criteo and a two-tower retrieval model on MovieLens.
---

# Quick start

To predict whether someone will click an ad, start with DeepFM. To find movies a user might like based on what they have watched, skip to DSSM. Each example runs independently on a small dataset included in the repository.

## Set up the environment

Open a terminal at the repository root and run `uv sync`. Run the Python examples from that directory with `uv run python`, or select the same environment as your Jupyter kernel. Both examples default to CPU. For NVIDIA, AMD or Ascend hardware, see the [installation guide](/guide/install).

## Example 1. Predict clicks with DeepFM

Each Criteo row represents one ad impression. A `label` of `1` means the user clicked the ad, and `0` means they did not. DeepFM takes the numeric and categorical features from that row and predicts the probability of a click.

### Prepare the Criteo inputs {#criteo-input}

Split the data into training, validation and test sets first. Build the categorical vocabularies and fit the numeric scaler on the training set. A category first seen in the test set then maps to `0`, just as a new category would at prediction time.

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

# Read the sample bundled with the repository
data_path = "examples/ranking/data/criteo/criteo_sample.csv"
data = pd.read_csv(data_path)
print(f"Rows: {len(data)}")

# Criteo columns I1-I13 are numeric and C1-C26 are categorical.
dense_features = [f"I{i}" for i in range(1, 14)]
sparse_features = [f"C{i}" for i in range(1, 27)]

# Split before fitting. Stratify by label so all three small splits contain clicks and non-clicks.
train, holdout = train_test_split(data, test_size=0.3, stratify=data["label"], random_state=2022)
val, test = train_test_split(holdout, test_size=2/3, stratify=holdout["label"], random_state=2022)

scaler = MinMaxScaler().fit(train[dense_features].fillna(0))
vocabs = {
    name: {key: i for i, key in enumerate(train[name].dropna().astype(str).unique(), start=1)}
    for name in sparse_features
}

def encode_features(frame):
    dense = scaler.transform(frame[dense_features].fillna(0)).astype("float32")
    x = {name: dense[:, i] for i, name in enumerate(dense_features)}
    for name in sparse_features:
        x[name] = frame[name].astype("string").map(vocabs[name]).fillna(0).to_numpy(dtype="int64")
    return x

dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [
    SparseFeature(name, vocab_size=len(vocabs[name]) + 1, embed_dim=16, padding_idx=0)
    for name in sparse_features
]

# Reuse the training vocabularies and scaler for validation, testing and prediction. Map missing or unseen categories to 0.
dg = DataGenerator(encode_features(train), train["label"].to_numpy())
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=encode_features(val), y_val=val["label"].to_numpy(),
    x_test=encode_features(test), y_test=test["label"].to_numpy(),
    batch_size=256,
)

```

### Train and evaluate DeepFM

Continue in the same Python session. The trainer runs for two epochs and prints validation AUC after each one. The last line reports AUC on the held-out test set.

```python
# The FM branch models pairwise interactions, and the MLP learns higher-order relationships.
model = DeepFM(
    deep_features=dense_feas + sparse_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=2,
    device="cpu",  # Use "cuda:0" for an NVIDIA or ROCm GPU, or "npu:0" in an Ascend environment.
    model_path="./saved/quick_start_deepfm",
)

# Create the directory before the trainer writes its checkpoint.
os.makedirs("./saved/quick_start_deepfm", exist_ok=True)
trainer.fit(train_dl, val_dl)
# Pass the trained model explicitly to evaluate.
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

With timestamped impression logs, reserve the later records for validation and testing. When scoring a new impression, reuse the `vocabs` and `scaler` fitted during training. [Compare PyTorch and ONNX on one impression](/serving/demo)

To see the training output directly, open a terminal at the repository root and run the bundled script.

```bash
cd examples/ranking
uv run python run_criteo.py --model_name deepfm --epoch 2 --device cpu
```

---

## Example 2. Retrieve movies with DSSM

DSSM computes one vector per user and one per movie. The user tower reads profile attributes and viewing history. The item tower reads the movie ID and genre. After training, you can store the catalog vectors and search for movies close to a given user vector.

```python
import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match

torch.manual_seed(2022)

# Read the sample bundled with the repository
data_path = "examples/matching/data/ml-1m/ml-1m_sample.csv"
data = pd.read_csv(data_path)
data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
print(f"Rows: {len(data)}")

# Encode categorical features
# This example assumes a known catalog. Index the user and movie attributes, then hold out each user's last interaction.
user_col, item_col = "user_id", "movie_id"
sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "cate_id"]

# Reserve index 0 for padding and unknown IDs.
feature_max_idx = {}
for feat in sparse_features:
    encoder = LabelEncoder()
    data[feat] = encoder.fit_transform(data[feat]) + 1
    if feat == item_col:
        raw_movie_ids = encoder.classes_  # Index i corresponds to the original movie ID raw_movie_ids[i - 1].
    feature_max_idx[feat] = data[feat].max() + 1

# Join user and item attributes to the sampled interactions when building model inputs.
user_cols = ["user_id", "gender", "age", "occupation", "zip"]
item_cols = ["movie_id", "cate_id"]

user_profile = data[user_cols].drop_duplicates("user_id")
item_profile = data[item_cols].drop_duplicates("movie_id")

# Build examples from viewing histories
# Pair each positive example (label 1) with three negatives (label 0).
df_train, df_test = generate_seq_feature_match(
    data,
    user_col=user_col,
    item_col=item_col,
    time_col="timestamp",
    item_attribute_cols=[],
    sample_method=1,
    mode=0,
    neg_ratio=3,
    min_item=0,
)

# Join attributes by user and movie ID, then pad viewing histories to the same length.
x_train = gen_model_input(
    df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50
)
y_train = x_train.pop("label")
x_test = gen_model_input(
    df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50
)

# Define the features
# The user tower combines profile features with a pooled viewing history.
user_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in user_cols
]
# Mean pooling turns a variable number of watched movies into one fixed-width vector.
user_features += [
    SequenceFeature(
        "hist_movie_id",
        vocab_size=feature_max_idx["movie_id"],
        embed_dim=16,
        pooling="mean",
        shared_with="movie_id", padding_idx=0,
    )
]

item_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16, padding_idx=0)
    for name in item_cols
]

# DataLoader
# Build a training loader and separate inference loaders for users and movies.
all_item = df_to_dict(item_profile)
test_user = x_test

dg = MatchDataGenerator(x=x_train, y=y_train)
train_dl, test_dl, item_dl = dg.generate_dataloader(
    test_user,
    all_item,
    batch_size=256,
    num_workers=0,  # Read in the current process on Windows or in a notebook.
)

# Define the model
# The two towers map users and movies into the same vector space.
model = DSSM(
    user_features,
    item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64], "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "activation": "prelu"},
)

# Train the model and extract embeddings
trainer = MatchTrainer(
    model,
    mode=0,
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=2,
    device="cpu",  # Use "cuda:0" for an NVIDIA or ROCm GPU, or "npu:0" in an Ascend environment.
    model_path="./saved/quick_start_dssm",
)

# Create the checkpoint directory. inference_embedding reads the weights from this path.
os.makedirs("./saved/quick_start_dssm", exist_ok=True)
trainer.fit(train_dl)

# Keep one vector per test user and one per catalog movie, in input order.
user_embedding = trainer.inference_embedding(
    model=model,
    mode="user",
    data_loader=test_dl,
    model_path="./saved/quick_start_dssm",
)
item_embedding = trainer.inference_embedding(
    model=model,
    mode="item",
    data_loader=item_dl,
    model_path="./saved/quick_start_dssm",
)

print(f"User embedding shape: {user_embedding.shape}")
print(f"Item embedding shape: {item_embedding.shape}")
```

The final two lines print the shapes of the embedding matrices. The user matrix has one row per test user, the movie matrix has one row per candidate movie, and both have 64 columns. To check whether those vectors retrieve the last movie a user watched, open a new terminal at the repository root and run the Annoy example.

```bash
cd examples/matching
uv run --extra annoy python run_ml_dssm.py --epoch 1 --device cpu
```

This script writes weights to `examples/matching/data/ml-1m/saved/model.pth`. The Python example above writes them to `saved/quick_start_dssm/model.pth`.

---

## Run on a different device

Change `device="cpu"` to `"cuda:0"` for the first NVIDIA or AMD GPU, or to `"npu:0"` in a configured Ascend environment. On Windows or in a notebook, start with `num_workers=0` to read the data in the current process.

## Next steps

- Train WideDeep, DeepFM, DCN or DIN in the [ranking tutorial](../tutorials/ctr.md).
- Work with DSSM, YoutubeDNN or MIND in the [retrieval tutorial](../tutorials/retrieval.md).
- Train MMOE, PLE or ESMM in the [multi-task tutorial](../tutorials/pipeline.md).
- Export a model to ONNX or build a vector index in the [serving guide](../serving/intro.md).

---

## Common questions

### Where are the trained weights?

In both examples, `fit` writes `model.pth` to the trainer's `model_path`. To reload it, create the model with the same feature definitions and configuration, then restore the weights with `model.load_state_dict(...)`. Keep the categorical vocabularies too. An integer ID must refer to the same movie or category at training and prediction time.

### How do I export a model to ONNX?

A ranker can be exported as a single file. With DSSM, you can export the user and item towers separately and refresh their embeddings on different schedules. [Export and check a ranking model](/serving/demo) · [Export two towers](/serving/onnx)
