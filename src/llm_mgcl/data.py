"""Loading, filtering and splitting the Yelp Multimodal Recommendation dataset."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def load_raw(root: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load reviews, business coordinates and photo summaries.

    ``root`` is either the Hugging Face path ``hf://datasets/wzehui/Yelp-Multimodal-Recommendation``
    or a local directory containing the same CSV files.
    """
    reviews = pd.read_csv(f"{root}/review.csv", usecols=["user_id", "business_id", "stars"])
    business = pd.read_csv(f"{root}/business.csv", usecols=["business_id", "latitude", "longitude"])
    # Note: the file name in the dataset is spelled "photo_summay.csv".
    photo_summary = pd.read_csv(f"{root}/photo_summay.csv")
    return reviews, business, photo_summary


def filter_users(reviews: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    """Drop rows with missing values and users with fewer than ``min_interactions`` reviews."""
    interactions = reviews.dropna().copy()
    cnt = interactions.groupby("user_id")["business_id"].count()
    valid_users = cnt[cnt >= min_interactions].index
    return interactions[interactions["user_id"].isin(valid_users)]


def split_per_user(
    interactions: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle each user's interactions and split them 80/10/10 into train/val/test."""
    train_frac, val_frac = ratios[0], ratios[0] + ratios[1]
    train_parts, val_parts, test_parts = [], [], []
    for _, df in interactions.groupby("user_id"):
        df = df.sample(frac=1, random_state=seed)
        n = len(df)
        train_end = int(train_frac * n)
        val_end = int(val_frac * n)
        train_parts.append(df.iloc[:train_end])
        val_parts.append(df.iloc[train_end:val_end])
        test_parts.append(df.iloc[val_end:])
    return pd.concat(train_parts), pd.concat(val_parts), pd.concat(test_parts)


def positive_sets(df: pd.DataFrame) -> dict[str, set]:
    """``user_id -> set(business_id)``."""
    return df.groupby("user_id")["business_id"].apply(set).to_dict()


@dataclass
class Interactions:
    """Train/val/test split plus the integer id mappings (built from the training set only)."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    user2idx: dict = field(init=False)
    item2idx: dict = field(init=False)

    def __post_init__(self):
        users = self.train_df["user_id"].unique()
        items = self.train_df["business_id"].unique()
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {b: i for i, b in enumerate(items)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2item = {i: b for b, i in self.item2idx.items()}
        self.train_pos = positive_sets(self.train_df)
        self.val_pos = positive_sets(self.val_df)
        self.test_pos = positive_sets(self.test_df)

    @property
    def n_users(self) -> int:
        return len(self.user2idx)

    @property
    def n_items(self) -> int:
        return len(self.item2idx)

    def item_train_counts(self) -> pd.Series:
        """Number of training interactions per item (``business_id`` index)."""
        return self.train_df.groupby("business_id").size()


def prepare_interactions(reviews: pd.DataFrame, min_interactions: int, ratios, seed: int) -> Interactions:
    interactions = filter_users(reviews, min_interactions)
    train_df, val_df, test_df = split_per_user(interactions, ratios, seed)
    return Interactions(train_df, val_df, test_df)
