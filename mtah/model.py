"""Frame classifier (H/A/D)."""

import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import features as feat
from . import noise

H, A, D, X = 0, 1, 2, 3
SUBSAMPLE = 4  # every 4th frame for training: neighbouring frames are near-duplicates


def make_classifier(seed=0):
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=seed)


def training_table(games, seeds=(1, 2)):
    """Stack features/labels of degraded, both-orientation copies of the training games."""
    Xs, ys = [], []
    for g in games:
        for s in seeds:
            for view in (g, feat.swapped(g)):
                deg = noise.degrade(view, seed=1000 * s + zlib.crc32(g["name"].encode()) % 997 + 7 * (view is not g))
                F = feat.features(deg)
                lab = view["labels"]
                rows = np.flatnonzero((lab != X) & (view["period"] > 0))[::SUBSAMPLE]
                Xs.append(F.iloc[rows])
                ys.append(lab[rows])
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def fit(games, seeds=(1, 2), model_seed=0):
    Xt, yt = training_table(games, seeds)
    clf = make_classifier(model_seed)
    clf.fit(Xt, yt)
    return clf


def proba(clf, game):
    """P(H, A, D) per frame, averaged over the game and its team-swapped view (symmetric)."""
    p1 = clf.predict_proba(feat.features(game))
    p2 = clf.predict_proba(feat.features(feat.swapped(game)))
    order = list(clf.classes_)
    p1 = p1[:, [order.index(k) for k in (H, A, D)]]
    p2 = p2[:, [order.index(k) for k in (A, H, D)]]  # swapped view: its H is our A
    return 0.5 * (p1 + p2)
