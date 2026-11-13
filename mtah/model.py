"""Frame classifier (H/A/D) + Viterbi decoding into whole possessions."""

import math
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


def viterbi(p, period, c_switch, c_dead, eps=1e-4):
    """Most likely H/A/D path per period with switching penalties (in nats).

    c_switch is the cost of a direct H <-> A change of possession, c_dead the cost of entering or
    leaving a dead ball. Frames outside a period are X.
    """
    out = np.full(len(period), X, dtype=np.int8)
    logp = np.log(np.clip(p, eps, 1.0))
    cost = [[0.0, c_switch, c_dead], [c_switch, 0.0, c_dead], [c_dead, c_dead, 0.0]]
    for per in (1, 2):
        rows = np.flatnonzero(period == per)
        if not len(rows):
            continue
        lp = logp[rows].tolist()
        n = len(rows)
        back = np.zeros((n, 3), dtype=np.int8)
        score = list(lp[0])
        for t in range(1, n):
            new = [0.0, 0.0, 0.0]
            row = back[t]
            for j in range(3):
                best, arg = -math.inf, 0
                for i in range(3):
                    v = score[i] - cost[i][j]
                    if v > best:
                        best, arg = v, i
                new[j] = best + lp[t][j]
                row[j] = arg
            score = new
        path = np.empty(n, dtype=np.int8)
        path[-1] = int(np.argmax(score))
        for t in range(n - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        out[rows] = path
    return out
