"""Train and tune on the EXTERNAL Metrica sample games only. Never reads the challenge match.

    python train.py              # leave-one-game-out validation + penalty grid, then final fit

Writes models/model.joblib and models/params.json. The decoding penalties are chosen by
leave-one-game-out spellF1 on the sample games, scored with the challenge's own metric library.
"""

import itertools
import json
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "repo", "evaluation"))
import possession_metrics as pm  # noqa: E402

from mtah import model, noise, sample  # noqa: E402

GRID_SWITCH = (4, 8, 15, 25, 40, 60)
GRID_DEAD = (4, 8, 15, 25, 40)
VAL_SEED = 99


def evaluate(gt, pred):
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"]


def main():
    games = sample.load_all()
    results = {}
    probs = []
    for k, held in enumerate(games):
        train = [g for j, g in enumerate(games) if j != k]
        clf = model.fit(train)
        test = noise.degrade(held, seed=VAL_SEED + k)
        p = model.proba(clf, test)
        probs.append((held, p))
        raw = np.where(held["period"] > 0, np.argmax(p, axis=1), model.X).astype(np.int8)
        print("{}: frame-argmax spellF1 {:.3f} macroF1 {:.3f}".format(
            held["name"], *evaluate(held["labels"], raw)), flush=True)

    for cs, cd in itertools.product(GRID_SWITCH, GRID_DEAD):
        scores = [evaluate(h["labels"], model.viterbi(p, h["period"], cs, cd))
                  for h, p in probs]
        results[(cs, cd)] = (np.mean([s[0] for s in scores]), np.mean([s[1] for s in scores]),
                             [round(s[0], 3) for s in scores])
        print("switch {:>3} dead {:>3}  spellF1 {:.3f}  macroF1 {:.3f}  per game {}".format(
            cs, cd, *results[(cs, cd)]), flush=True)

    best = max(results, key=lambda k: results[k][0])
    print("best: switch {} dead {} -> LOGO spellF1 {:.3f}".format(*best, results[best][0]))

    clf = model.fit(games)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    joblib.dump(clf, os.path.join(HERE, "models", "model.joblib"))
    with open(os.path.join(HERE, "models", "params.json"), "w") as fh:
        json.dump({"c_switch": best[0], "c_dead": best[1],
                   "logo_spell_f1": round(results[best][0], 4),
                   "logo_macro_f1": round(results[best][1], 4)}, fh, indent=2)
    print("saved models/model.joblib and models/params.json")


if __name__ == "__main__":
    main()
