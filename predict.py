"""Produce the possession file for the challenge match from the model trained by train.py.

    python predict.py --name <team_name>       # writes submissions/<team_name>.csv

Deterministic: same data + same models/ -> byte-identical file. Prints status only.
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from mtah import challenge, model  # noqa: E402

sys.path.insert(0, os.path.join(challenge.REPO, "evaluation"))
import possession_metrics as pm  # noqa: E402

LETTERS = np.array(["H", "A", "D", "X"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="submission", help="team name -> <name>.csv")
    parser.add_argument("--out", default=os.path.join(HERE, "submissions"))
    args = parser.parse_args()

    clf = joblib.load(os.path.join(HERE, "models", "model.joblib"))
    with open(os.path.join(HERE, "models", "params.json")) as fh:
        params = json.load(fh)

    print("loading match tracking ...", flush=True)
    game = challenge.load()
    print("computing frame probabilities ...", flush=True)
    p = model.proba(clf, game)
    print("decoding possessions ...", flush=True)
    codes = model.viterbi(p, game["period"], params["c_switch"], params["c_dead"])

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.name + ".csv")
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, c) for i, c in enumerate(LETTERS[codes]))

    reason = pm.validate_submission(pm.read_label_csv(path),
                                    challenge.challenge_games.source_frames("tactical"))
    print("wrote", os.path.relpath(path, HERE))
    print("validation:", "OK" if reason is None else "FAILED - " + reason)


if __name__ == "__main__":
    main()
