"""Load the challenge match (tactical ATD) into the same "game" dict the sample games use.

Uses only the challenge repository's public conventions (games.py, DATA.md):
  * video_frame = frame_id - 1, one ATD record per video frame
  * graded halves = games.half_video_blocks(), everything else is period 0
  * home = first team in the metadata (FIFA/EPTS convention; Barcelona for this match)
  * coordinates normalised 0-1, scaled by the pitch in games.py

Nothing here prints data.
"""

import os
import sys

import numpy as np

from .epts import read_epts

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.environ.get("MTAH_REPO") or os.path.join(ROOT, "repo")
sys.path.insert(0, REPO)
import games as challenge_games  # noqa: E402


def data_dir():
    """MTAH_DATA_DIR if set, else the folder above this repository (where the bundle sits)."""
    return os.environ.get("MTAH_DATA_DIR") or ROOT


def load(game_name=challenge_games.GAME):
    meta_path, raw_path = challenge_games.require(["atd_meta", "atd_raw"], game_name, data_dir())
    n = challenge_games.source_frames("tactical", game_name)
    raw = read_epts(meta_path, raw_path)
    rows = raw["frame_id"] - 1
    if rows.min() < 0 or rows.max() >= n or len(np.unique(rows)) != len(rows):
        raise ValueError("ATD frame ids do not map one-to-one onto video frames")

    P = raw["px"].shape[1]
    px = np.full((n, P), np.nan, dtype=np.float32)
    py = np.full((n, P), np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    px[rows], py[rows], bx[rows], by[rows] = raw["px"], raw["py"], raw["bx"], raw["by"]

    period = np.zeros(n, dtype=np.int8)
    for half, first, last in challenge_games.half_video_blocks(game_name):
        period[first:last + 1] = half

    team = raw["team"]
    if not ((team == 0).any() and (team == 1).any()):
        raise ValueError("metadata does not assign track slots to both teams")

    L, W = challenge_games.GAMES[game_name]["pitch"]
    return {"name": game_name, "px": px * L, "py": py * W, "team": team, "bx": bx * L,
            "by": by * W, "period": period, "labels": None, "pitch": (L, W)}
