"""Metrica public sample games (github.com/metrica-sports/sample-data) -- the EXTERNAL training and
tuning data. Nothing in this module touches the challenge match.

Per-frame H/A/D labels are derived mechanically from the sample games' event data:

  * possession belongs to the team of the latest on-ball event (pass, carry, recovery, shot,
    set piece, ball lost, ball out). Challenges, cards and "fault received" are not on-ball events:
    a won duel without a following on-ball action does not change possession, as CHALLENGE.md asks.
  * the ball is dead from a ball-out / shot-out / goal / offside / hand-ball (at the event's end
    frame) or from a foul (at its frame) until the next on-ball event, which is the restart.
  * frames before the first and after the last event of a period are ungraded (X).
"""

import json
import os

import numpy as np
import pandas as pd

from .epts import read_epts

H, A, D, X = 0, 1, 2, 3
PITCH = (105.0, 68.0)

ROOT = os.environ.get("MTAH_SAMPLE_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "external", "sample-data", "data")

NOT_ON_BALL = {"CHALLENGE", "CARD", "FAULT RECEIVED"}
DEAD_AFTER_WORDS = ("OUT", "GOAL", "OFFSIDE", "HAND BALL", "END HALF")


def _dead_after(etype, subtype):
    """Does this on-ball event end with the ball dead (at its end frame)?"""
    if etype == "BALL OUT":
        return True
    if etype in ("SHOT", "BALL LOST", "PASS", "RECOVERY"):
        words = subtype.split("-")
        return any(w in words for w in DEAD_AFTER_WORDS)
    return False


def _foul(etype, subtype):
    if etype == "FAULT RECEIVED":
        return True
    if etype == "CHALLENGE":
        words = subtype.split("-")
        return "FAULT" in words and "ADVANTAGE" not in words
    return False


def event_labels(events, period, n_rows, row_of_frame):
    """events: list of dicts with start, end (frame ids), team (0/1), type, subtype, period."""
    labels = np.full(n_rows, X, dtype=np.int8)
    for p in sorted(set(period[period > 0].tolist())):
        evs = sorted((e for e in events if e["period"] == p), key=lambda e: (e["start"], e["i"]))
        if not evs:
            continue
        changes = []  # (frame id, label)
        pending_dead = None
        prev_end = None  # latest end frame of any on-ball event so far
        last_end = max(e["end"] for e in evs)
        for e in evs:
            et, st = e["type"], e["subtype"]
            if et in NOT_ON_BALL:
                if _foul(et, st):
                    pending_dead = e["start"] if pending_dead is None else min(pending_dead,
                                                                               e["start"])
                continue
            # Every restart implies a stoppage before it: if nothing has flagged one yet, the
            # ball went dead when the previous on-ball action ended.
            restart = et == "SET PIECE" or (et == "PASS" and "GOAL KICK" in st.split("-"))
            if restart and pending_dead is None and prev_end is not None \
                    and prev_end < e["start"]:
                pending_dead = prev_end
            # A dead ball takes effect only once an event starts at or after it; events that
            # start earlier (e.g. a zero-length carry listed after the BALL OUT) leave it pending.
            if pending_dead is not None and pending_dead <= e["start"]:
                changes.append((pending_dead, D))
                pending_dead = None
            changes.append((e["start"], e["team"]))
            if _dead_after(et, st):
                end = max(e["end"], e["start"])
                pending_dead = end if pending_dead is None else min(pending_dead, end)
            prev_end = max(prev_end or e["end"], e["end"], e["start"])
        if pending_dead is not None:
            changes.append((pending_dead, D))
        first = evs[0]["start"]
        rows = np.flatnonzero(period == p)
        # Fill forward from each change point. sort is stable, so for equal frames the later
        # event wins.
        changes.sort(key=lambda c: c[0])
        change_frames = np.array([c[0] for c in changes])
        vals = np.array([c[1] for c in changes], dtype=np.int8)
        frames = row_of_frame(rows)
        idx = np.searchsorted(change_frames, frames, side="right") - 1
        lab = np.where(idx >= 0, vals[np.clip(idx, 0, None)], X)
        lab[(frames < first) | (frames > last_end)] = X
        labels[rows] = lab
    return labels


def _game(px, py, team, bx, by, period, labels, name):
    L, W = PITCH
    return {"name": name, "px": px * L, "py": py * W, "team": team, "bx": bx * L, "by": by * W,
            "period": period, "labels": labels, "pitch": PITCH}


def load_csv_game(n):
    folder = os.path.join(ROOT, "Sample_Game_{}".format(n))
    parts, teams = [], []
    ball = period = frames = None
    for side, t in (("Home", 0), ("Away", 1)):
        df = pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(
            n, side)), skiprows=2)
        cols = list(df.columns)
        bcol = cols.index("Ball")
        xy = df.iloc[:, 3:bcol].to_numpy(dtype=np.float32)
        parts.append(xy)
        teams += [t] * (xy.shape[1] // 2)
        ball = df.iloc[:, bcol:bcol + 2].to_numpy(dtype=np.float32)
        period = df["Period"].to_numpy()
        frames = df["Frame"].to_numpy()
    xy = np.concatenate(parts, axis=1)
    px, py = xy[:, 0::2], xy[:, 1::2]
    period = np.where(np.isin(period, (1, 2)), period, 0).astype(np.int8)

    ev = pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(n)))
    events = [{"i": i, "start": int(r["Start Frame"]), "end": int(r["End Frame"]),
               "team": 0 if r["Team"] == "Home" else 1, "type": str(r["Type"]).upper(),
               "subtype": "" if pd.isna(r["Subtype"]) else str(r["Subtype"]).upper(),
               "period": int(r["Period"])} for i, r in ev.iterrows()]
    row_frames = frames
    labels = event_labels(events, period, len(frames), lambda rows: row_frames[rows])
    return _game(px, py, np.array(teams, dtype=np.int8), ball[:, 0], ball[:, 1], period, labels,
                 "sample{}".format(n))


def load_epts_game(n=3):
    folder = os.path.join(ROOT, "Sample_Game_{}".format(n))
    raw = read_epts(os.path.join(folder, "Sample_Game_{}_metadata.xml".format(n)),
                    os.path.join(folder, "Sample_Game_{}_tracking.txt".format(n)))
    meta, fid = raw["meta"], raw["frame_id"]
    period = np.zeros(len(fid), dtype=np.int8)
    for p, (a, b) in enumerate((("first_half_start", "first_half_end"),
                                ("second_half_start", "second_half_end")), start=1):
        if meta.provider.get(a) and meta.provider.get(b):
            period[(fid >= int(meta.provider[a])) & (fid <= int(meta.provider[b]))] = p

    data = json.load(open(os.path.join(folder, "Sample_Game_{}_events.json".format(n))))["data"]
    events = []
    for i, e in enumerate(data):
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = sub["name"]
        elif isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub)
        else:
            sub = ""
        events.append({"i": i, "start": int(e["start"]["frame"]),
                       "end": int(e["end"]["frame"] or e["start"]["frame"]),
                       "team": meta.team_ids.index(e["team"]["id"]),
                       "type": e["type"]["name"].upper(), "subtype": sub.upper(),
                       "period": int(e["period"])})
    labels = event_labels(events, period, len(fid), lambda rows: fid[rows])
    return _game(raw["px"], raw["py"], raw["team"], raw["bx"], raw["by"], period, labels,
                 "sample{}".format(n))


def load_all():
    return [load_csv_game(1), load_csv_game(2), load_epts_game(3)]
