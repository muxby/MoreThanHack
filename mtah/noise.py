"""Degrade the clean (corrected) sample tracking so it resembles automatic, uncorrected tracking.

Why this exists: in the corrected sample data the ball is missing almost exactly when it is dead,
so a model trained on it as-is learns "no ball = dead ball". The automatic challenge feed loses the
ball on ~41% of graded frames, most of them in live play (DATA.md), and keeps seeing it during
many stoppages. The settings below describe that published behaviour in general terms -- none is
fitted to the challenge match.
"""

import numpy as np

FPS = 25


def _runs(rng, n, mean_on, mean_off, p_start_on=0.6):
    """Boolean mask of alternating on/off runs with geometric lengths (means in frames)."""
    mask = np.empty(n, dtype=bool)
    i, state = 0, rng.random() < p_start_on
    while i < n:
        length = int(rng.geometric(1.0 / (mean_on if state else mean_off)))
        mask[i:i + length] = state
        i += length
        state = not state
    return mask


def degrade(game, seed, ball_vis_s=4.0, ball_miss_s=1.5, crowd_drop=0.35,
            park_prob=0.6, player_miss=0.06, pos_sigma=0.15, ball_sigma=0.3):
    rng = np.random.default_rng(seed)
    g = dict(game)
    px, py = game["px"].copy(), game["py"].copy()
    bx, by = game["bx"].copy(), game["by"].copy()
    labels, period = game["labels"], game["period"]
    T, P = px.shape

    # 1. Dead-ball stretches: the corrected data has no ball there. The automatic tracker often
    #    sees a ball parked at the restart spot (throw-in, corner, free kick), so for some
    #    stoppages place a stationary ball where play restarts, for the tail of the stoppage.
    dead = labels == 2
    edges = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b >= T or not np.isfinite(game["bx"][b]) or rng.random() > park_prob:
            continue
        start = a + int((b - a) * rng.uniform(0.2, 0.8))
        bx[start:b] = game["bx"][b] + rng.normal(0, 0.5)
        by[start:b] = game["by"][b] + rng.normal(0, 0.5)

    # 2. Ball visibility: alternating seen/lost runs everywhere, plus extra losses when the ball
    #    is in a crowd (occlusion), which is where it matters most.
    visible = _runs(rng, T, ball_vis_s * FPS, ball_miss_s * FPS)
    d = np.hypot(px - game["bx"][:, None], py - game["by"][:, None])
    with np.errstate(invalid="ignore"):
        crowd = np.sum(d < 3.0, axis=1) >= 3
    crowd_lost = _runs(rng, T, 0.6 * FPS, 0.6 * FPS) & crowd & (rng.random() < crowd_drop)
    visible &= ~crowd_lost
    bx[~visible], by[~visible] = np.nan, np.nan
    bx += rng.normal(0, ball_sigma, T).astype(np.float32)
    by += rng.normal(0, ball_sigma, T).astype(np.float32)

    # 3. Players: positional jitter and track gaps.
    px += rng.normal(0, pos_sigma, px.shape).astype(np.float32)
    py += rng.normal(0, pos_sigma, py.shape).astype(np.float32)
    for j in range(P):
        if not np.isfinite(px[:, j]).any():
            continue
        gap = 3 * FPS  # mean track gap; on-runs sized so the lost share is `player_miss`
        lost = ~_runs(rng, T, gap * (1 - player_miss) / player_miss, gap, p_start_on=0.95)
        px[lost, j], py[lost, j] = np.nan, np.nan

    g.update(px=px, py=py, bx=bx, by=by)
    return g
