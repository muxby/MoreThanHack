"""Per-frame features from tracking only, identical for the sample games and the challenge match.

Input is a "game" dict in metres: px, py (T, P) with NaN for missing, team (P,) 0 = home / 1 = away,
bx, by (T,), period (T,) with 0 outside play. Every feature is computed within a period and never
across one.

Coordinates are first made canonical per period: home attacks towards +x. The attack direction is
read from the data itself (the team whose players sit nearer x = 0 on average defends that goal).
"""

import numpy as np
import pandas as pd

FPS = 25
BALL_INTERP_MAX = 12        # frames of missing ball bridged by linear interpolation (~0.5 s)
SPEED_CAP = 12.0            # m/s; faster is a track jump / identity switch, not running
CONTEXT = (12, 50, 125)     # half-widths of centred context windows, frames
OUT_MARGIN = 0.5            # metres beyond a line before the ball counts as out


def canonical(game):
    """Copy with x/y flipped per period so that home (team 0) always attacks +x."""
    L, W = game["pitch"]
    px, py = game["px"].copy(), game["py"].copy()
    bx, by = game["bx"].copy(), game["by"].copy()
    team = game["team"]
    for p in (1, 2):
        rows = game["period"] == p
        if not rows.any():
            continue
        home_x = np.nanmean(px[rows][:, team == 0])
        away_x = np.nanmean(px[rows][:, team == 1])
        if home_x > away_x:
            px[rows], py[rows] = L - px[rows], W - py[rows]
            bx[rows], by[rows] = L - bx[rows], W - by[rows]
    out = dict(game)
    out.update(px=px, py=py, bx=bx, by=by)
    return out


def swapped(game):
    """The same match with the teams' roles exchanged: used for symmetric training and inference.

    The labels swap H <-> A. No explicit mirroring is needed: canonical() re-reads the attack
    direction from the data, so the new home team attacks +x after it.
    """
    out = dict(game)
    out["team"] = (1 - game["team"]).astype(np.int8)
    if game.get("labels") is not None:
        lab = game["labels"].copy()
        lab[game["labels"] == 0], lab[game["labels"] == 1] = 1, 0
        out["labels"] = lab
    return out


def _interp_ball(b, period, max_gap):
    """Linear interpolation of short gaps, per period."""
    out = b.copy()
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        if not len(rows):
            continue
        s = pd.Series(b[rows])
        out[rows] = s.interpolate(limit=max_gap, limit_area="inside").to_numpy()
    return out


def _since_seen(seen, period, cap):
    """Frames since the ball was last seen (forward) and until it is next seen (backward)."""
    since = np.full(len(seen), cap, dtype=np.float32)
    until = np.full(len(seen), cap, dtype=np.float32)
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        if not len(rows):
            continue
        s = seen[rows]
        idx = np.arange(len(s))
        last = np.maximum.accumulate(np.where(s, idx, -10 ** 9))
        nxt = np.minimum.accumulate(np.where(s, idx, 10 ** 9)[::-1])[::-1]
        since[rows] = np.minimum(idx - last, cap)
        until[rows] = np.minimum(nxt - idx, cap)
    return since, until


def _speeds(px, py, period):
    """Per-player speed in m/s from position differences, NaN at gaps, capped."""
    vx = np.diff(px, axis=0, prepend=np.nan) * FPS
    vy = np.diff(py, axis=0, prepend=np.nan) * FPS
    same = np.r_[False, period[1:] == period[:-1]]
    sp = np.hypot(vx, vy)
    sp[~same] = np.nan
    sp[sp > SPEED_CAP] = np.nan
    return sp, vx


def base_features(game):
    g = canonical(game)
    L, W = g["pitch"]
    px, py, team, period = g["px"], g["py"], g["team"], g["period"]
    T = len(period)
    seen = np.isfinite(g["bx"]) & np.isfinite(g["by"])
    bx = _interp_ball(g["bx"], period, BALL_INTERP_MAX)
    by = _interp_ball(g["by"], period, BALL_INTERP_MAX)
    have = np.isfinite(bx) & np.isfinite(by)

    f = {}
    f["ball_seen"] = seen.astype(np.float32)
    f["ball_have"] = have.astype(np.float32)
    f["since_seen"], f["until_seen"] = _since_seen(seen, period, 250)
    f["bx"], f["by"] = bx, by
    f["ball_touch_dist"] = np.minimum(np.minimum(by, W - by), np.minimum(bx, L - bx))
    f["ball_out"] = ((bx < -OUT_MARGIN) | (bx > L + OUT_MARGIN) | (by < -OUT_MARGIN)
                     | (by > W + OUT_MARGIN)).astype(np.float32)
    f["ball_out"][~have] = np.nan
    bvx = np.diff(bx, prepend=np.nan) * FPS
    bvy = np.diff(by, prepend=np.nan) * FPS
    bsp = np.hypot(bvx, bvy)
    bsp[bsp > 45] = np.nan
    f["ball_speed"] = pd.Series(bsp).rolling(5, center=True, min_periods=1).median().to_numpy()

    sp, vx = _speeds(px, py, period)
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    tracked = np.isfinite(px)

    with np.errstate(all="ignore"), _quiet():
        for k, name in ((0, "h"), (1, "a")):
            m = team == k
            d = dist[:, m]
            f["d_" + name] = np.nanmin(d, axis=1)
            ds = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
            f["d2_" + name] = np.where(np.isfinite(ds[:, 1]), ds[:, 1], np.nan) \
                if ds.shape[1] > 1 else np.nan
            f["n3_" + name] = np.sum(d < 3, axis=1).astype(np.float32)
            f["n8_" + name] = np.sum(d < 8, axis=1).astype(np.float32)
            f["n_" + name] = tracked[:, m].sum(axis=1).astype(np.float32)
            x, y = px[:, m], py[:, m]
            # attack direction: home +x, away -x -> express x as "distance from own goal"
            xa = x if k == 0 else L - x
            f["cx_" + name] = np.nanmean(xa, axis=1)
            f["cy_" + name] = np.nanmean(y, axis=1)
            f["sx_" + name] = np.nanstd(x, axis=1)
            f["sy_" + name] = np.nanstd(y, axis=1)
            xs = np.sort(np.where(np.isfinite(xa), xa, np.inf), axis=1)
            f["line_" + name] = np.where(np.isfinite(xs[:, 1]), xs[:, 1], np.nan)  # 2nd deepest
            xs2 = np.sort(np.where(np.isfinite(xa), xa, -np.inf), axis=1)
            f["front_" + name] = np.where(np.isfinite(xs2[:, -1]), xs2[:, -1], np.nan)
            f["v_" + name] = np.nanmean(sp[:, m], axis=1)
            f["vfast_" + name] = np.nanmean(sp[:, m] > 4.0, axis=1)
            fwd = vx[:, m] if k == 0 else -vx[:, m]
            fwd = np.where(np.abs(fwd) > SPEED_CAP, np.nan, fwd)
            f["vfwd_" + name] = np.nanmean(fwd, axis=1)
        f["v_all"] = np.nanmean(sp, axis=1)
        f["n_all"] = tracked.sum(axis=1).astype(np.float32)

    # Ball relative to each team's goal and shape, and pairwise contrasts.
    f["bx_att_h"] = bx                      # distance from home goal
    f["dd"] = f["d_h"] - f["d_a"]
    f["dlog"] = np.log1p(f["d_h"]) - np.log1p(f["d_a"])
    f["near_h"] = np.where(np.isfinite(f["dd"]), (f["dd"] < 0).astype(np.float32), np.nan)
    f["dcx"] = f["cx_h"] - f["cx_a"]        # home further from own goal = pushing up
    f["dsx"] = f["sx_h"] - f["sx_a"]
    f["dsy"] = f["sy_h"] - f["sy_a"]
    f["dline"] = f["line_h"] - f["line_a"]
    f["dfront"] = f["front_h"] - f["front_a"]
    f["dv"] = f["v_h"] - f["v_a"]
    f["dvfwd"] = f["vfwd_h"] - f["vfwd_a"]
    f["ball_rel_h"] = bx - (f["cx_h"])
    f["ball_rel_a"] = (L - bx) - f["cx_a"]
    return pd.DataFrame(f).astype(np.float32)


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *a):
        self._w.__exit__(*a)


CONTEXT_COLS = ("ball_seen", "near_h", "dd", "dlog", "dcx", "dsx", "dsy", "dline", "dfront", "dv",
                "dvfwd", "v_all", "ball_speed", "ball_out", "bx", "n3_h", "n3_a", "vfast_h",
                "vfast_a", "n_all", "ball_touch_dist")


def _per_period(values, period, fn):
    """Apply fn (DataFrame -> DataFrame) to each period block of a 2-D float array."""
    out = np.full(values.shape, np.nan, dtype=np.float32)
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        if len(rows):
            out[rows] = fn(pd.DataFrame(values[rows])).to_numpy(dtype=np.float32)
    return out


def features(game):
    """Base features plus centred rolling means at several scales, per period."""
    base = base_features(game)
    period = game["period"]
    ctx = base[list(CONTEXT_COLS)].to_numpy(dtype=np.float32)
    parts = [base]
    for w in CONTEXT:
        r = _per_period(ctx, period,
                        lambda df: df.rolling(2 * w + 1, center=True, min_periods=1).mean())
        parts.append(pd.DataFrame(r, columns=[c + "_m%d" % w for c in CONTEXT_COLS]))
    # Asymmetric context: just before vs just after a frame -- transitions show up here.
    cols = ("near_h", "dd", "dcx", "v_all", "ball_seen")
    vals = base[list(cols)].to_numpy(dtype=np.float32)
    for w in (25, 75):
        before = _per_period(vals, period, lambda df: df.rolling(w, min_periods=1).mean())
        after = _per_period(vals, period,
                            lambda df: df[::-1].rolling(w, min_periods=1).mean()[::-1])
        parts.append(pd.DataFrame(before, columns=[c + "_b%d" % w for c in cols]))
        parts.append(pd.DataFrame(after, columns=[c + "_a%d" % w for c in cols]))
    out = pd.concat(parts, axis=1)
    out[period == 0] = np.nan
    return out
