"""Generic FIFA/EPTS reader (metadata XML + raw text), used for both the Metrica sample game and the
challenge match.

The raw file is "<frame>:<group>;<group>;...:<ball group>", and the metadata's
DataFormatSpecification blocks say, per frame range, which player each group belongs to and which
channels it carries (x, y and optionally speed). Substitutions create several specifications with
different column orders, so every column is mapped to a player id rather than to a position.

Error messages here are deliberately structural (counts of columns, names of tags) and never echo
data values.
"""

import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def _strip(tag):
    return tag.split("}", 1)[-1]


def _iter(root, name):
    for element in root.iter():
        if _strip(element.tag) == name:
            yield element


class EptsMeta:
    """Teams, players and per-frame-range column layouts from the metadata XML."""

    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.team_ids = [team.get("id") for team in _iter(root, "Team")]
        if len(self.team_ids) != 2:
            raise ValueError("expected 2 teams in metadata, found {}".format(len(self.team_ids)))
        # Player order as listed; team index 0 = first team in the file = home (EPTS convention).
        self.player_ids = []
        self.player_team = []
        for player in _iter(root, "Player"):
            team_id = player.get("teamId")
            if team_id not in self.team_ids:
                continue
            self.player_ids.append(player.get("id"))
            self.player_team.append(self.team_ids.index(team_id))
        self.player_team = np.array(self.player_team, dtype=np.int8)
        index = {pid: i for i, pid in enumerate(self.player_ids)}

        channel_of = {}  # player channel id -> (player index, channel name)
        for pc in _iter(root, "PlayerChannel"):
            pid = pc.get("playerId")
            if pid in index:
                channel_of[pc.get("id")] = (index[pid], pc.get("channelId"))

        self.provider = {}
        for param in _iter(root, "ProviderParameter"):
            kids = {_strip(k.tag): (k.text or "") for k in param}
            if "Name" in kids:
                self.provider[kids["Name"]] = kids.get("Value", "")

        # Each spec: (start, end, columns) where columns is a list of (kind, player_idx, channel)
        # in the order they appear once ':' and ';' are both treated as ','.
        self.specs = []
        for spec in _iter(root, "DataFormatSpecification"):
            columns = [("frame", -1, "frame")]
            for element in spec.iter():
                tag = _strip(element.tag)
                if tag == "PlayerChannelRef":
                    ref = element.get("playerChannelId")
                    if ref not in channel_of:
                        raise ValueError("PlayerChannelRef without a known PlayerChannel")
                    player_idx, channel = channel_of[ref]
                    columns.append(("player", player_idx, channel))
                elif tag == "BallChannelRef":
                    columns.append(("ball", -1, element.get("channelId")))
            self.specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), columns))
        if not self.specs:
            raise ValueError("no DataFormatSpecification in metadata")

    @property
    def n_players(self):
        return len(self.player_ids)


def read_epts(meta_path, raw_path, n_frames=None):
    """Positions in normalised 0-1 pitch units.

    Returns dict with
      frame_id (T,), px, py (T, P) float32 with NaN for missing, bx, by (T,), team (P,)
    Rows are ordered by frame id as it appears in the raw file.
    """
    meta = EptsMeta(meta_path)
    with open(raw_path) as handle:
        lines = handle.read().splitlines()
    lines = [line for line in lines if line.strip()]
    frame_ids = np.array([int(line[:line.index(":")]) for line in lines], dtype=np.int64)

    T, P = len(lines), meta.n_players
    px = np.full((T, P), np.nan, dtype=np.float32)
    py = np.full((T, P), np.nan, dtype=np.float32)
    bx = np.full(T, np.nan, dtype=np.float32)
    by = np.full(T, np.nan, dtype=np.float32)

    covered = np.zeros(T, dtype=bool)
    for start, end, columns in meta.specs:
        rows = np.flatnonzero((frame_ids >= start) & (frame_ids <= end))
        if not len(rows):
            continue
        covered[rows] = True
        text = "\n".join(lines[i] for i in rows).replace(":", ",").replace(";", ",")
        table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                            na_values=["NaN", "nan", ""], keep_default_na=True)
        if table.shape[1] != len(columns):
            # Some exports omit trailing channels; accept extra/missing only if structure is clear.
            raise ValueError("raw file has {} columns where the metadata spec declares {}".format(
                table.shape[1], len(columns)))
        values = table.to_numpy()
        for col, (kind, player_idx, channel) in enumerate(columns):
            if kind == "player" and channel == "x":
                px[rows, player_idx] = values[:, col]
            elif kind == "player" and channel == "y":
                py[rows, player_idx] = values[:, col]
            elif kind == "ball" and channel == "x":
                bx[rows] = values[:, col]
            elif kind == "ball" and channel == "y":
                by[rows] = values[:, col]
    if not covered.all():
        raise ValueError("{} raw lines fall outside every metadata spec".format(
            int((~covered).sum())))
    return {"frame_id": frame_ids, "px": px, "py": py, "bx": bx, "by": by,
            "team": meta.player_team, "meta": meta}
