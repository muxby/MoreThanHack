# Ball possession from automatic tracking — More than a Hack 2027, Phase 1

Per-frame `H` / `A` / `D` labels for the BAR–ATH match from the automatic tracking data (ATD) alone.
A gradient-boosted frame classifier is trained **only on external data**: Metrica's three public sample
games, artificially degraded to look like automatic tracking. Viterbi decoding then turns the frame
probabilities into whole possessions.

**Nothing is trained, fitted, tuned or calibrated on the challenge match, its video or its ground truth.**
The public 5-minute ground truth is never read by this code.

## Reproduce

```bash
pip install -r requirements.txt

# 1. external training data (public, Metrica Sports)
git clone https://github.com/metrica-sports/sample-data ../external/sample-data

# 2. the challenge repository (for games.py and the official metric library)
git clone https://github.com/metrica-sports/more-than-a-hack-2027 ../repo

# 3. train + tune on the sample games  -> models/model.joblib, models/params.json
python train.py

# 4. predict the challenge match        -> submissions/<team>.csv
export MTAH_DATA_DIR=/path/to/bundle    # folder containing tactical/BAR-ATH_tactical_atd.{xml,txt}
python predict.py --name <team>
```

Expected layout (defaults, overridable with `MTAH_DATA_DIR`, `MTAH_REPO`, `MTAH_SAMPLE_DIR`):

```
More Than Hack/
├─ tactical/            match bundle (NOT in this repository)
├─ repo/                challenge repository
├─ external/sample-data Metrica public sample data
└─ solution/            this repository
```

The match data is licensed for the competition only and is excluded by `.gitignore`.

## Method

See [TECHNICAL.md](TECHNICAL.md).
