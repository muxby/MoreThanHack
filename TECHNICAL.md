# Technical document — ball possession from automatic tracking

## 1. Approach

Per-frame possession is a sequence-labelling problem, but the ranking metric (spellF1) rewards whole
possessions, not frames. A frame classifier alone flickers. On our external validation the
frame-wise argmax reaches only spellF1 ≈ 0.33 despite macroF1 ≈ 0.92. The pipeline therefore has two
stages:

1. **Frame classifier.** A gradient-boosted tree model (scikit-learn `HistGradientBoostingClassifier`)
   gives P(H), P(A), P(D) for every frame from 135 tracking features.
2. **Viterbi decoding.** A three-state path (H, A, D) maximises Σ log P − penalties, with a cost
   `c_switch` for a direct H↔A change and `c_dead` for entering or leaving a dead ball. This
   produces possessions with realistic durations instead of flicker. It lifts external spellF1 from
   ≈ 0.33 to 0.78.

The model is **team-symmetric**. Each match is also fed with the teams' roles exchanged, and the two
predictions are averaged, so neither side gets a learned "home" bias.

## 2. Features (tracking only)

Coordinates are made canonical per half: home attacks +x, with the direction read from the data
itself. All features are computed inside a half, never across halftime.

- **Ball:** seen or not; frames since last seen and until next seen; position (short gaps of
  ≤ 0.5 s linearly bridged); speed; distance to the nearest line; out-of-bounds flag.
- **Ball proximity per team:** nearest and second-nearest distance; players within 3 m and 8 m;
  the H−A difference (and its log ratio).
- **Team shape per team:** centroid depth (from its own goal), spread in x and y, defensive-line
  depth (second-deepest player), most advanced player; H−A differences of all of these. The team in
  possession stretches the pitch, and the defending team compacts and drops.
- **Movement:** mean speed per team, share of players above 4 m/s, mean forward velocity, overall
  mean speed (play stops at dead balls).
- **Context:** centred rolling means of 21 key features over ±0.5 s, ±2 s and ±5 s. Causal and
  anti-causal 1 s and 3 s means of ball proximity, shape, speed and ball visibility (transitions
  show up as before/after contrasts).

Because the ball is missing on 41% of graded frames, the team-shape and movement features carry the
possession signal through ball gaps.

## 3. Assets used and dropped

| asset | used? | why |
|---|---|---|
| tactical ATD (players + ball) | **yes** | the only input to the model |
| Smart Tagging | no | no external equivalent to learn or tune its weight from; any hand-set weight would be an untested guess |
| homography | no | everything needed is already in pitch coordinates |
| tactical / broadcast video | no | not needed; the rules forbid training or calibrating on the video |
| broadcast ATD | no | not synchronised to the tactical timeline |
| public 5-minute ground truth | **no** | never read by the pipeline; not used for training, tuning or selection |

## 4. Training data (all external)

**Metrica Sports public sample data**, games 1–3 (github.com/metrica-sports/sample-data): corrected
tracking for players and ball, plus event data, with anonymised teams. That data was labelled by
Metrica, not by us. No data about the challenge match is used anywhere in training or tuning.

**Labels are derived from those games' events by an algorithm** (`mtah/sample.py`):

- possession = team of the latest on-ball event (pass, carry, recovery, shot, set piece, ball lost,
  ball out). Challenges, cards and "fault received" do not change possession.
- dead ball from a ball-out, a shot out or a goal, an offside or hand-ball (at the event's end
  frame), or a foul (at its frame), until the next on-ball event (the restart). Every restart
  (set piece or goal kick) implies a stoppage that began when the previous on-ball action ended.
- **Controlled-possession rule**, taken from the wording of CHALLENGE.md: a spell shorter than
  3 s ("keeps the ball a few seconds") with fewer than 3 passes ("more than two consecutive passes")
  is not a change of possession. It returns to the previous team, unless it follows a restart.

The resulting mean possession is 14–15 s per sample game. No constant here was fitted to the
challenge ground truth.

**Simulated automatic-tracking noise** (`mtah/noise.py`). In the corrected sample data the ball is
missing almost exactly when it is dead (≈ 99%), and a model would learn "no ball = dead ball". The
automatic feed instead loses the ball during play and often sees it parked at restarts. The training
copies therefore get:

- alternating visible/lost runs of the ball (means 4.0 s / 1.5 s), plus extra losses in crowds of 3+
  players within 3 m;
- for 60% of stoppages, a stationary ball parked at the restart spot for the tail of the stoppage;
- position jitter (0.15 m players, 0.3 m ball) and player track gaps (6% of frames, mean 3 s).

These settings are hand-set from the general descriptions in DATA.md, not fitted. Every game is used
in two orientations and with two noise seeds.

## 5. Tuning

The only tuned parameters are the two Viterbi penalties. They were chosen by **leave-one-game-out**
cross-validation over the three sample games, scored with the challenge's own metric library
(`possession_metrics.evaluate`, spellF1). The models were trained on two games and evaluated on the
held-out game with a fresh noise seed.

| | spellF1 | macroF1 |
|---|---|---|
| frame argmax (no decoding) | 0.30–0.35 | 0.91–0.93 |
| **Viterbi, c_switch = 40, c_dead = 25** (selected) | **0.780** (0.765 / 0.767 / 0.809) | 0.917 |

The surface is flat near the optimum (0.77–0.78 for `c_switch` 25–60 and `c_dead` 8–40), so the
choice is not fragile. The final model is retrained on all three games.

## 6. Possession rules as implemented

- **Loose balls and challenges:** possession stays with the last team until the opponent controls
  the ball for the spell lengths above. Brief deflections are absorbed by the decoding penalty.
- **Restarts:** a possession that begins after a dead ball is always a possession, as CHALLENGE.md
  states that restarts have a clear possessor.
- **Dead ball:** learnt from ball-out-of-bounds, a parked ball, players slowing and ball visibility,
  and decoded as its own state. Dead-ball ownership is not modelled, as instructed.
- **Ungraded frames** (before kickoff, halftime, after the final whistle) are written as `X`.

## 7. Limitations

- **Domain gap:** the external labels come from event data, while the challenge ground truth comes
  from corrected tracking ownership. Definitions of when a possession starts can differ by a fraction
  of a second, and spellF1 is sensitive to that at high IoU thresholds.
- **Noise model:** simulated noise is simpler than real automatic tracking. It does not model
  identity switches between teams, duplicated players or false ball detections, so the external
  0.78 is an optimistic upper bound for the challenge match.
- **Three training games:** a small and anonymised sample. More external games (for example the
  DFL/IDSSE open data) would be the first improvement.
- **Smart Tagging is unused.** It carries strong team evidence, but there is no external data to
  calibrate how much to trust it.

## 8. Reproduction

See [README.md](README.md). `python train.py` followed by `python predict.py --name <team>`
regenerates the submission byte for byte: all randomness is seeded and the classifier is
deterministic.
