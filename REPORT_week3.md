# Week 3 report: adding two action vectors, scored on the 127 test doubles

**One run, 11 November 2026.** Every Week 2 action set was scored once on the test doubles, in the frozen state
spaces, with no decoder. The list of action sets, references and scores was fixed in `src/pertact/doubles.py` before
the run. Nothing was chosen or tuned from these numbers.

**Short version.**
* Adding two mean shifts lands close to the real double: the median miss is 14% (PCA-50) and 17% (scVI-30) of the
  squared real shift.
* No InfoNCE variant lands closer. The best are 25% (strong anchor, PCA-50) and 44% (per-action scaling, scVI-30).
* InfoNCE as trained is worse than adding no vector for 81% (PCA-50) and 95% (scVI-30) of the doubles.
* One score favours InfoNCE: naming the double from a single cell in PCA-50 (top-1 0.19 for the rescaled sets
  against 0.165 for the mean shift). In scVI-30 the two tie.
* Next to the Week 1 embeddings (section 2.5, PCA-50, not like for like): one-hot and GenePT through their trained
  decoders land about as close as the summed mean shifts (0.12–0.14). The InfoNCE vectors in that table were added
  directly and have not been through a trained decoder.

## 1. What was run

**Prediction.** For a double a+b: start + f(a) + f(b), in PCA-50 and in scVI-30. Each gene has one vector, made
from training pairs only.

**Action sets** (the same nine as the Week 2 tests, from `retrieval.build_action_sets`):

| set | what it is |
|---|---|
| mean shift | average OT displacement on training pairs (closed form) |
| InfoNCE as trained | 3 seeds |
| InfoNCE, global α | all vectors × one scalar fitted on training pairs |
| InfoNCE, per action | one scalar per gene fitted on training pairs |
| InfoNCE direction at mean-shift length | control: only the direction is InfoNCE's |
| InfoNCE with the ρ = 1 anchor | trained with the squared-error term |
| mean-shift direction at InfoNCE length | control: only the length is InfoNCE's |
| shuffled mean shift | mean shifts permuted across genes |
| no vector | f = 0 |

**Scores.**
* **The centre.** The real shift of a double is the centre of its test cells minus the centre of the test controls,
  taken per gem group and averaged with the double's cell counts. The predicted shift f(a) + f(b) is compared with it:
  * *miss ratio*: squared miss divided by the squared length of the real shift. 0 is perfect, 1 is what adding no
    vector scores;
  * *cosine*: direction only;
  * *length ratio*: predicted length divided by real length.
* **One vector instead of two.** The same scores for the gene with the longer training mean shift alone, and for
  whichever of the two genes lands closer (this second reference looks at the double, so it is an upper bound for
  using one vector).
* **Naming the double.** Each of the 33,966 test cells starts from the mean test control of its gem group. Every
  candidate combination is added to that start, and the true one is ranked by how close it lands to the cell.
  Candidates are the 127 measured doubles (chance top-1 = 0.008), or every combination of the 105 genes: no
  perturbation, 105 singles and 5,460 pairs (5,566 candidates, chance 0.0002). The same ranking is also done once per
  double from its centre.

**Protocol.**
* All 127 doubles are test data and none was set aside for development (decision of 11 November; a split may be
  made later).
* The code was checked without touching a double: on synthetic cells (`tests/test_doubles.py`), and on validation
  singles, where naming the perturbation from the gem-mean start is the Week 2 reverse test. All 126 values agree
  with `results/week2_retrieval.csv` (largest difference 1e-16).
* Then one run on the doubles: `python -I -m pertact.doubles`, 45 seconds on the laptop.

## 2. Results

Values for InfoNCE sets are means over 3 seeds. Seed sd of the miss ratio is at most 0.03 for the rescaled and
anchored sets, and up to 0.16 for the two sets at InfoNCE's trained length (scVI-30); seed sd of the cosine is at
most 0.012.

### 2.1 The centre

**PCA-50**

| action set | median miss ratio | better than no vector | median cosine | median length ratio | beats the mean-shift sum |
|---|---|---|---|---|---|
| **mean shift** | **0.139** | 94% | **0.959** | 1.08 | n/a |
| InfoNCE as trained | 1.93 | 19% | 0.728 | 1.93 | 0% |
| InfoNCE, global α | 0.525 | 86% | 0.728 | 0.83 | 14% |
| InfoNCE, per action | 0.483 | 93% | 0.770 | 0.86 | 11% |
| InfoNCE direction at mean-shift length | 0.549 | 86% | 0.766 | 1.05 | 2% |
| InfoNCE with the ρ = 1 anchor | 0.253 | 89% | 0.927 | 1.20 | 6% |
| mean-shift direction at InfoNCE length | 1.34 | 34% | 0.946 | 2.06 | 1% |
| shuffled mean shift | 1.00 | 50% | 0.560 | 1.08 | 2% |
| no vector | 1 | n/a | n/a | 0 | 6% |

**scVI-30**

| action set | median miss ratio | better than no vector | median cosine | median length ratio | beats the mean-shift sum |
|---|---|---|---|---|---|
| **mean shift** | **0.169** | 98% | **0.942** | 1.11 | n/a |
| InfoNCE as trained | 2.70 | 5% | 0.761 | 2.33 | 0% |
| InfoNCE, global α | 0.468 | 86% | 0.761 | 0.98 | 20% |
| InfoNCE, per action | 0.438 | 98% | 0.815 | 0.98 | 13% |
| InfoNCE direction at mean-shift length | 0.491 | 93% | 0.808 | 1.07 | 7% |
| InfoNCE with the ρ = 1 anchor | 0.460 | 86% | 0.902 | 1.39 | 1% |
| mean-shift direction at InfoNCE length | 2.50 | 10% | 0.924 | 2.46 | 0% |
| shuffled mean shift | 1.40 | 30% | 0.332 | 1.11 | 0% |
| no vector | 1 | n/a | n/a | 0 | 2% |

"Better than no vector" and "beats the mean-shift sum" are shares of the 127 doubles.

* **The sum of two mean shifts is the closest.** It removes a median 86% (PCA-50) and 83% (scVI-30) of the squared
  shift and points along the real shift (cosine 0.96 / 0.94).
* **It is a little too long.** The sum is longer than the real shift for 61% / 68% of the doubles (median 1.08× /
  1.11×). On validation singles the same vectors have a length ratio of 0.99 / 0.97, so the extra length comes from
  adding, not from the vectors.
* **InfoNCE's gap is direction, not only length.** At exactly the mean shift's lengths, InfoNCE's directions miss by
  0.55 / 0.49 against 0.14 / 0.17. Their sums have a cosine of 0.73–0.82 against 0.94–0.96.
* **InfoNCE as trained makes the prediction worse than adding nothing** for 81% / 95% of the doubles. Each vector is
  about 2× too long and the sum is 1.9× / 2.3× too long.
* **The anchored set is the closest InfoNCE set in PCA-50** (0.253). It is also the set whose vectors are nearly the
  mean shift (cosine 0.97 to it, Week 2 section 5.3). In scVI-30 its remaining overshoot (1.39×) costs more.
* **Reading the cosines in PCA-50:** shuffled mean shifts already reach 0.56 there, because the mean shifts share one
  large common direction (Week 2 section 5.4). In scVI-30 the shuffled value is 0.33.
* **Noise.** Sampling noise in the two centres explains a median 0.9% / 1.2% of the squared real shift, so it is a
  small part of the mean shift's 14% / 17%.

### 2.2 Does the second vector help?

Median miss ratio of the sum, of the stronger gene's vector alone, and of the better of the two alone:

| action set | PCA-50: sum / stronger / better | sum beats the stronger gene | scVI-30: sum / stronger / better | sum beats the stronger gene |
|---|---|---|---|---|
| mean shift | 0.139 / 0.269 / 0.267 | 72% | 0.169 / 0.283 / 0.280 | 76% |
| InfoNCE, per action | 0.483 / 0.573 / 0.556 | 70% | 0.438 / 0.504 / 0.498 | 66% |
| InfoNCE with the ρ = 1 anchor | 0.253 / 0.299 / 0.290 | 56% | 0.460 / 0.324 / 0.316 | 37% |
| InfoNCE as trained | n/a | 4% | n/a | 13% |

* **For the mean shift, adding the second vector halves the miss.** That is the composition claim holding in the
  state space.
* **For the anchored set in scVI-30 the second vector hurts**, because each vector is already too long.

### 2.3 Naming the double

**From one cell** (33,966 test cells; top-1 / top-5):

| action set | PCA-50, 127 doubles | scVI-30, 127 doubles | PCA-50, all 5,566 combinations (top-1) | scVI-30, all 5,566 (top-1) |
|---|---|---|---|---|
| mean shift | 0.165 / 0.390 | **0.212** / 0.501 | 0.039 | 0.035 |
| InfoNCE as trained | 0.132 / 0.373 | 0.112 / 0.331 | 0.007 | 0.014 |
| InfoNCE, global α | **0.196 / 0.461** | 0.201 / 0.482 | 0.040 | **0.046** |
| InfoNCE, per action | 0.192 / 0.439 | 0.207 / 0.496 | 0.041 | 0.036 |
| InfoNCE direction at mean-shift length | 0.194 / 0.444 | 0.208 / **0.504** | 0.034 | 0.034 |
| InfoNCE with the ρ = 1 anchor | 0.166 / 0.396 | 0.179 / 0.457 | 0.031 | 0.022 |
| mean-shift direction at InfoNCE length | 0.119 / 0.301 | 0.106 / 0.311 | 0.009 | 0.007 |
| shuffled mean shift | 0.007 / 0.038 | 0.006 / 0.044 | 0.000 | 0.000 |

Seed sd is at most 0.006 (127 doubles) and 0.003 (all combinations).

**From the double's centre** (127 queries; top-1):

| action set | PCA-50, 127 doubles | scVI-30, 127 doubles | PCA-50, all 5,566 | scVI-30, all 5,566 |
|---|---|---|---|---|
| mean shift | 0.528 | **0.606** | **0.205** | **0.205** |
| InfoNCE as trained | 0.173 | 0.123 | 0.000 | 0.000 |
| InfoNCE, global α | 0.407 | 0.467 | 0.094 | 0.157 |
| InfoNCE, per action | 0.478 | 0.538 | 0.139 | 0.121 |
| InfoNCE direction at mean-shift length | 0.451 | 0.559 | 0.134 | 0.100 |
| InfoNCE with the ρ = 1 anchor | **0.530** | 0.438 | 0.131 | 0.068 |

Seed sd is up to 0.04 here, because there are only 127 queries.

* **From one cell in PCA-50, the rescaled InfoNCE sets name the double more often than the mean shift** (0.19–0.20
  against 0.165; the per-action set is ahead for 78 of the 127 doubles). In scVI-30 the two tie (0.207–0.208 against
  0.212).
* **This repeats the pattern on validation singles** (Week 2 section 5.3, gem-mean start): 0.257 against 0.216 in
  PCA-50, 0.261 against 0.264 in scVI-30.
* **From the centre, the mean shift is ahead or level everywhere.** A possible reason for the difference, not tested:
  InfoNCE's vectors are more spread out (pairwise cosine 0.14 against 0.49 in PCA-50), which keeps candidates apart
  when the query is one noisy cell, while the centre rewards landing in the right place.
* **Among all 5,566 combinations, from one cell,** the mean shift and the rescaled sets are level in PCA-50 (0.039
  against 0.040–0.041); the global-α set is ahead in scVI-30 (0.046 against 0.035).

### 2.4 Where the mean-shift sum misses

A secondary view: the doubles grouped by the effect strength of their two genes (the Week 2 thirds, per space).

| the two genes | PCA-50: n / miss ratio / cosine / length ratio | scVI-30: n / miss ratio / cosine / length ratio |
|---|---|---|
| medium + medium | 36 / 0.097 / 0.959 / 0.93 | 21 / 0.154 / 0.930 / 1.02 |
| medium + strong | 37 / 0.124 / 0.965 / 1.13 | 31 / 0.167 / 0.951 / 1.27 |
| medium + weak | 20 / 0.148 / 0.952 / 0.99 | 30 / 0.144 / 0.936 / 0.94 |
| strong + strong | 13 / 0.252 / 0.975 / **1.42** | 21 / 0.322 / 0.958 / **1.43** |
| strong + weak | 14 / 0.149 / 0.957 / 1.02 | 17 / 0.123 / 0.959 / 1.07 |
| weak + weak | 7 / 0.436 / 0.770 / 0.89 | 7 / 0.468 / 0.760 / 0.98 |

* **Two strong genes together move cells less than their sum.** The direction is right (cosine 0.96–0.98) and the
  sum is 1.4× too long.
* **Two weak genes** give a small real shift, so noise takes a large share (only 7 doubles).
* **With the best possible length for each double,** the mean-shift sum's median miss would still be 0.08 (PCA-50)
  and 0.11 (scVI-30): that part is direction.
* The mean-shift sum is worse than no vector for 8 doubles in PCA-50 (four of them contain CEBPE) and 3 in scVI-30;
  DUSP9+MAPK1 and OSR2+PTPN12 are in both lists.

### 2.5 Next to the Week 1 embeddings (PCA-50 only, not like for like)

**Read this first.** The rows marked *trained decoder* are a Week 1 embedding plus a decoder trained on singles. The
rows marked *added directly* are bare vectors summed in the state space: the mean shift and every InfoNCE set. **No
Week 2 action vector in this table has been through a trained decoder on the doubles.** The like-for-like test, the
same vectors as decoder inputs scored on the doubles, has not been run (section 4).

**How.** Week 1 saved each decoder's predicted mean expression profile for every test double. PCA-50 is a linear map
of expression, so each profile was projected into PCA-50 and scored as in 2.1 and 2.3. scVI-30 is not a linear map
of expression and is left out. No model was run and no action set was scored again
(`python -I scripts/week3_week1_in_state_space.py`).

127 test doubles, PCA-50. Week 1 rows are latent additive / decoder-only, means of 3 seeds:

| method | how the prediction is made | median miss ratio | median cosine | median length ratio | names the double from one cell (top-1) | from the centre (top-1) |
|---|---|---|---|---|---|---|
| additive expression (Week 1 baseline) | sum of measured changes | **0.114** | **0.968** | 1.09 | 0.165 | **0.567** |
| one-hot | trained decoder | 0.120 / 0.131 | 0.966 / 0.964 | 0.86 / 0.87 | 0.162 / 0.167 | 0.535 / 0.538 |
| GenePT | trained decoder | 0.140 / 0.119 | 0.956 / 0.962 | 0.94 / 0.93 | 0.148 / 0.158 | 0.472 / 0.507 |
| mean shift | added directly | 0.139 | 0.959 | 1.08 | 0.165 | 0.528 |
| STRING node2vec | trained decoder | 0.225 / 0.217 | 0.931 / 0.944 | 0.82 / 0.70 | 0.123 / 0.129 | 0.328 / 0.381 |
| PCA loadings | trained decoder | 0.234 / 0.249 | 0.935 / 0.936 | 0.73 / 0.68 | 0.128 / 0.122 | 0.294 / 0.299 |
| InfoNCE with the ρ = 1 anchor | added directly | 0.253 | 0.927 | 1.20 | 0.166 | 0.530 |
| InfoNCE, per action | added directly | 0.483 | 0.770 | 0.86 | 0.192 | 0.478 |
| InfoNCE, global α | added directly | 0.525 | 0.728 | 0.83 | **0.196** | 0.407 |
| InfoNCE direction at mean-shift length | added directly | 0.549 | 0.766 | 1.05 | 0.194 | 0.451 |
| InfoNCE as trained | added directly | 1.93 | 0.728 | 1.93 | 0.132 | 0.173 |

Seed sd is at most 0.02 on the miss ratio and 0.011 on naming from one cell. On the 108 doubles STRING covers, its
miss ratio is 0.204 / 0.202.

* **Nothing clearly beats simple addition.** One-hot and GenePT through the decoder land about as close (0.12–0.14)
  as the additive expression baseline (0.114) and the summed mean shifts (0.139).
* **The ordering of the Week 1 methods matches their gene-space scores:** additive, one-hot and GenePT ahead, then
  STRING and PCA loadings.
* **The errors go in opposite directions.** The decoders undershoot (length ratio 0.68–0.94); the plain sums
  overshoot (1.08–1.09).
* **InfoNCE added directly is behind one-hot and GenePT on landing.** The anchored set is level with STRING and PCA
  loadings; the rescaled sets are well behind.
* **Rescaled InfoNCE names the double from one cell more often** (0.19–0.20) than every other row (0.12–0.17).
* **The summed mean shift is nearly the Week 1 additive baseline.** Their squared misses per double correlate at 0.99.
* **Limits.** Only the part of each Week 1 prediction inside the 50 components counts. Week 1 predictions exist for
  the 127 measured doubles only, so there is no all-combinations column.

## 3. What this means

* **Week 2's open question** was whether InfoNCE beats the closed-form mean shift once doubles are scored. For
  landing on the double it does not, in either space, with any rescaling or the anchor.
* **InfoNCE's one advantage is narrow and unchanged from Week 2:** naming the perturbation from a single cell in
  PCA-50, once its lengths are rescaled.
* **The additive model leaves 14–17% of the squared shift unexplained,** of which about 1 point is sampling noise in
  the double's centre. The rest is error in the single vectors plus what addition cannot represent. One fixed vector
  per gene cannot reduce the second part, whatever loss trains it. The clearest pattern in it is that two strong genes
  overshoot together.

## 4. Caveats

* **One look.** There are no development doubles. If these numbers are used to pick a variant, a later score for
  that variant on the same doubles is optimistic.
* **No action vector has been through a trained decoder on the doubles.** That like-for-like test (frozen actions as
  decoder inputs, scored on doubles with the Week 1 metrics) is written, `python -I -m pertact.evaluate actions
  --doubles` with the six fixed sets in the README, and has not been run. It needs the decoders retrained on the
  POT-based vectors.
* **The breakdown in 2.4 is secondary.** The strength labels were in the code before the run, but the breakdown was
  not one of the headline scores.
* **Naming scores are averaged over cells,** as in Week 2, so doubles with more cells (50 to 1,001) weigh more.
* **The "better of the two" reference looks at the double.**

## 5. Outputs

* Code: `src/pertact/doubles.py`; tests: `tests/test_doubles.py`.
* Scores per action set, seed and rule: `results/week3_doubles.csv`.
* Scores per double: `results/week3_doubles_by_pair.csv`.
* Week 1 predictions in PCA-50: `scripts/week3_week1_in_state_space.py`, `results/week3_doubles_week1_embeddings.csv`
  and `week3_doubles_week1_embeddings_by_pair.csv`.
* Log: `logs/week3_doubles.log`.
