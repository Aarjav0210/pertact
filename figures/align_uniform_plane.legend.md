**Alignment and uniformity of every action set (validation cells).** Query q = s(c) + f(a), with c
the heaviest OT partner control; keys = all 9,444 held-out perturbed cells; k⁺ = the query's own cell.
Alignment = mean ‖q − k⁺‖² / D₀ (D₀ = mean squared distance between random training cells). Uniformity = mean
over queries of log mean over keys of exp(−‖q − k‖² / (t·D₀)), t = the space's chosen temperature (PCA-50 0.03,
scVI-30 0.1); lower = more spread. Alignment / t + uniformity + log N is the InfoNCE row loss with all N = 9,444 validation cells as
a negative. Large markers: saved action sets (InfoNCE sets averaged over seeds 0–2). Gray = references (zero,
shuffled mean shift, mean shift); blue = InfoNCE as trained; yellow = InfoNCE rescaled (one global α, per-action
c_a), length-swapped (InfoNCE direction at mean-shift length, mean-shift direction at InfoNCE length) and the
anchored model (unlabelled yellow points sit between mean shift and InfoNCE dir, MS length). Lines: seed 0
retrained over τ ∈ {0.01, 0.03, 0.1, 0.3} (blue) and over the anchor weight ρ ∈ {0.01, 0.03, 0.1, 0.3, 1} at the
chosen τ (yellow, running from InfoNCE toward mean shift as ρ grows). Grid points beyond the axes: PCA-50 τ = 0.1: alignment 1.11, uniformity -37.4, MRR 0.176, top-1 0.019; PCA-50 τ = 0.3: alignment 5.28, uniformity -165.4, MRR 0.101, top-1 0.000; scVI-30 τ = 0.3: alignment 1.80, uniformity -20.4, MRR 0.102, top-1 0.003.
**A, B.** The plane per state space. **C.** Identification top-1 (pooled-OT start, 106 candidates) against
alignment relative to the mean shift. **D.** Retrieval MRR against uniformity relative to the mean shift (zero
action omitted). C, D: circles PCA-50, squares scVI-30. Source: scripts/align_uniform_probe.py,
scripts/plot_align_uniform.py.
