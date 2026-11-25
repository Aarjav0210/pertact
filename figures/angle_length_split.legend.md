**Action directions compared at matched length (validation cells).** Every action is f(a) = c·ℓ_a·u_a with
‖u_a‖ = 1; c ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 2.5} is one global factor and the x-axis is the resulting median
length relative to the mean shift m_a (mean OT displacement on training pairs). Directions u: mean shift; shuffled
(mean-shift directions permuted across genes); InfoNCE (direction of the freely trained action); InfoNCE, direction
only (trained with its length held at ‖m_a‖, random start). Solid: ℓ_a = ‖m_a‖. Dashed: ℓ_a = u_a·m_a, the
least-squares length on the line u_a. InfoNCE curves are the mean of seeds 0–2 with ±1 sd bands. Vertical line:
mean-shift length. **A, B.** Instance retrieval MRR (query = heaviest OT partner control + f(a), candidates = all
validation perturbed cells). **C, D.** Identification top-1 over 106 candidates from the pooled-OT start. Source:
scripts/angle_length_split.py, scripts/plot_angle_length.py.
