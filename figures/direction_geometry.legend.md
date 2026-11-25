**Action directions: spread across genes against
distance to held-out transitions.** u(a) = f(a)/‖f(a)‖ for the 105 training genes. Target m̂_val(a): the gene's mean
transition on validation OT pairs (perturbed state minus OT-weighted control states), normalised; no set is scored
against the data it was fitted on. Distance = mean ‖u(a) − m̂_val(a)‖² = 2 − 2 cos (lower = closer). Spread = log
mean over gene pairs of exp(−2‖u(a) − u(b)‖²), Wang & Isola's uniformity (lower = more spread). Random = 105 Gaussian
unit vectors. InfoNCE and InfoNCE direction only (trained with length held at the mean-shift length): mean of seeds
0–2. **A.** Circles PCA-50, squares scVI-30. **B.** Distance by tier of the training mean-shift norm (tertiles); dark
PCA-50, light scVI-30. Values: PCA-50 InfoNCE: distance 0.59, spread -3.06; PCA-50 InfoNCE, direction only: distance 0.29, spread -2.47; PCA-50 mean shift: distance 0.16, spread -1.73; PCA-50 random: distance 2.02, spread -3.85; scVI-30 InfoNCE: distance 0.56, spread -3.27; scVI-30 InfoNCE, direction only: distance 0.29, spread -3.05; scVI-30 mean shift: distance 0.20, spread -2.49; scVI-30 random: distance 2.00, spread -3.71. Source: scripts/plot_direction_geometry.py.
