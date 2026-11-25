**Action directions on the sphere,
PCA-50, after Wang & Isola (2020) Fig. 2.** Columns: random unit vectors in 50-D, the mean shift (training pairs),
InfoNCE as trained (seed 0). **A.** Distance from each gene's direction to its own held-out mean transition,
‖u(a) − m̂_val(a)‖ = √(2 − 2 cos); 0 = same direction, √2 = orthogonal. **B.** Directions projected onto each set's
own top-2 plane, renormalised to the unit circle and rotated so their mean sits at angle 0; dots = genes, band =
von Mises density (κ = 8). The plane keeps 10%, 68% and 31%
of the squared norm, so B is a 2-D view; spread is computed in full 50-D (values in direction_geometry).
Source: scripts/plot_direction_geometry.py.
