Run the unmodified CRAN mcmcse 1.5-0 ordinary overlapping batch means estimator.
The R package selects its own batch length, with r=1 and adjust=FALSE. Requires
R 4.2.2, the pinned mcmcse package in the benchmark image, Python 3 and NumPy.
No build step. `./run.sh INPUT.npy OUTPUT.npy`
