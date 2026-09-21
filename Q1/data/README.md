# Q1 Data

Datasets are **not** committed to Git. Download them locally into this folder
(e.g. `Q1/data/FashionMNIST/`); everything here is ignored except this README
and `split_indices/`.

## `split_indices/`

Store the exact train/validation/test index arrays used for each experiment
(e.g. `fashion_mnist_seed0_val10pct.npz` or `.json`) so every split can be
reproduced. These files are small and **are** tracked by Git.
