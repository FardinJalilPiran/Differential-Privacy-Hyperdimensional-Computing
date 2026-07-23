# Data

This folder is where datasets land at runtime. Its contents are gitignored.

## MNIST (default)

Downloaded automatically the first time you run any script — no manual step
needed. Roughly 55 MB, cached in `data/MNIST/` and reused afterwards.

## The dataset used in the paper

The results reported in the paper come from in-situ high-speed camera
measurements of a laser powder bed fusion build on an EOS M270 machine at NIST
(256x256 frames at 1000 fps, nickel alloy 625, 40.5-degree overhang). That data
is not publicly redistributable and is not included in this repository.

## Using your own data

Add a loader to `src/dphd/data.py` returning:

- `X`: float32 array, shape `(n_samples, n_features)`, values roughly in [0, 1]
- `y`: int64 array, shape `(n_samples,)`, labels `0 .. n_classes-1`

then register it in `load_dataset` and pass `--dataset <yourname>`. Nothing else
in the codebase is dataset specific.
