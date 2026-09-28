# Third-party code and data

## Not bundled: the Localize-MI BIDS reader

`HISDataset_BIDS` reads the Localize-MI dataset through `load_bids`, which
belongs to the dataset authors' own release:

> https://github.com/iTCf/mikulan_et_al_2020 — `fx_bids.py`

That repository carries **no licence**, so its code is not redistributed here
even though it is publicly viewable. An earlier draft of this package contained
`eeg_sl/data/bids_io.py`, a near-verbatim copy of their `load_bids` and
`load_trans`; it has been removed.

To use `HISDataset_BIDS`, fetch `fx_bids.py` from the link above and put it on
`PYTHONPATH`, or drop it in `eeg_sl/data/`. Everything else in this package —
the model, the collator, the simulation datasets — works without it.

Please cite the dataset:

> Mikulan, E. et al. Simultaneous human intracerebral stimulation and HD-EEG,
> ground-truth for source localization methods. *Scientific Data* **7**, 127
> (2020). https://doi.org/10.1038/s41597-020-0467-x

## Datasets

No dataset ships with this repository. Localize-MI is public; the Tsinghua
SSVEP set and the clinical recordings used in pre-training are obtained from
their own holders under their own terms.

## Geometry assets

The released `demist-assets` attachment contains a head model derived from the
Colin27 MNI template, together with a leadfield and eLORETA weights computed on
it. Cite the Colin27 template if you use them:

> Holmes, C. J. et al. Enhancement of MR images using registration for signal
> averaging. *J. Comput. Assist. Tomogr.* **22**, 324–333 (1998).
