# Changes

This is a record of all past `ttsim` releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/) and all
releases are available on [Anaconda.org](https://anaconda.org/conda-forge/ttsim).

## v1.0a2 — 2025-07-xx

- {gh}`13` Put `plot_tt_dag` and `plot_interface_dag` into `ttsim.plot.dag` namespace;
  rename to `tt` and `interface`. ({ghuser}`MImmesberger`)

- {gh}`11` Fix several bugs in `plot_tt_dag`. ({ghuser}`MImmesberger`)

- {gh}`9` Raise an error when passing data but no targets. ({ghuser}`hmgaudecker`)

- {gh}`8` Allow for input data as targets. ({ghuser}`MImmesberger`)

- {gh}`6` Fail if the leaf name of an object in the policy environment differs from the
  last element of the path ({ghuser}`MImmesberger`, {ghuser}`hmgaudecker`)

## v1.0a1 — 2025-07-24

- All development happened in a single GETTSIM repository. See
  [the GETTSIM changelog](https://gettsim.readthedocs.io/en/latest/changes.html) for all
  history.
