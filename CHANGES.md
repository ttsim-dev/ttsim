# Changes

This is a record of all past `ttsim` releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/) and all
releases are available on [Anaconda.org](https://anaconda.org/conda-forge/ttsim).

## v1.0.1 — unpublished

- {gh}`42` Pre-sort user data by p_id for performance ({ghuser}`JuergenWiemers`)

- {gh}`41` Improve performance of `tt.shared.join` and
  `ttsim.interface_dag_elements.fail_if.foreign_keys_are_invalid_in_data`
  ({ghuser}`JuergenWiemers`)

- {gh}`40` Improve performance of `aggregation_numpy` and `data_converters`
  ({ghuser}`JuergenWiemers`)

## v1.0 — 2025-08-09

- {gh}`38` Plotting: Replace `nodes` selection type with `all_paths`.

- {gh}`37` Make it possible to pass all main args as class methods.
  ({ghuser}`MImmesberger`)

- {gh}`32` Build inputs template and plotting DAG from specialized environment based on
  policy_inputs ({ghuser}`MImmesberger`)

- {gh}`34` Optimize JAX performance in data preparation pipeline
  ({ghuser}`JuergenWiemers`)

- {gh}`23` Remove orphaned policy inputs from the TT DAG. ({ghuser}`MImmesberger`)

- {gh}`19` Clearer architecture ({ghuser}`hmgaudecker`)

- {gh}`17` Add type for sparse dicts with int keys param. ({ghuser}`MImmesberger`)

- {gh}`16` Add fail/warn mechanism to ColumnObjects and ParamFunctions.
  ({ghuser}`hmgaudecker`)

- {gh}`15` Do not call len() on unsized arrays. ({ghuser}`hmgaudecker`)

- {gh}`14` Do not loop over the attributes of Jax arrays in
  `fail_if.backend_has_changed` ({ghuser}`hmgaudecker`)

- {gh}`13` Put `plot_tt_dag` and `plot_interface_dag` into `ttsim.plot.dag` namespace;
  rename to `tt` and `interface`. ({ghuser}`MImmesberger`)

- {gh}`11` Fix several bugs in `plot_tt_dag`. ({ghuser}`MImmesberger`)

- {gh}`9` Raise an error when passing data but no targets. ({ghuser}`hmgaudecker`)

- {gh}`8` Allow for input data as targets. ({ghuser}`MImmesberger`)

- {gh}`6` Fail if the leaf name of an object in the policy environment differs from the
  last element of the path ({ghuser}`MImmesberger`, {ghuser}`hmgaudecker`)

- Prior to this, all development happened in a single GETTSIM repository. See
  [the GETTSIM changelog](https://gettsim.readthedocs.io/en/latest/changes.html) for the
  history.
