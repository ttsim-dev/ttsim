# Changes

This is a record of all past `ttsim` releases and what went into them in reverse
chronological order. We follow [semantic versioning](https://semver.org/) and all
releases are available on [Anaconda.org](https://anaconda.org/conda-forge/ttsim).

## Unreleased

- {gh}`102` Canonicalize input dtypes: coerce uint columns to signed int, normalize
  pandas-nullable / Float64 / pyarrow columns to numpy (NA→NaN for floats, fail for
  int/bool with NA). Closes #97, #94. ({ghuser}`hmgaudecker`)
- {gh}`101` Adopt the package-wide beartype claw with a typed exception hierarchy at
  user-facing boundaries (`InputDataError`, `EntryPointError`, `TTTargetsError`,
  `PolicyFunctionDefinitionError`, …) so malformed input is rejected with curated
  errors. Every `@*_function` decorator (`@policy_function`, `@param_function`,
  `@agg_by_p_id_function`, `@agg_by_group_function`, `@group_creation_function`) now
  also requires the wrapped function to carry an annotation on every parameter and on
  the return value; missing annotations raise the decorator's `*DefinitionError` at
  decoration time. Floor pin `jaxtyping >= 0.3.10` so the cloudpickle round-trip of
  jaxtyping types works out of the box (the three `_array_types` `object()` sentinels
  are now `__reduce__`-backed singleton classes upstream). ({ghuser}`hmgaudecker`)

## v1.2.1 — 2026-05-24

- {gh}`99` Require `dags >= 0.6`; adapt to its new wrapper annotations.
  ({ghuser}`hmgaudecker`)
- {gh}`92` Let piecewise polynomials return scalars if the input is a scalar.
  ({ghuser}`MImmesberger`)

## v1.2.0 — 2026-03-19

- {gh}`76` Refactor piecewise polynomial (implements GEP 8). ({ghuser}`hmgaudecker`,
  {ghuser}`MImmesberger`)
- {gh}`86` Require `dags >= 0.5.1`. ({ghuser}`hmgaudecker`)
- {gh}`83` Allow non-string annotations in vectorization. ({ghuser}`hmgaudecker`)
- {gh}`80` Generalize type hints for time conversion functions. ({ghuser}`hmgaudecker`)
- {gh}`75` Improve test coverage. ({ghuser}`MImmesberger`)
- {gh}`71` Fix pygraphviz plugin registration on macOS. ({ghuser}`MImmesberger`)
- {gh}`68` Get rid of UnitTest style tests. ({ghuser}`hmgaudecker`)
- {gh}`67` Add glob-style pattern matching for `node_colormap` in DAG plotting.
  ({ghuser}`hmgaudecker`)

## v1.1.1 — 2026-01-12

- {gh}`65` Add Python 3.14 support. ({ghuser}`hmgaudecker`)
- {gh}`63` Use ty as the type checker for ttsim. Many improvements to type safety.
  ({ghuser}`hmgaudecker`)
- {gh}`56` Add and expose time converters targeting stocks.
- {gh}`53` Raise error if scalar is passed to ColumnFunction expecting array input.
  ({ghuser}`MImmesberger`)
- {gh}`52` Fix fail_if checks: input data tree was too greedy and environment had an
  ambiguous name. ({ghuser}`MImmesberger`)
- {gh}`49` Fix test failure on WSL2 due to different DAG execution order
  ({ghuser}`JuergenWiemers`)

## v1.1.0 — 2025-08-26

- {gh}`44` Add a default fixed colormap to plots and general improvements to plotting.
  ({ghuser}`hmgaudecker`)

- {gh}`46` Optimization for `fail_if.foreign_keys_are_invalid_in_data`
  ({ghuser}`JuergenWiemers`)

- {gh}`42` Pre-sort user data by p_id for performance ({ghuser}`JuergenWiemers`,
  {ghuser}`hmgaudecker`)

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
