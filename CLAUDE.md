# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project Overview

TTSIM (Taxes and Transfers SIMulator backend) provides the computing engine for GETTSIM,
the German Taxes and Transfers Simulator. It uses a DAG-based architecture to compute
tax and benefit calculations with support for NumPy and JAX backends.

## Development Commands

This project uses [pixi](https://pixi.sh) for environment management.

```bash
# Run tests (NumPy backend)
pixi run -e py314 tests

# Run tests (JAX backend)
pixi run -e py314-jax tests-jax

# Run a single test file
pixi run -e py314 pytest tests/test_main.py

# Run a single test
pixi run -e py314 pytest tests/test_main.py::test_function_name -v

# Type checking with ty
pixi run -e ty ty

# Available environments: py311, py312, py313, py314, py314-jax, py314-cuda, py314-metal, ty
```

Before finishing any task that modifies code, always run these three verification steps
in order:

1. `pixi run -e py314-jax ty` (type checker)
1. `pixi run -e py314-jax prek run --all-files` (quality checks: linting, formatting,
   yaml, etc.)
1. `pixi run -e py314-jax tests -n 7` (full test suite)

Pre-commit hooks run automatically via pre-commit.ci. Linting uses Ruff with `ALL` rules
enabled.

## Architecture

### Two-Level DAG System (see GEP 7)

1. **Interface DAG** (`src/ttsim/interface_dag_elements/`): High-level orchestration
   connecting inputs to outputs. Key nodes include:

   - `policy_date`: Date for which the policy environment is set up
   - `orig_policy_objects`: All functions and parameters shipped with the system
   - `policy_environment`: Functions/parameters relevant at the policy date
   - `input_data`: User-provided data (via DataFrame + mapper or direct pytree)
   - `tt_targets`: Which outputs to compute
   - `specialized_environment`: Combines policy environment with data, processes
     parameters, builds the TT DAG
   - `results`: Final outputs in user-requested format
   - `templates`: Helpers for constructing input data mappers

1. **TT DAG** (`src/ttsim/tt/`): The core computation layer built within
   `specialized_environment`. Contains `ColumnObject` subclasses that operate on data
   columns:

   - `PolicyFunction`: Main computation functions (auto-vectorized from scalar
     definitions)
   - `PolicyInput`: Input data declarations
   - `AggByGroupFunction`, `AggByPIDFunction`: Aggregation functions
   - `ParamFunction`, `ParamObject`: Parameter processing (scalars or dictionaries, not
     columns)

### Entry Point Flow (see GEP 7)

`ttsim.main()` in `entry_point.py` is the single entry point. Users specify:

- `main_target` or `main_targets`: What to compute (use `MainTarget` for autocompletion)
- `policy_date_str`: Date for the policy environment (ISO format `YYYY-MM-DD`)
- `input_data`: User data (via `InputData` helper classes)
- `tt_targets`: Which tax/transfer outputs to compute (via `TTTargets`)

The function:

1. Harmonizes inputs into qualified names (qnames)
1. Loads interface functions from `interface_dag_elements/`
1. Builds the interface DAG using the `dags` library
1. Creates a combined function with `dags.concatenate_functions()`
1. Executes and returns the requested `main_target(s)`

### Backend Abstraction

The `backend` parameter (`"numpy"` or `"jax"`) controls computation:

- `xnp(backend)`: Returns numpy or jax.numpy module
- `dnp(backend)`: Returns module for data operations
- PolicyFunctions are automatically vectorized based on `vectorization_strategy`

### Key Conventions (see GEPs 1, 2, 4, 6)

- **Qualified names (qnames)**: Double-underscore separated paths (e.g.,
  `"housing_benefits__amount"`)
- **Tree paths**: Tuple representation of the same (e.g.,
  `("housing_benefits", "amount")`)
- **`p_id`**: Primary person identifier, must be present in all data
- **Group identifiers (`[x]_id`)**: Columns ending in `_id` where `[x]` != `p` identify
  groupings of individuals. Values repeat for all individuals in a group.
- **Person pointers (`p_id_[x]`)**: References to another individual's `p_id` (e.g.,
  `p_id_parent_1` for parent 1, `p_id_recipient` for benefit recipient). Value of -1
  indicates no link.
- **Time unit suffixes**: `_y`, `_q`, `_m`, `_w`, `_d` for year, quarter, month, week,
  day
- **Aggregation level suffixes**: `_[x]`, etc. indicating the unit of aggregation
- **Suffix ordering**: Time unit before aggregation level (e.g., `betrag_m_[x]` =
  monthly amount at x level)
- **Auto-aggregation**: If `my_col` exists and `my_col_[x]` is requested, a sum
  aggregation is auto-generated

### Test Structure

- `tests/`: Main test suite for ttsim-backend
- `src_mettsim/tests_middle_earth/`: Tests for the example "Middle Earth" policy
  implementation
- Use pixi environment `py314` (numpy, current Python), `py311` (numpy, oldest supported
  Python), `py-314-jax` (Jax, CPU) or `py314-cuda` (Jax, CUDA) to select Python version
  and backend
- Markers: `@pytest.mark.skipif_jax`, `@pytest.mark.skipif_numpy`

### mettsim Package

`src_mettsim/` contains an example implementation ("Middle Earth" fictional tax system)
demonstrating how to build a policy environment using TTSIM primitives. It's installed
as a separate editable package.

## Relevant GEPs

The
[GETTSIM Enhancement Protocols](https://github.com/ttsim-dev/gettsim/tree/main/docs/geps)
define conventions that ttsim implements:

- **GEP 1**: Naming conventions (identifiers, German names, time/unit suffixes)
- **GEP 2**: Internal data representation (1-d arrays, group identifiers, person
  pointers)
- **GEP 3**: Parameters of the taxes and transfers system (YAML structure, types)
- **GEP 4**: DAG-based computational backend (core ttsim architecture)
- **GEP 5**: Optional rounding via `RoundingSpec`
- **GEP 6**: Unified architecture (namespaces, qualified names, `start_date`/`end_date`
  for functions)
- **GEP 7**: User interface (`main()` function, `MainTarget`, input/output handling)
