# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
- `pixi run tests` - Run full test suite with pytest
- `pixi run tests-jax` - Run tests with JAX backend for performance testing
- `pixi run -e py311 tests` - Run tests with specific Python version (py311, py312, py313)

### Code Quality
- `pixi run mypy` - Type checking with mypy
- `ruff check` - Lint code (automatically configured in pyproject.toml)
- `ruff format` - Format code
- `pre-commit run --all-files` - Run all pre-commit hooks

### Documentation
- Build docs using the `docs` environment (configuration in pyproject.toml)

## Codebase Architecture

### Core Structure
GETTSIM implements the German tax and transfer system using a function-based, DAG (Directed Acyclic Graph) architecture:

- **`src/_gettsim/`**: German tax system implementation organized by policy domains
- **`src/ttsim/`**: Generic tax-transfer simulation framework (domain-agnostic)
- **`src/gettsim/`**: Main user-facing API

### Policy Module Organization
Each policy area (e.g., `kindergeld/`, `einkommensteuer/`) follows this pattern:
- Core logic in `.py` files with `@policy_function()` decorators
- Historical parameters in `.yaml` files with effective date ranges
- Input definitions in `inputs.py`
- Module initialization in `__init__.py`

Key policy modules:
- **Tax**: `einkommensteuer/`, `lohnsteuer/`, `solidaritätszuschlag/`
- **Social Insurance**: `sozialversicherung/` (health, pension, unemployment sub-modules)
- **Transfers**: `kindergeld/`, `kinderzuschlag/`, `elterngeld/`, `arbeitslosengeld_2/`, `wohngeld/`
- **Family**: `familie/`, `unterhalt/`, `unterhaltsvorschuss/`

### Function Implementation Patterns
- Functions use `@policy_function()` decorator with validity date ranges
- Historical accuracy: parameters and logic are precisely dated
- Type hints required for all functions
- Pure functions with clear input/output contracts
- Support for both NumPy and JAX backends

### Testing Structure
- **`src/_gettsim_tests/`**: German tax system tests
- **`tests/ttsim/`**: Generic framework tests
- Test data organized by policy area and effective date (e.g., `kindergeld/2024-01-01/`)
- YAML-based test cases with structured input/output specifications

### Key Files
- **`pyproject.toml`**: All project configuration (Pixi, Ruff, MyPy, Pytest)
- **`pixi.lock`**: Dependency lock file
- **`conftest.py`**: Pytest configuration and fixtures

### Development Tools
- **Pixi**: Dependency and environment management with multiple Python versions
- **Ruff**: Linting and formatting (configured for comprehensive rules)
- **MyPy**: Type checking (strict configuration)
- **Pre-commit**: Code quality hooks

### Special Considerations
- Parameter files (`.yaml`) contain historical policy parameters with German/English descriptions
- Test data reflects real-world scenarios and edge cases
- Multiple backend support requires consideration of vectorization constraints
- Date-sensitive calculations require proper temporal parameter handling
