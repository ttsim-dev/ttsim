from __future__ import annotations

import importlib.util
import inspect
import sys
from typing import TYPE_CHECKING, Literal

import yaml

from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)
from ttsim.tt.column_objects_param_function import (
    ColumnObject,
    ParamFunction,
)

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from ttsim.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
        OrigParamSpec,
    )


@interface_input()
def root() -> Path:
    """The root directory of the policy environment's elements."""


@interface_function()
def column_objects_and_param_functions(
    root: Path,
) -> FlatColumnObjectsParamFunctions:
    """
    The original ColumnObjectParamFunctions dictionary.

    "Original" means:
    - Module names are not removed from the path.
    - The last path element is the ColumnObject's original name, not the leaf name.
    """
    return {
        k: v
        for path in _find_files_recursively(root=root, suffix=".py")
        for k, v in _tree_path_to_orig_column_objects_params_functions(
            path=path,
            root=root,
        ).items()
    }


@interface_function()
def param_specs(root: Path) -> FlatOrigParamSpecs:
    """
    The original parameter specifications, typically loaded from yaml files.

    "Original" means:
    - Module names are not removed from the path.
    - The outermost key in each yaml file becomes the leaf name of the tree path.
      Beyond that, the contents of the yaml files are not parsed.
    """
    return {
        k: v
        for path in _find_files_recursively(root=root, suffix=".yaml")
        for k, v in _tree_path_to_orig_yaml_object(path=path, root=root).items()
    }


def _find_files_recursively(root: Path, suffix: Literal[".py", ".yaml"]) -> list[Path]:
    """
    Find all files with *suffix* in *root* and its subdirectories.

    Parameters
    ----------
    root:
        The path from which to start the search for Python files.
    suffix:
        The suffix of files to look for.

    Returns
    -------
    Absolute paths to all discovered files with *suffix*.
    """
    names_to_exclude = {"__init__.py"}
    return [
        file for file in root.rglob(f"*{suffix}") if file.name not in names_to_exclude
    ]


def _tree_path_to_orig_column_objects_params_functions(
    path: Path,
    root: Path,
) -> FlatColumnObjectsParamFunctions:
    """Extract all active PolicyFunctions and GroupByFunctions from a module.

    Parameters
    ----------
    path
        The path to the module from which to extract the active functions.
    root
        The path to the directory that contains the functions.

    Returns
    -------
    A flat tree of ColumnObjectParamFunctions.
    """
    module = load_module(path=path, root=root)
    tree_path = path.relative_to(root).parts
    return {
        (*tree_path, name): obj
        for name, obj in inspect.getmembers(module)
        if isinstance(obj, ColumnObject | ParamFunction)
    }


def load_module(path: Path, root: Path) -> ModuleType:
    name = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
    spec = importlib.util.spec_from_file_location(name=name, location=path)
    # Assert that spec is not None and spec.loader is not None, required for mypy
    _msg = f"Could not load module spec for {path},  {root}"
    if spec is None:
        raise ImportError(_msg)
    if spec.loader is None:
        raise ImportError(_msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def _tree_path_to_orig_yaml_object(path: Path, root: Path) -> FlatOrigParamSpecs:
    """Extract all active PolicyFunctions and GroupByFunctions from a module.

    Parameters
    ----------
    path
        The path to the yaml file from which to extract parameter specifications.
    root
        The path to the policy environment's root directory.

    Returns
    -------
    A flat tree of yaml contents.
    """
    raw_contents: dict[str, OrigParamSpec] = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=yaml.CSafeLoader,
    )
    tree_path = path.relative_to(root).parts
    return {(*tree_path, name): obj for name, obj in raw_contents.items()}
