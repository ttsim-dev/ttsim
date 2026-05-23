from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
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
from ttsim.typing import FlatColumnObjectsParamFunctions

if TYPE_CHECKING:
    from ttsim.typing import (
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
    """Find all files with *suffix* in *root* and its subdirectories.

    Args:
        root: The path from which to start the search for Python files.
        suffix: The suffix of files to look for.

    Returns:
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

    Args:
        path: The path to the module from which to extract the active functions.
        root: The path to the directory that contains the functions.

    Returns:
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
    canonical_name = _find_canonical_module_name(path)
    if canonical_name is not None:
        # Use the proper Python import path so objects defined in the module
        # have an importable `__module__` — required for `cloudpickle.dumps`
        # to round-trip the policy environment via pickle-by-reference.
        return importlib.import_module(canonical_name)

    # Policy environment is a bare directory tree (no `__init__.py` chain
    # reaching `sys.path`). Objects defined in modules loaded this way carry
    # a non-importable `__module__` and cannot be cloudpickled by reference.
    name = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
    spec = importlib.util.spec_from_file_location(name=name, location=path)
    _msg = f"Could not load module spec for {path},  {root}"
    if spec is None:
        raise ImportError(_msg)
    if spec.loader is None:
        raise ImportError(_msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def _find_canonical_module_name(path: Path) -> str | None:
    """Return the canonical dotted import name for ``path``, if importable.

    Walks up the ``__init__.py`` chain from ``path``'s parent. If the topmost
    ``__init__.py``-having directory's parent is on ``sys.path``, the dotted
    name formed by joining the directory names (top to bottom) plus the file
    stem is the canonical import path. Returns ``None`` if ``path`` is not
    part of a proper Python package.
    """
    parts = [path.stem]
    current = path.parent
    while (current / "__init__.py").is_file():
        parts.append(current.name)
        current = current.parent
    if len(parts) == 1:
        return None
    sys_path_resolved = {Path(p).resolve() for p in sys.path if Path(p).exists()}
    if current.resolve() in sys_path_resolved:
        return ".".join(reversed(parts))
    return None


def _tree_path_to_orig_yaml_object(path: Path, root: Path) -> FlatOrigParamSpecs:
    """Extract all active PolicyFunctions and GroupByFunctions from a module.

    Args:
        path: The path to the yaml file from which to extract parameter specifications.
        root: The path to the policy environment's root directory.

    Returns:
        A flat tree of yaml contents.

    """
    raw_contents: dict[str, OrigParamSpec] = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=yaml.SafeLoader,
    )
    tree_path = path.relative_to(root).parts
    return {(*tree_path, name): obj for name, obj in raw_contents.items()}
