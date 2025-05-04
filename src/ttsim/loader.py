from __future__ import annotations

import importlib.util
import inspect
import sys
from typing import TYPE_CHECKING, Literal

import dags.tree as dt

from ttsim.ttsim_objects import TTSIMObject

if TYPE_CHECKING:
    import datetime
    from pathlib import Path
    from types import ModuleType

    from ttsim.typing import FlatTTSIMObjectDict, NestedTTSIMObjectDict


def active_ttsim_objects_tree(
    resource_dir: Path, date: datetime.date
) -> NestedTTSIMObjectDict:
    """
    Traverse `resource_dir` and return all TTSIMObjects for a given date.

    Parameters
    ----------
    resource_dir:
        The directory to traverse.
    date:
        The date for which policy objects should be loaded.

    Returns
    -------
    A tree of active TTSIMObjects.
    """

    orig_flat_objects_tree = orig_ttsim_objects_tree(resource_dir)

    flat_objects_tree = {
        (*orig_path[:-2], obj.leaf_name): obj
        for orig_path, obj in orig_flat_objects_tree.items()
        if obj.is_active(date)
    }

    return dt.unflatten_from_tree_paths(flat_objects_tree)


def orig_ttsim_objects_tree(resource_dir: Path) -> FlatTTSIMObjectDict:
    """
    Load the original TTSIMObjects tree from the resource directory.

    "Original" means:
    - Module names are not removed from the path.
    - The last path element is the TTSIMObject's original name, not the leaf name.

    Parameters
    ----------
    resource_dir:
        The resource directory to load the TTSIMObjects tree from.
    """
    return {
        k: v
        for path in _find_files_recursively(root=resource_dir, suffix=".py")
        for k, v in _get_orig_ttsim_objects_from_module(
            path=path, root_path=resource_dir
        ).items()
    }


def _get_orig_ttsim_objects_from_module(
    path: Path,
    root_path: Path,
) -> FlatTTSIMObjectDict:
    """Extract all active PolicyFunctions and GroupByFunctions from a module.

    Parameters
    ----------
    path
        The path to the module from which to extract the active functions.
    root_path
        The path to the directory that contains the functions.
    date
        The date for which to extract the active functions.

    Returns
    -------
    A flat tree of TTSIMObjects.
    """
    module = _load_module(path=path, root_path=root_path)
    tree_path = path.relative_to(root_path).parts
    return {
        (*tree_path, name): obj
        for name, obj in inspect.getmembers(module)
        if isinstance(obj, TTSIMObject)
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


def _load_module(path: Path, root_path: Path) -> ModuleType:
    name = path.relative_to(root_path).with_suffix("").as_posix().replace("/", ".")
    spec = importlib.util.spec_from_file_location(name=name, location=path)
    # Assert that spec is not None and spec.loader is not None, required for mypy
    _msg = f"Could not load module spec for {path},  {root_path}"
    if spec is None:
        raise ImportError(_msg)
    assert spec.loader is not None, _msg
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module
