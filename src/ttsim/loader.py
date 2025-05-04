from __future__ import annotations

import importlib.util
import inspect
import sys
from typing import TYPE_CHECKING, Literal

import dags.tree as dt
import yaml

from ttsim.ttsim_objects import TTSIMObject

if TYPE_CHECKING:
    import datetime
    from pathlib import Path
    from types import ModuleType

    from ttsim.typing import (
        FlatTTSIMObjectDict,
        NestedTTSIMObjectDict,
        OrigYamlParamSpec,
        OrigYamlTree,
    )


def active_ttsim_objects_tree(root: Path, date: datetime.date) -> NestedTTSIMObjectDict:
    """
    Traverse `root` and return all TTSIMObjects for a given date.

    Parameters
    ----------
    root:
        The directory to traverse.
    date:
        The date for which policy objects should be loaded.

    Returns
    -------
    A tree of active TTSIMObjects.
    """

    orig_flat_objects_tree = orig_ttsim_objects_tree(root)

    flat_objects_tree = {
        (*orig_path[:-2], obj.leaf_name): obj
        for orig_path, obj in orig_flat_objects_tree.items()
        if obj.is_active(date)
    }

    return dt.unflatten_from_tree_paths(flat_objects_tree)


def orig_ttsim_objects_tree(root: Path) -> FlatTTSIMObjectDict:
    """
    Load the original TTSIMObjects tree from the resource directory.

    "Original" means:
    - Module names are not removed from the path.
    - The last path element is the TTSIMObject's original name, not the leaf name.

    Parameters
    ----------
    root:
        The resource directory to load the TTSIMObjects tree from.
    """
    return {
        k: v
        for path in _find_files_recursively(root=root, suffix=".py")
        for k, v in _tree_path_to_orig_ttsim_objects(path=path, root=root).items()
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


def _tree_path_to_orig_ttsim_objects(path: Path, root: Path) -> FlatTTSIMObjectDict:
    """Extract all active PolicyFunctions and GroupByFunctions from a module.

    Parameters
    ----------
    path
        The path to the module from which to extract the active functions.
    root
        The path to the directory that contains the functions.

    Returns
    -------
    A flat tree of TTSIMObjects.
    """
    module = _load_module(path=path, root=root)
    tree_path = path.relative_to(root).parts
    return {
        (*tree_path, name): obj
        for name, obj in inspect.getmembers(module)
        if isinstance(obj, TTSIMObject)
    }


def _load_module(path: Path, root: Path) -> ModuleType:
    name = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
    spec = importlib.util.spec_from_file_location(name=name, location=path)
    # Assert that spec is not None and spec.loader is not None, required for mypy
    _msg = f"Could not load module spec for {path},  {root}"
    if spec is None:
        raise ImportError(_msg)
    assert spec.loader is not None, _msg
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def orig_yaml_tree(root: Path) -> OrigYamlTree:
    """
    Load the original contents of yaml files found in *root*.

    "Original" means:
    - Module names are not removed from the path.
    - The contents of the yaml files are not parsed, just the outermost key becomes part
      of the tree path

    Parameters
    ----------
    root:
        The resource directory to load the TTSIMObjects tree from.
    """
    return {
        k: v
        for path in _find_files_recursively(root=root, suffix=".yaml")
        for k, v in _tree_path_to_orig_yaml_object(path=path, root=root).items()
        # FixMe: Temporary solution so that old stuff works in parallel.
        if "parameters" not in k
    }


def _tree_path_to_orig_yaml_object(path: Path, root: Path) -> OrigYamlTree:
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
    raw_contents: dict[str, OrigYamlParamSpec] = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=yaml.CLoader,
    )
    tree_path = path.relative_to(root).parts
    return {(*tree_path, name): obj for name, obj in raw_contents.items()}
