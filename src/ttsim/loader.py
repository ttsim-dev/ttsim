from __future__ import annotations

import importlib.util
import inspect
import itertools
import sys
from typing import TYPE_CHECKING

from ttsim.shared import (
    create_tree_from_path_and_value,
    merge_trees,
)
from ttsim.ttsim_objects import TTSIMFunction, TTSIMObject

if TYPE_CHECKING:
    import datetime
    from collections.abc import Iterable
    from pathlib import Path
    from types import ModuleType

    from ttsim.typing import NestedTTSIMObjectDict


def load_objects_tree_for_date(
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
    paths_to_objects = _find_python_files_recursively(resource_dir)

    objects_tree: NestedTTSIMObjectDict = {}

    for path in paths_to_objects:
        new_objects_tree = get_active_ttsim_objects_tree_from_module(
            path=path, date=date, root_path=resource_dir
        )

        objects_tree = merge_trees(
            left=objects_tree,
            right=new_objects_tree,
        )
    return objects_tree


def get_active_ttsim_objects_tree_from_module(
    path: Path,
    root_path: Path,
    date: datetime.date,
) -> dict[str, TTSIMFunction]:
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
    The tree of active PolicyFunctions and GroupByFunctions.
    """
    module = _load_module(path, root_path)

    ttsim_objects_orig_names = {
        name: obj
        for name, obj in inspect.getmembers(module)
        if isinstance(obj, TTSIMObject)
    }

    _fail_if_multiple_ttsim_objects_are_active_at_the_same_time(
        ttsim_objects_orig_names.values(),
        module_name=root_path / path,
    )

    active_ttsim_objects = {
        obj.leaf_name: obj
        for obj in ttsim_objects_orig_names.values()
        if obj.is_active(date)
    }

    return create_tree_from_path_and_value(
        path=_convert_path_to_tree_path(
            path=path, root_path=root_path, remove_module=True
        ),
        value=active_ttsim_objects,
    )


def _fail_if_multiple_ttsim_objects_are_active_at_the_same_time(
    ttsim_objects: Iterable[TTSIMObject],
    module_name: Path,
) -> None:
    """Raises an ConflictingTimeDependentObjectsError if multiple objects with the
    same leaf name are active at the same time.

    Parameters
    ----------
    ttsim_objects
        List of TTSIMObjects to check for conflicts.
    module_name
        The name of the module from which the TTSIMObjects are extracted.

    Raises
    ------
    ConflictingTimeDependentObjectsError
        If multiple objects with the same leaf name are active at the same time.
    """
    # Create mapping from leaf names to objects.
    leaf_names_to_objects: dict[str, list[TTSIMObject]] = {}
    for obj in ttsim_objects:
        if obj.leaf_name in leaf_names_to_objects:
            leaf_names_to_objects[obj.leaf_name].append(obj)
        else:
            leaf_names_to_objects[obj.leaf_name] = [obj]

    # Check for overlapping start and end dates for time-dependent functions.
    for leaf_name, objects in leaf_names_to_objects.items():
        dates_active = [(f.start_date, f.end_date) for f in objects]
        for (start1, end1), (start2, end2) in itertools.combinations(dates_active, 2):
            if start1 <= end2 and start2 <= end1:
                raise ConflictingTimeDependentObjectsError(
                    affected_ttsim_objects=objects,
                    leaf_name=leaf_name,
                    module_name=module_name,
                    overlap_start=max(start1, start2),
                    overlap_end=min(end1, end2),
                )


class ConflictingTimeDependentObjectsError(Exception):
    def __init__(
        self,
        affected_ttsim_objects: list[TTSIMObject],
        leaf_name: str,
        module_name: Path,
        overlap_start: datetime.date,
        overlap_end: datetime.date,
    ) -> None:
        self.affected_ttsim_objects = affected_ttsim_objects
        self.leaf_name = leaf_name
        self.module_name = module_name
        self.overlap_start = overlap_start
        self.overlap_end = overlap_end

    def __str__(self) -> str:
        overlapping_objects = [
            obj.__getattribute__("original_function_name", obj.leaf_name)
            for obj in self.affected_ttsim_objects
            if obj
        ]
        return f"""
        Functions with leaf name {self.leaf_name} in module {self.module_name} have
        overlapping start and end dates. The following functions are affected: \n\n
        {", ".join(overlapping_objects)} \n Overlapping
        from {self.overlap_start} to {self.overlap_end}."""


def _find_python_files_recursively(root_path: Path) -> list[Path]:
    """
    Find all Python files reachable from the given root path.

    Parameters
    ----------
    root_path:
        The path from which to start the search for Python files.

    Returns
    -------
    Absolute paths to all discovered Python files.
    """
    return [file for file in root_path.rglob("*.py") if file.name != "__init__.py"]


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


def _convert_path_to_tree_path(
    *, path: Path, root_path: Path, remove_module: bool
) -> tuple[str, ...]:
    """
    Convert the path from the package root to a tree path.

    Removes the package root and module name from the path.

    Parameters
    ----------
    path:
        The path to a Python module.
    root_path:
        The root path of GETTSIM's taxes and transfers modules.

    Returns
    -------
    The tree path, to be used as a key in the functions tree.

    Examples
    --------
    >>> _convert_path_to_tree_path(
    ...     path=RESOURCE_DIR / "de" / "dir" / "functions.py",
    ...     root_path=RESOURCE_DIR,
    ... )
    ("dir")
    """
    parts = path.relative_to(root_path).parts
    if remove_module:
        return parts[:-1]
    else:
        return parts
