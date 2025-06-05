from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from ttsim.tt_dag_elements.column_objects_param_function import (
    ColumnObject,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ttsim.tt_dag_elements.typing import (
        QualNameColumnFunctionsWithProcessedParamsAndScalars,
        QualNameData,
        QualNameTargetList,
    )


_DUMMY_COLUMN_OBJECT = ColumnObject(
    leaf_name="dummy",
    start_date=datetime.date(1900, 1, 1),
    end_date=datetime.date(2099, 12, 31),
)


def raw_results__columns(
    qual_name_input_data: QualNameData,
    tax_transfer_function: Callable[[QualNameData], QualNameData],
) -> QualNameData:
    return tax_transfer_function(qual_name_input_data)


def raw_results__params(
    names__target_params: QualNameTargetList,
    column_functions_with_processed_params_and_scalars: QualNameColumnFunctionsWithProcessedParamsAndScalars,  # noqa: E501
) -> QualNameData:
    return {
        pt: column_functions_with_processed_params_and_scalars[pt]
        for pt in names__target_params
    }


def raw_results__from_input_data(
    names__targets_from_input_data: QualNameTargetList,
    processed_data: QualNameData,
) -> QualNameData:
    return {ot: processed_data[ot] for ot in names__targets_from_input_data}


def raw_results__combined(
    raw_results__columns: QualNameData,
    raw_results__params: QualNameData,
    raw_results__from_input_data: QualNameData,
) -> QualNameData:
    return {
        **raw_results__columns,
        **raw_results__params,
        **raw_results__from_input_data,
    }
