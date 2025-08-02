from __future__ import annotations

import numpy as np
from ttsim.main import main
from ttsim.main_target import MainTarget
from ttsim.main_args import InputData
from ttsim.tt import policy_function, policy_input, param_function, ConsecutiveIntLookupTableParamValue
from ttsim import copy_environment
import mettsim.middle_earth as middle_earth

print("Starting debug test script... (RELEVANT INPUT DATA)")

result = main(
    main_target=MainTarget.templates.input_data_dtypes,
    policy_date_str="2000-01-01",
    orig_policy_objects={"root": middle_earth.ROOT_PATH},
    tt_targets={"tree": {"wealth_tax": {"amount_y": None}}},
    input_data=InputData.tree(
        tree={
            "p_id": np.array([0]),
            "wealth_tax": {
                "exempt_from_wealth_tax": np.array([True]),
            },
        }
    ),
)

print("\n" + "="*60)
print("FINAL SCRIPT RESULT (RELEVANT INPUT DATA):")
print("="*60)
print(result)
