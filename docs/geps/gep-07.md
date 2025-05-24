(gep-7)=

# GEP 7 — GETTSIM's User Interface

```{list-table}
- * Author
  * [Hans-Martin von Gaudecker](https://github.com/hmgaudecker)
- * Status
  * Draft
- * Type
  * Standards Track
- * Created
  * 2025-04-11
- * Resolution
  * One of [Accepted | Rejected |
    Withdrawn](<https://gettsim.zulipchat.com/#narrow/stream/309998-GEPs/topic/GEP.XXX>)
    once resolution has been found
```

## Abstract

This GEP proposes a new user interface for GETTSIM that simplifies data input/output
handling, reduces the learning curve for new users, and provides more flexibility in
working with different datasets. The interface redesign aims to address the challenges
identified during the 2024 GETTSIM workshop and, more generally, experience with using
GETTSIM versions up to 0.7.0. At the same time, we maintain compatibility with the
namespace architecture introduced in GEP 6.

## Motivation and Scope

The current GETTSIM interface presents several challenges that affect both new and
experienced users \[[1]\]:

1. **High Entry Barrier**: Users need detailed knowledge of the Directed Acyclic Graph
   (DAG) structure and precise input requirements, making it difficult for newcomers to
   get started.

1. **Data Mapping Complexity**: Matching existing datasets with GETTSIM's requirements
   is challenging due to the fine-grained nature of the graph.

1. **Limited Flexibility**: The current interface makes it difficult to work with
   different datasets / areas of the

This GEP aims to address these issues by introducing a more intuitive and flexible
interface while maintaining GETTSIM's computational robustness.

## Usage and Impact

### New Interface Components

1. **Template-Based Data Input**

   ```python
   from gettsim import get_input_template, set_up_policy_environment


   get_input_template(
       target_variables=["kindergeld", "einkommensteuer"],
       policy_environment=set_up_policy_environment("2025-01-01"),
   )
   ```

   A yaml-representation of the template is:

   ```yaml
   p_id: int
   p_id_kindergeldempfänger: bool
   p_id_ehepartner: bool
   einkommensteuer:
     gemeinsam_veranlagt: bool
   einkommen:
     aus_abhängiger_beschäftigung:
       bruttolohn_y: float
   ```

   Users may then replace the type hints in the template by the column name in a dataset
   they will provide.

   ```yaml
   p_id: p_id
   p_id_kindergeldempfänger: mother_id
   p_id_ehepartner: married_spouse_id
   einkommensteuer:
     gemeinsam_veranlagt: files_jointly
   einkommen:
     aus_abhängiger_beschäftigung:
       bruttolohn_y: earnings
   ```

   *Note:* The output of `get_input_template` can easily become quite daunting because
   it shows the root nodes of the graph. E.g., for many static labour supply
   applications, one does not need to know how to calculate pensions because retirees
   are excluded from the sample. To simplify the input by just setting all pension
   payments to zero, it is often easiest to look at a visual representation of the DAG,
   see below.

1. **One-stop-shop**

   Setting up a policy environment and computing the results used to require two steps.
   For many applications, this is all a user needs to do; it should be doable in a
   simpler way. The one-stop-shop (`oss`) will achieve this:

   ```python
   def oss(
       date: str,
       inputs_df: pd.DataFrame,
       inputs_tree_to_df_columns: NestedInputsPathsToDfColumns,
       targets_tree: NestedTargetDict,
   ) -> NestedDataDict:
       """One-stop-shop for computing taxes and transfers.

       Args:
           date:
               The date to compute taxes and transfers for. The date determines the policy
               environment for which the taxes and transfers are computed.
           inputs_df:
               The DataFrame containing the data.
           inputs_tree_to_inputs_df_columns:
               A tree that has the inputs required by GETTSIM as the path (sequence of
               keys) and maps them to the data provided by the user. The leaves of the tree
               are strings that reference column names in *inputs_df* or constants.
           targets_tree_to_df_columns:
               A tree that has the desired targets as the path (sequence of keys) and maps
               them to the data columns the user would like to have.
   ```

   These are the absolute minimal requirements a computation needs:

   - The date at which the policy environment is set up and evaluated
   - The data on individuals / households in standard dataframe format, i.e., a
     2-dimensional table
   - A *pytree* mapping GETTSIM's expected input structure to columns in the dataframe
     or constants (e.g., when there are no pensioners in the data, pension payments can
     be quickly set to zero for everyone)
   - A *pytree* mapping the desired targets to columns they should be called in the
     output dataframe

   As an example, let us calculate long term care insurance (Pflegeversicherung)
   contributions for three people, one of whom has an underage child living in her
   household:

   ```pycon
   >> from gettsim import oss
   >>> inputs_df = pd.DataFrame(
   ...     {
   ...         "age": [25, 45, 3, 65],
   ...         "wage": [950, 950, 0, 950],
   ...         "id": [0, 1, 2, 3],
   ...         "hh_id": [0, 1, 1, 2],
   ...         "mother_id": [-1, -1, 1, -1],
   ...         "has_kids": [False, True, False, True],
   ...     }
   ... )
   >>> inputs_map = {
   ...     "p_id": "id",
   ...     "hh_id": "hh_id",
   ...     "alter": "age",
   ...     "familie":{
   ...         "p_id_elternteil_1": "mother_id",
   ...         "p_id_elternteil_2": -1,
   ...     },
   ...     "einkommensteuer": {
   ...         "einkünfte": {
   ...             "aus_nichtselbstständiger_arbeit": {"bruttolohn_m": "wage"},
   ...             "ist_selbstständig": False,
   ...             "aus_selbstständiger_arbeit": {"betrag_m": 0.0},
   ...         }
   ...     },
   ...     "sozialversicherung": {
   ...         "pflege": {
   ...             "beitrag": {
   ...                 "hat_kinder": "has_kids",
   ...             }
   ...         },
   ...         "kranken": {
   ...             "beitrag":{
   ...                 "bemessungsgrundlage_rente_m": 0.0,
   ...                 "privat_versichert": False
   ...             }
   ...         }
   ...     },
   ... }
   >>> targets_map={
   ...        "sozialversicherung": {
   ...            "pflege": {
   ...                "beitrag": {
   ...                    "betrag_versicherter_m": "ltci_contrib",
   ...                }
   ...            }
   ...        }
   ...    }
   >>> oss(
   ...     date="2025-01-01",
   ...     inputs_df=inputs_df,
   ...     inputs_tree_to_inputs_df_columns=inputs_map,
   ...     targets_tree_to_outputs_df_columns=targets_map,
   ... )
      ltci_contrib
   0         14.72
   1          9.82
   2          0.00
   3          9.82
   ```

   *Note:* We picked an example with little, but not zero, complexity. The amount of
   inputs is simply necessary because public long term care insurance contributions
   depend on various kinds of income (from dependent employment, from self-employment,
   pensions), the combination of the insured person's age and her children, and whether
   the insured person is covered by private health insurance.

1. **DAG-based granular interface**

   On the other end of the spectrum, the interface so far was not flexible enough for
   advanced use cases. In particular, many checks were run time and again, slowing down
   the computation. The new interface will allow users to customize the graph to their
   needs.

   There will be two basic steps to this:

   1. Create a custom function to run.
   1. Run this function.

   Once more, this is powered by `dags`, so the custom function can be freely composed
   from many elements. The simplest example is to recreate the above by just requesting
   `outputs_df` as the final target, along with the checks included by default in `oss`:

   ```python
   get_outputs_df = gettsim.concatenate_functions(
       targets=["outputs_df", "fail_if_data_is_invalid"],
   )
   targets_df = get_outputs_df(
       policy_date="2025-01-01",
       evaluation_date="2025-01-01",
       inputs_df=inputs_df,
       inputs_tree_to_inputs_df_columns=inputs_map,
       targets_tree_to_outputs_df_columns=targets_map,
   )["outputs_df"]
   ```

   In fact, the `oss` function is just a wrapper around this.

   The flexibility comes because there are many intermediate targets, all of which can
   be requested. It is also possible to do things one after the other. For example, to
   increase the contribution rate to long term care insurance of people above 23 who do
   not have children by 1 percentage point, one could do:

   ```python
   # Set up the base environment
   get_policy_environment = gettsim.concatenate_functions(
       targets=["policy_environment", "set_up_policy_environment"],
   )
   base_environment = get_policy_environment(policy_date="2025-01-01")
   # Modify
   modified_environment = copy.deepcopy(base_environment)
   modified_environment.params_tree["sozialversicherung"]["pflege"]["beitrag"][
       "beitragssatz_nach_kinderzahl"
   ].value["zusatz_kinderlos"] += 0.01
   # Compute the results
   get_outputs_df = gettsim.concatenate_functions(
       targets=["outputs_df"],
       policy_environment=modified_environment,
   )
   targets_df = get_outputs_df(
       evaluation_date="2025-01-01",
       inputs_df=inputs_df,
       inputs_tree_to_inputs_df_columns=inputs_map,
       targets_tree_to_outputs_df_columns=targets_map,
   )["outputs_df"]
   ```

   An incomplete and tentative list of nodes *(current names in the code, writing them
   down shows that they need to change...)*:

   - `policy_environment`
   - `fail_if...`
   - `data_tree`
   - `processed_params_tree`
   - `function_targets`
   - `taxes_and_transfers_function`
   - `warn_if_functions_overridden_by_data`
   - `targets_tree`
   - `outputs_df`

   Inputs with default values:

   - `inputs_df`
   - `inputs_tree_to_inputs_df_columns`
   - `targets_tree_to_outputs_df_columns`
   - `rounding=True`
   - `debug=False`
   - `jit=False`

1. **Ecosystem**

   More functionality will be added in external packages. Check out:

   - [gettsim-personas](https://github.com/ttsim-dev/gettsim-personas): Pre-defined
     example personas ("Musterhaushalte")
   - [soep-preparation](https://github.com/ttsim-dev/soep-preparation): A pipeline
     preparing the SOEP data for use with GETTSIM

1. **Interactive Graph Interface**

   We focus on the infrastructure for the moment; this will be easy to add and will
   require a much more interactive and user-driven approach. Top-down planning does not
   seem useful at this point.

## Backward Compatibility

This interface represents a significant change. Transition should be very smooth,
however, when specifying the currently used columns as the `inputs_tree_to_df_columns`
argument in the `oss` function.

## Discussion

- **ENH: Interface, 2024 edition · Issue #781 · iza-institute-of-labor-economics/gettsim
  \- Part 1**.
  [https://github.com](https://github.com/iza-institute-of-labor-economics/gettsim/issues/781)

## Copyright

This document has been placed in the public domain.

### References
