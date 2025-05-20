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

1. **One-stop-shop**

   Setting up a policy environment and computing the results used to require two steps.
   For many applications, this is all a user needs to do; it should be doable in a
   simpler way. The one-stop-shop (`oss`) will achieve this:

   ```python
   def oss(
       date: str,
       df: pd.DataFrame,
       inputs_tree_to_df_columns: NestedInputsPathsToDfColumns,
       targets_tree: NestedTargetDict,
   ) -> NestedDataDict:
       """One-stop-shop for computing taxes and transfers.

       Args:
           date:
               The date to compute taxes and transfers for. The date determines the policy
               environment for which the taxes and transfers are computed.
           df:
               The DataFrame containing the data.
           inputs_tree_to_df_columns:
               A nested dictionary that maps GETTSIM's expected input structure to the data
               provided by the user. Keys are strings that provide a path to an input.

               Values can be:
               - Strings that reference column names in the DataFrame.
               - Numeric or boolean values (which will be broadcasted to match the length
                 of the DataFrame).
           targets_tree:
               The targets tree.
   ```

   Example:

   ```pycon
   >>> inputs_tree_to_df_columns = {
   ...     "einkommensteuer": {
   ...         "gemeinsam_veranlagt": "joint_taxation",
   ...         "einkünfte": {
   ...             "aus_nichtselbstständiger_arbeit": {
   ...                 "bruttolohn_m": "gross_wage_m",
   ...             },
   ...         },
   ...     },
   ...     "alter": 30,
   ...     "p_id": "p_id",
   ... }
   >>> df = pd.DataFrame(
   ...     {
   ...         "gross_wage_m": [1000, 2000, 3000],
   ...         "joint_taxation": [True, True, False],
   ...         "p_id": [0, 1, 2],
   ...     }
   ... )
   >>> oss(
   ...     date="2024-01-01",
   ...     inputs_tree_to_df_columns=inputs_tree_to_df_columns,
   ...     targets_tree=targets_tree,
   ...     df=df,
   ... )
   ```

1. **DAG-based granular interface**

   On the other end of the spectrum, the interface so far was not flexible enough for
   advanced use cases. In particular, many checks were run time and again, slowing down
   the computation. The new interface will allow users to customize the graph to their
   needs.

   ```python

   ```

1. **Ecosystem**

   More functionality will be added in external packages. Check out:

   - [gettsim-personas](https://github.com/ttsim-dev/gettsim-personas): Pre-defined
     example personas ("Musterhaushalte")
   - [soep-preparation](https://github.com/ttsim-dev/soep-preparation): A pipeline
     preparing the SOEP data for use with GETTSIM

1. **Interactive Graph Interface**

   ```python
   # Create interactive graph visualization
   graph = gettsim.visualize_graph(target_variables=["kindergeld"], interactive=True)

   # Export selected variables as template
   template = graph.export_template()
   ```

### Benefits

1. **Simplified Onboarding**: New users can start with pre-defined personas and
   gradually customize their inputs. They may use the SOEP preparation pipeline directly
   or compare their own version of the SOEP with the outputs generated by that.
1. **Flexible Data Mapping**: Users can map their existing datasets to GETTSIM's
   structure without modifying their data.
1. **Interactive Exploration**: Visual interface allows users to explore dependencies
   and select relevant variables.

## Backward Compatibility

This interface represents a significant change. Backward compatibility can be achieved,
however, by specifying the currently used columns as the `inputs_tree_to_df_columns`
argument in the `oss` function.

## Detailed Description

Most of the elements are self-explanatory based on the examples above or do not quite
belong here, like the components of the ecosystem.

### DAG-based granular interface

## Discussion

- Related Issue: #781 - Interface discussion from 2024 GETTSIM workshop
- Related PR: #787 - Model classes implementation
- Zulip Discussion: [Link to discussion thread]

## Copyright

This document has been placed in the public domain.

### References

1. **ENH: Interface, 2024 edition · Issue #781 ·
   iza-institute-of-labor-economics/gettsim - Part 1**.
   [https://github.com](https://github.com/iza-institute-of-labor-economics/gettsim/issues/781)
