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
experienced users:

1. **High Entry Barrier**: Users need detailed knowledge of the Directed Acyclic Graph
   (DAG) structure and precise input requirements, making it difficult for newcomers to
   get started.

1. **Data Mapping Complexity**: Matching existing datasets with GETTSIM's requirements
   is challenging due to the fine-grained nature of the graph.

1. **Limited Flexibility**: The current interface makes it difficult to work with
   different datasets / areas of the taxes and transfers system.

This GEP aims to address these issues by introducing a more intuitive and flexible
interface while maintaining GETTSIM's computational robustness.

## Usage and Impact

### New Interface Components

1. **A single entry point**

   The `main` function is the single entry point for all GETTSIM functionality. For
   computing taxes and transfers, the minimal requirements are:

   - The output(s) to compute (various options)
   - The date at which the policy environment is set up and evaluated
   - Data on individuals / households in one of various formats
   - Typically, the set of targets to compute (could be left out, in which case all
     targets are computed)

   Here is an example (the variables `inputs_df`, `inputs_map`, and `targets_tree` will
   be shown below).

   ```python
   from gettsim import InputData, Output, Targets, main

   outputs_df = main(
       output=Output.name(("results", "df_with_mapper")),
       date_str="2025-01-01",
       input_data=InputData.df_and_mapper(
           df=inputs_df,
           mapper=inputs_map,
       ),
       targets=Targets(tree=targets_tree),
   )
   ```

   All elements that are not atomic are specified as GETTSIM objects, which means that
   users can benefit from autocompletion and type hints provided by their IDE.

   The first argument, `output`, specifies the output to compute. In this case, we want
   the "results" in the "DataFrame with mapper" format. That is, GETTSIM will compute
   all desired targets and return a DataFrame with columns specified by the user.

   Say we want to compute the contributions to long term care insurance
   (Pflegeversicherung). The fourth argument, `targets`, specifies the set of targets to
   compute. Because we ask for the "results" in the "DataFrame with mapper" format, this
   actually has to be a mapping from the targets to the columns in the output DataFrame.
   In this case, the argument `targets` needs to be a *pytree*, which provides the
   mapping:

   ```python
   targets_tree = {
       "sozialversicherung": {
           "pflege": {
               "beitrag": {
                   "betrag_versicherter_m": "ltci_contrib",
               }
           }
       }
   }
   ```

   That is, the call to `main` above will return a DataFrame with one column
   `ltci_contrib` and of the same length as the input data.

   The second argument, `date_str`, specifies the date at which the policy environment
   is set up and evaluated.

   Say we want to compute the long term care insurance contribution for three people,
   one of whom has an underage child living in her household. Our data looks as follows:

   <table border="1" class="dataframe">
     <thead>
       <tr style="text-align: right;">
         <th></th>
         <th>age</th>
         <th>wage</th>
         <th>id</th>
         <th>hh_id</th>
         <th>mother_id</th>
         <th>has_kids</th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <th>0</th>
         <td>25</td>
         <td>950</td>
         <td>0</td>
         <td>0</td>
         <td>-1</td>
         <td>False</td>
       </tr>
       <tr>
         <th>1</th>
         <td>45</td>
         <td>950</td>
         <td>1</td>
         <td>1</td>
         <td>-1</td>
         <td>True</td>
       </tr>
       <tr>
         <th>2</th>
         <td>3</td>
         <td>0</td>
         <td>2</td>
         <td>1</td>
         <td>1</td>
         <td>False</td>
       </tr>
       <tr>
         <th>3</th>
         <td>65</td>
         <td>950</td>
         <td>3</td>
         <td>2</td>
         <td>-1</td>
         <td>True</td>
       </tr>
     </tbody>
   </table>

   We can use this DataFrame directly, however, we need to tell GETTSIM how to map the
   columns of the DataFrame to the columns it knows about, This is done by a mapper,
   which is a *pytree* that in our case looks as follows

   ```python
   inputs_map = {
       "p_id": "id",
       "hh_id": "hh_id",
       "alter": "age",
       "familie": {
           "p_id_elternteil_1": "mother_id",
           "p_id_elternteil_2": -1,
       },
       "einkommensteuer": {
           "einkünfte": {
               "aus_nichtselbstständiger_arbeit": {"bruttolohn_m": "wage"},
               "ist_selbstständig": False,
               "aus_selbstständiger_arbeit": {"betrag_m": 0.0},
           }
       },
       "sozialversicherung": {
           "pflege": {
               "beitrag": {
                   "hat_kinder": "has_kids",
               }
           },
           "kranken": {
               "beitrag": {"bemessungsgrundlage_rente_m": 0.0, "privat_versichert": False}
           },
       },
   }
   ```

   Note that we set several variables to scalars. E.g., we do not consider self-employed
   people, pensioners, or people with (substitutive) private health insurance.

   *Note:* We picked an example with little, but not zero, complexity. The amount of
   inputs is simply necessary because public long term care insurance contributions
   depend on various kinds of income (from dependent employment, from self-employment,
   pensions), the combination of the insured person's age and her children, and whether
   the insured person is covered by private health insurance.

   Finally, here is the output of our example:

   <table border="1" class="dataframe">
     <thead>
       <tr style="text-align: right;">
         <th></th>
         <th>ltci_contrib</th>
       </tr>
       <tr>
         <th>p_id</th>
         <th></th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <th>0</th>
         <td>14.72</td>
       </tr>
       <tr>
         <th>1</th>
         <td>9.82</td>
       </tr>
       <tr>
         <th>2</th>
         <td>0.00</td>
       </tr>
       <tr>
         <th>3</th>
         <td>9.82</td>
       </tr>
     </tbody>
   </table>

1. **DAG-based granular interface**

   On the other end of the spectrum, the interface so far was not flexible enough for
   advanced use cases. In particular, many checks were run time and again, slowing down
   the computation. The updated interface will allow users to customize the graph to
   their needs.

   ...

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
however, when ...

## Discussion

- **ENH: Interface, 2024 edition · Issue #781 · iza-institute-of-labor-economics/gettsim
  \- Part 1**.
  [https://github.com](https://github.com/iza-institute-of-labor-economics/gettsim/issues/781)

## Copyright

This document has been placed in the public domain.

### References
