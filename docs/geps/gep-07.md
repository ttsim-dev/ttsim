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

1. Basic workflow

   There is a single entry point for GETTSIM: The `main` function. It is powered by a
   DAG in the background.

   This means that the user will have to start by telling it the desired target
   ("`main_target`") or set of targets ("`main_targets`"). Ultimately, the main target
   will typically be a dataset with values for taxes and transfers. However,
   intermediate steps can be useful. E. g., the taxes and transfers system at a
   particular date (the "`policy_environ­ment`"), which she wants to modify in order to
   model a reform.

   The targets determine the required inputs. For example, in order to compute taxes and
   transfers for a set of households, one will need as primitives

   - data on these households ("`input_data`")
   - the set of taxes and transfers to be computed ("`tt_targets`"). This could be left
     out, in which case all possible targets will be computed. However, that will often
     be a daunting task in terms of requirements on the data and computer memory.
   - the date

   If only a policy environment is to be returned, just the date is required as an
   input.

1. **Worked example**

   Here is an example (the variables `inputs_df`, `inputs_map`, and `targets_tree` will
   be shown below).

   ```python
   from gettsim import InputData, MainTarget, TTTargets, main


   outputs_df = main(
       main_target=MainTarget.results.df_with_mapper,
       date_str="2025-01-01",
       input_data=InputData.df_and_mapper(
           df=inputs_df,
           mapper=inputs_map,
       ),
       tt_targets=TTTargets(tree=targets_tree),
   )
   ```

   All elements that are not atomic are specified as GETTSIM objects, which means that
   users can benefit from autocompletion and type hints provided by their IDE (see
   below).

   The first argument, `main_target`, specifies the type of object to compute. In this
   case, we want the "results" in the "DataFrame with mapper" format. That is, GETTSIM
   will compute all desired targets and return a DataFrame with columns specified by the
   user.

   Say we want to compute the contributions to long term care insurance
   (Pflegeversicherung). The fourth argument, `tt_targets`, specifies the set of taxes
   and transfers ("`tt`") to compute. Because we ask for the "results" in the "DataFrame
   with mapper" format, this actually has to be a mapping from the targets to the
   columns in the output DataFrame. In this case, the argument `tt_targets` needs to be
   a *pytree*, which provides that mapping:

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
   `ltci_contrib`, which will be of the same length as the input data.

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

   We can use this DataFrame directly. All we need to do is to tell GETTSIM how to map
   the columns of that DataFrame to the names of inputs it knows about. This is done by
   a *mapper*, which again is a *pytree*. In our case, it looks as follows:

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

   All *leaves* of the tree are either column names in the data or scalars. E.g., we do
   not consider self-employed people, pensioners, or people with (substitutive) private
   health insurance. Instead of requiring some default value in the data, we can simply
   use a scalar value in the mapper.

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

1. **Underlying structure**

   The interface DAG looks as follows:

   .. todo::

   Embed the interface DAG without fail/warn nodes as described here:
   https://mystmd.org/guide/reuse-jupyter-outputs

   The **policy environment** consists of all functions relevant at some point in time.
   E.g., when requesting a policy environment for some date in the 2020s, Erziehungsgeld
   will not be part of it because it was replaced by Elterngeld long before. Users
   wishing to implement reforms—whether they consist of changing parameter values or
   replacing functions—will do so at the level of the policy environment.

   The **input data** are the data provided by the user. They need to be passed in one
   of several forms.

   The **processed data** are a version of the input data that can be consumed by
   downstream functions, which includes conversions of identifiers.

   Users specify the **taxes and transfers targets** ("`tt_targets`"), which GETTSIM
   will return. When left out, GETTSIM will return all functions it can compute.

   The elements of the **specialized environment** combine policy environment and data.
   Even if users do not typically need to work with these elements, they are so central
   to GETTSIM that it is useful to list them here.

   - **with derived functions and without tree logic** adds aggregations (e.g., adding
     up individual incomes to income at the Steuernummer level) and time conversions
     (e.g., from month to year). Doing so requires knowing the names of the columns in
     the data.
   - **with processed params and scalars**. The parameters of the taxes and transfers
     system are stored in special objects. Some of them require further conversion
     through functions that do not depend on household data ("`param_functions`").
     Similarly, it is possible to pass scalars instead of data columns for things that
     are not observed in a dataset or that can be assumed constant in a particular
     application (e.g., setting pension payments to zero when looking at the labor
     supply of 30-year olds). In this step, these functions are run and all parameters
     are converted to their final form (e.g., a `ScalarParam` becomes just a number).
     Where relevant, policy functions are replaced by scalers passed as input data.
   - **with partialled params and scalars** partials all parameters and scalars to the
     functions that make up the taxes and transfers DAG. That is, the resulting
     functions only depend on column arguments (either passed as input data or computed
     earlier in the DAG).
   - **taxes and transfers DAG** is the DAG representation of the functions in the
     previous step.
   - **taxes and transfers function** is the function that takes the columns in the
     processed data as arguments and returns the desired taxes and transfers targets.
     Running this function leads to raw results (there still contain internals and
     should not be used by non-GETTSIM functions)

   The **results** contain the output of the taxes and transfers function, purged of
   internals and converted to the format requested by the user.

   The German taxes and transfers system is complex and specifying inputs can be a
   daunting task. The **templates** aim to help with that. E.g., asking for
   `MainTarget.templates.input_data` will return a nested dictionary that may include:

   ```python
   {
       "einkommensteuer": {
           "einkünfte": {"aus_forst_und_landwirtschaft": {"betrag_y": "FloatColumn"}}
       }
   }
   ```

   These templates can be modified to become a mapper, as in the above example. That is,
   "`FloatColumn`" could be replaced by the name of the column in the input data frame
   or by 0.0 if only employees who don't have other types of income are in the sample.

1. **Autocompletion features**

   The internal structure of the building blocks described in the previous section can
   be rather complex. In order to minimize errors arising from typos and misconceptions,
   GETTSIM provides objects that allow to take advantage of modern IDEs'/editors'
   autocompletion and type hinting features.

   For example, after:

   ```python
   from gettsim import main, MainTarget

   main(main_target=MainTarget.)
   ```

   tools like VS Code will show the options:

   ```python
   results
   policy_environment
   templates
   orig_policy_objects
   specialized_environment
   processed_data
   raw_results
   labels
   input_data
   tt_targets
   backend
   date_str
   date
   evaluation_date_str
   evaluation_date
   policy_date_str
   policy_date
   xnp
   dnp
   num_segments
   rounding
   warn_if
   fail_if
   ```

   Such objects are provided for all arguments to main that need a hierarchical
   structure. E. g. , the 'input-data' argument takes an instance of 'luput Data' like
   in the above example. Again, one will be able to benefit from auto completion
   features from typing the first 'l' onwards.

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
