(gep-3)=

# GEP 3 — Parameters of the taxes and transfers system

```{list-table}
- * Author
  * [Hans-Martin von Gaudecker](https://github.com/hmgaudecker)
- * Status
  * Provisional
- * Type
  * Standards Track
- * Created
  * 2022-03-28
- * Updated
  * 2025-06-xx
- * Resolution
  * [Accepted](https://gettsim.zulipchat.com/#narrow/stream/309998-GEPs/topic/GEP.2003)
```

## Abstract

This GEP describes the structure of the parameters of the taxes and transfers system.
This includes the format of the yaml files (initial input) and storage of the processed
parameters.

## Motivation and Scope

The parameters of the taxes and transfers system are among the core elements of GETTSIM.
Along with the functions operating on them and the input data, they determine all
quantities of interest. Sensibly structuring and cleanly documenting the meaning and
sources of these parameters requires particular care.

## Usage and Impact

GETTSIM developers should closely look at the Section {ref}`gep-3-structure-yaml-files`
before adding new parameters.

(gep-3-structure-yaml-files)=

## Structure of the YAML files

Each YAML file contains a number of parameters at the outermost level of indentation.
Each of these parameters in turn is a dictionary with at least three keys: `name`,
`description`, and the `YYYY-MM-DD`-formatted date on which it first took effect. Values
usually change over time; each time a value is changed, another `YYYY-MM-DD` entry is
added.

1. The `name` key has two sub-keys `de` and `en`, which are

   - short names without re-stating the realm of the parameter (e.g. "Arbeitslosengeld
     II" or "Kinderzuschlag");
   - not sentences;
   - correctly capitalised.

   Example (from `kindergeld`):

   ```yaml
   altersgrenze:
     name:
       de: Alter, ab dem Kindergeld nicht mehr gezahlt wird.
       en: Age at which child benefit is no longer paid.
   ```

1. The `description` key has two sub-keys `de` and `en`, which

   - are exhaustive explanations of the parameter;

   - show the § and Gesetzbuch of that parameter (including its entire history if the
     location has changed);

   - mention bigger amendments / Neufassungen and be as helpful as possible to make
     sense of that parameter.

     Example:

     ```yaml
     altersgrenze:
       description:
         de: >-
           § 32 Art. 2-4 EStG.
           Für minderjährige Kinder besteht ohne Bedingungen ein Anspruch auf Kindergeld.
           Auch für erwachsene Kinder kann bis zu einer Altersgrenze unter bestimmten
           Bedingungen ein Anspruch auf Kindergeld bestehen.
         en: >-
           § 32 Art. 2-4 EStG.
           Underage children are entitled to child benefit without any conditions. Also adult
           children up to a specified age are entitled to child benefit under certain
           conditions.
     ```

1. The `unit` key informs on the unit of the values (Euro or DM if monetary, share of
   some other value, ...).

   - In some cases (e.g., factor for the calculation of the marginal employment
     threshold), there is no unit.

   - It should be capitalised.

   - Possible values:

     - `Euros`,
     - `DM`,
     - `Share`,
     - `Percent`,
     - `Years`,
     - `Months`,
     - `Hours`,
     - `Square Meters`,
     - `Euros / Square Meter`,
     - *None*.

   Example:

   ```yaml
   altersgrenze:
     name:
       de: Alter, ab dem Kindergeld nicht mehr gezahlt wird.
     unit: Euro
   ```

1. The `type` key signals to GETTSIM how the parameter is to be interpreted. It must be
   specified as one of:

   - `scalar`,
   - `dict`,
   - `piecewise_constant`,
   - `piecewise_linear`,
   - `piecewise_quadratic`,
   - `piecewise_cubic`,
   - `require_converter`

   `scalar` is self-explanatory; `dict` must be a homogeneous dictionary with string or
   integer keys and scalar values (int, float, bool).

   `piecewise_constant`, `piecewise_linear`, `piecewise_quadratic`, `piecewise_cubic`
   will be converted automatically to be used with the `piecewise_polynomial` function.

   `require_converter` can be anything. However there must be a converter function in
   the codebase.

1. The `reference_period` key informs on the reference period of the values, if
   applicable. Possible values:

   - `Year`,
   - `Quarter`,
   - `Month`,
   - `Week`,
   - `Day`,
   - `Hour`,
   - *None*

1. The optional `add_jahresanfang` can be used to make the parameter that is relevant at
   the start of the year (relative to the date for which the policy environment is set
   up) available to GETTSIM functions.

   If specified, two parameters will be available:

   ```
   ("path", "to", "parameter")
   ("path", "to", "parameter_jahresanfang")
   ```

   Example from `arbeitslosengeld` / :

   ```yaml
   beitragssatz:
     name:
       de: Beitragssatz zur Arbeitslosenversicherung
     unit: Share
     reference_period: null
     type: scalar
     add_jahresanfang: true
   ```

1. The YYYY-MM-DD key(s)

   - hold all historical values for a specific parameter or set of parameters in
     dictionaries
   - contain a precise reference to the law in the `reference` subkey;
   - may add additional descriptions in the `note` key;
   - is present with a note or reference only if a parameter ceases to exist starting on
     a particular date;
   - in case of a `scalar` type, the key of the scalar is `value`.

   The remainder of this section explains this element in much more detail.

### The `reference` key of [YYYY-MM-DD]

- cites the law (Gesetz "G. v."), decree (Verordnung "V. v.") or proclamation
  (Bekanntmachung "B. v.") that changes the law
- uses German language
- follows the style
  - `Artikel [n] [G./V./B.] v. [DD.MM.YYYY] BGBl. I S. [SSSS].` for references published
    until 2022 (the page should be the first page of the law/decree/proclamation, not
    the exact page of the parameter)
  - `Artikel [n] [G./V./B.] v. [DD.MM.YYYY] BGBl. [YYYY] I Nr. [NNN].` for references
    from 2023 on
- does not add information "geändert durch" (it is always a change) or the date the law
  comes into force (this would just repeat the date key one level above)

Example:

```yaml
beitragssatz:
  name:
    de: Beitragssatz zur Arbeitslosenversicherung
  2019-01-01:
    value: 0.0125
    reference: V. v. 21.12.2018 BGBl. I S. 2663
```

### The `note` key of [YYYY-MM-DD]

This optional key may contain a free-form note holding any information that may be
relevant for the interpretation of the parameter, the implementer, user, ...

```yaml
beitragssatz:
  name:
    de: Beitragssatz zur Arbeitslosenversicherung
  2019-01-01:
    value: 0.0125
    reference: V. v. 21.12.2018 BGBl. I S. 2663
    note: >-
      Set to 0.013 in Art. 2 Nr. 15 G. v. 18.12.2018 BGBl. I S. 2651. Temporarily
      reduced to 0.0125 in BeiSaV 2019.
```

### The `updates_previous` key of [YYYY-MM-DD]

Often laws change only part of a parameter. To avoid error-prone code duplication, we
allow for such cases via `updates_previous` key.

This must not be used with a scalar parameter type. Furthermore, it cannot be used in
the first period a parameter is defined.

Example from `sozialversicherung` / `geringfügige_einkommen.yaml`:

```yaml
minijobgrenze_ost_west_unterschied
  name:
    de: Minijobgrenze
  unit: Euros
  reference_period: Month
  type: dict
  1997-01-01:
    west: 312
    ost: 266
  1998-01-01:
    updates_previous: true
    west: 317
```

### The values of [YYYY-MM-DD]

The general idea is to make the replication of the laws very obvious. If the law
includes a table, we will have a dictionary with keys 0, 1, 2, .... If the law includes
a formula, the formula should be included and its parameters referenced. Etc..

The following walks through several cases.

- The simplest case is a single parameter, which should be specified as:

  ```yaml
  kindergeld_stundengrenze:
    name:
      de: Wochenstundengrenze für Kindergeldanspruch
    2012-01-01:
      scalar: 20
  ```

- There could be a dictionary, potentially nested:

  ```yaml
  exmin:
    name:
      de: Höhen des Existenzminimums, festgelegt im Existenzminimumsbericht der Bundesregierung.
    2005-01-01:
      regelsatz:
        single: 4164
        paare: 7488
        kinder: 2688
      kosten_der_unterkunft:
        single: 2592
        paare: 3984
        kinder: 804
      heizkosten:
        single: 600
        paare: 768
        kinder: 156
  ```

- In some cases, a dictionary with numbered keys makes sense. It is important to use
  these, not lists!

  ```yaml
  kindergeld:
    name:
      de: Kindergeld, Betrag je nach Reihenfolge der Kinder.
    1975-01-01:
      1: 26
      2: 36
      3: 61
      4: 61
  ```

- Another example would be referring to the parameters of a piecewise linear function:

  > ```yaml
  > eink_anr_frei:
  >   name:
  >     de: Anrechnungsfreie Einkommensanteile
  >     en: Income shares not subject to transfer withdrawal
  >   type: piecewise_linear
  >   2005-01-01:
  >     0:
  >       lower_threshold: -inf
  >       upper_threshold: 0
  >       rate: 0
  >       intercept_at_lower_threshold: 0
  > ```

- In general, a parameter should appear for the first time that it is mentioned in a
  law, becomes relevant, etc..

  Only in exceptional cases it might be useful to set a parameter to some value
  (typically zero) even if it does not exist yet.

- If a parameter ceases to be relevant, is superseded by something else, etc., there
  must be a `YYYY-MM-DD` key with a note on this.

  Generally, this `YYYY-MM-DD` key will have an entry `scalar: null` regardless of the
  previous structure. Ideally, there would be a `reference` and potentially a `note`
  key. Example:

  ```yaml
  value: null
  note: arbeitslosenhilfe is superseded by arbeitslosengeld_2
  ```

  Only in exceptional cases it might be useful to set a parameter to some value
  (typically zero) even if it is not relevant any more.

  In any case, it **must** be the case that it is obvious from the `YYYY-MM-DD` entry
  that the (set of) parameter(s) is not relevant any more, else the previous ones will
  linger on.

(gep-3-handling-of-parameters-in-the-codebase)=

## Handling of parameters in the codebase

The contents of the YAML files become part of the `policy_params` dictionary. Its keys
correspond to the names of the YAML files. Each value will be a dictionary that follows
the structure of the YAML file. These values can be used in policy functions as
`[key]_params`.

The contents mostly follow the content of the YAML files. The main difference is that
all parameters are present in their required format; no further parsing shall be
necessary inside the functions. The important changes include:

- In the YAML files, parameters may be specified as deviations from other values,
  {ref}`see above <gep-3-deviation_from>`. All these are converted so that the relevant
  values are part of the dictionary.
- Similarly, values from the beginning of the year (via `access_different_date`,
  {ref}`see above <gep-3-access_different_date>`) of `[param]` will be available as:
  `[param]_[access_different_date]`.
- Parameters for piecewise polynomials are parsed.
- Parameters that are derived from other parameters are calculated (examples include
  `kinderzuschlag_max` starting in 2021 or calculating the phasing in of
  `vorsorgeaufwendungen_alter` over the 2005-2025 period).

These functions will be avaiable to users en bloque or one-by-one so they can specify
parameters as in the YAML file for their own policy parameters.

## Discussion

- <https://github.com/iza-institute-of-labor-economics/gettsim/pull/148>
- <https://gettsim.zulipchat.com/#narrow/stream/309998-GEPs/topic/GEP.2003>
- GitHub PR for update (changes because of `GEP-6 <gep-6>`):
  <https://github.com/iza-institute-of-labor-economics/gettsim/pull/855>

## Copyright

This document has been placed in the public domain.
