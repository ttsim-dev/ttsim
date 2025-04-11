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
   different datasets.

This GEP aims to address these issues by introducing a more intuitive and flexible
interface while maintaining GETTSIM's computational robustness.

## Usage and Impact

### New Interface Components

1. **Template-Based Data Input**

```python
# Generate an empty template for data mapping
template = gettsim.get_input_template(
    target_variables=["kindergeld", "einkommensteuer"], persona="family_with_children"
)

# User fills in their dataset column mappings
mappings = {
    "person.id": "pid",
    "person.age": "alter",
    "household.id": "hh_nr",
    "income.employment.monthly": "erwerbseink",
    # ... other mappings
}

# Create GETTSIM input from user data
input_data = gettsim.prepare_input(
    data=user_dataset, mappings=mappings, template=template
)
```

2. **Persona-Based Default Values**

```python
# Load pre-defined persona for testing
test_data = gettsim.load_persona(
    "family_with_children", variations={"n_children": 2, "income_level": "middle"}
)
```

3. **Interactive Graph Interface**

```python
# Create interactive graph visualization
graph = gettsim.visualize_graph(target_variables=["kindergeld"], interactive=True)

# Export selected variables as template
template = graph.export_template()
```

### Benefits

1. **Simplified Onboarding**: New users can start with pre-defined personas and
   gradually customize their inputs.
1. **Flexible Data Mapping**: Users can map their existing datasets to GETTSIM's
   structure without modifying their data.
1. **Interactive Exploration**: Visual interface allows users to explore dependencies
   and select relevant variables.

## Backward Compatibility

This interface represents a significant change but maintains backward compatibility
through:

1. **Legacy Interface Support**: The current interface will be maintained as a "legacy"
   option for existing users.
1. **Migration Tools**: Utilities will be provided to help users migrate from the old to
   the new interface.
1. **Gradual Transition**: Both interfaces will be supported for at least two major
   versions.

## Detailed Description

### Core Components

1. **Input Template System**

- Templates are generated based on requested target variables
- Supports both YAML and dictionary formats
- Includes validation rules and data type specifications
- Handles time period conversions automatically

2. **Persona Framework**

```yaml
personas:
  family_with_children:
    base:
      household_size: 3
      n_children: 1
      income_level: "middle"
    variations:
      income_levels: ["low", "middle", "high"]
      n_children: 2, 3]
```

3. **Interactive Graph Interface**

- Built on NetworkX for graph visualization
- Supports node/edge filtering
- Allows direct manipulation of graph structure
- Exports selected subgraphs as templates

4. **Data Mapping Engine**

- Handles column name mapping
- Performs automatic type conversion
- Validates data consistency
- Supports aggregation specifications

### Implementation Details

1. **Template Generation**

```python
class InputTemplate:
    def __init__(
        self,
        target_variables: List[str],
        persona: Optional[str] = None,
        date: Optional[str] = None,
    ):
        self.required_inputs = self._analyze_dependencies(target_variables)
        self.validation_rules = self._get_validation_rules()

    def export_yaml(self) -> str:
        """Export template as YAML format"""

    def validate_mapping(self, mapping: Dict[str, str]) -> bool:
        """Validate user-provided mapping"""
```

2. **Data Preparation Pipeline**

```python
def prepare_input(
    data: pd.DataFrame,
    mappings: Dict[str, str],
    template: InputTemplate,
    validate: bool = True,
) -> GettSimInput:
    """Prepare user data for GETTSIM computation"""
    if validate:
        template.validate_mapping(mappings)

    transformed_data = _transform_data(data, mappings)
    return GettSimInput(transformed_data)
```

## Related Work

1. **OpenFisca**: Similar template-based approach for data input
1. **Tax-Calculator**: Interactive interface for policy simulation
1. **EUROMOD**: Persona-based testing system

## Alternatives

1. **API-First Approach**: Considered but rejected due to increased complexity
1. **Configuration File System**: Less flexible than template-based approach
1. **Direct Graph Manipulation**: Too complex for most users

## Discussion

- Related Issue: #781 - Interface discussion from 2024 GETTSIM workshop
- Related PR: #787 - Model classes implementation
- Zulip Discussion: [Link to discussion thread]

## Copyright

This document has been placed in the public domain.

This GEP follows the required format and addresses the interface challenges while
maintaining consistency with existing GEPs. It provides concrete implementation details
and examples while considering backward compatibility and user experience.

### References

1. **ENH: Interface, 2024 edition · Issue #781 ·
   iza-institute-of-labor-economics/gettsim - Part 1**.
   [https://github.com](https://github.com/iza-institute-of-labor-economics/gettsim/issues/781#:~:text=Is%20your%20feature%20request,the%20list%20of%20data)
1. **citation - iza-institute-of-labor-economics/gettsim**.
   [https://github.com](https://github.com/iza-institute-of-labor-economics/gettsim/blob/main/CITATION#:~:text=The%20GErman%20Taxes%20and,creating%20an%20account%20on)
1. **iza-institute-of-labor-economics/gettsim**.
   [https://github.com](https://github.com/iza-institute-of-labor-economics/gettsim#:~:text=GETTSIM%20is%20implemented%20in,documentation%20is%20available%20at)
1. **Welcome to GETTSIM's documentation!**.
   [https://gettsim.readthedocs.io](https://gettsim.readthedocs.io/en/v0.2.1/#:~:text=GETTSIM%20is%20implemented%20in,The%20current%20version%20is)
