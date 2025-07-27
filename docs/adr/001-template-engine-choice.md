# ADR-001: Template Engine Choice for Model Card Generation

## Status
Accepted

## Context
The Model Card as Code Generator needs a flexible template engine to support multiple output formats (Hugging Face, Google Model Cards, EU CRA). The template engine must handle:

- Complex data structures and nested objects
- Conditional logic for optional sections
- Template inheritance and composition
- Multiple output formats (Markdown, JSON, HTML)
- Custom template creation by users

### Options Considered

#### Option 1: Jinja2
**Pros:**
- Mature, well-documented Python template engine
- Rich feature set with inheritance, macros, filters
- Large community and extensive ecosystem
- Built-in security features (auto-escaping)
- Excellent IDE support

**Cons:**
- Learning curve for complex templates
- Performance overhead for simple use cases

#### Option 2: Mustache (pystache)
**Pros:**
- Logic-less templates (simpler mental model)
- Language-agnostic (templates work across platforms)
- Good performance for simple cases
- Minimal learning curve

**Cons:**
- Limited conditional logic capabilities
- No template inheritance
- Requires complex data preparation for advanced features

#### Option 3: Custom Template System
**Pros:**
- Perfect fit for specific requirements
- Maximum performance optimization
- Complete control over features

**Cons:**
- High development and maintenance cost
- Reinventing the wheel
- Limited community support
- Testing and security burden

#### Option 4: f-strings + String Templates
**Pros:**
- Native Python feature
- Zero external dependencies
- Maximum performance
- Simple debugging

**Cons:**
- No advanced templating features
- Poor separation of logic and presentation
- Difficult to maintain for complex templates

## Decision
We will use **Jinja2** as the primary template engine for the Model Card Generator.

## Rationale

### Technical Factors
1. **Feature Completeness**: Jinja2 provides all required features including inheritance, macros, filters, and conditional logic
2. **Security**: Built-in auto-escaping prevents injection attacks
3. **Performance**: Compiled templates offer good performance for our use cases
4. **Ecosystem**: Large collection of existing filters and extensions

### Business Factors
1. **Developer Familiarity**: Most Python developers know Jinja2
2. **Maintenance**: Well-maintained with active community
3. **Documentation**: Excellent documentation reduces onboarding time
4. **Flexibility**: Supports our multi-format requirements

### Implementation Benefits
1. **Template Inheritance**: Base templates with format-specific overrides
2. **Macros**: Reusable components for common sections
3. **Filters**: Custom data formatting (e.g., metric formatting)
4. **Conditional Logic**: Handle optional sections elegantly

## Implementation Strategy

### Template Structure
```
templates/
├── base/
│   ├── model_card.j2           # Base template
│   ├── sections/
│   │   ├── model_details.j2
│   │   ├── evaluation.j2
│   │   └── ethical_considerations.j2
├── formats/
│   ├── huggingface.j2          # Extends base
│   ├── google.j2               # Extends base
│   └── eu_cra.j2              # Extends base
└── custom/
    └── user_templates/
```

### Custom Filters
```python
@jinja2_env.filter
def format_metric(value, precision=3):
    """Format numeric metrics with appropriate precision"""
    return f"{value:.{precision}f}"

@jinja2_env.filter
def compliance_badge(standard):
    """Generate compliance badge for given standard"""
    return f"![{standard}](https://img.shields.io/badge/{standard}-compliant-green)"
```

### Template Example
```jinja2
{# Base template: templates/base/model_card.j2 #}
# {{ model_name }}

{% block model_details %}
{% include 'sections/model_details.j2' %}
{% endblock %}

{% block evaluation %}
{% if evaluation_results %}
{% include 'sections/evaluation.j2' %}
{% endif %}
{% endblock %}

{% block compliance %}
{# Override in format-specific templates #}
{% endblock %}
```

## Consequences

### Positive
- **Rapid Development**: Existing Jinja2 knowledge accelerates development
- **Template Reuse**: Base templates reduce duplication across formats
- **User Extensibility**: Users can create custom templates easily
- **Maintainability**: Clear separation between logic and presentation

### Negative
- **Dependency**: Adds external dependency (acceptable trade-off)
- **Learning Curve**: Users creating custom templates need Jinja2 knowledge
- **Template Debugging**: Template errors can be harder to debug than Python code

### Mitigation Strategies
- **Template Validation**: Pre-validate templates before rendering
- **Error Handling**: Provide clear error messages for template issues
- **Documentation**: Comprehensive template creation guide
- **Examples**: Rich library of template examples and patterns

## Alternatives Considered
If Jinja2 proves insufficient in the future, we could:
- Add Mustache support for simple templates
- Implement hybrid approach with multiple template engines
- Create domain-specific template language for model cards

## Review Date
This decision will be reviewed in 6 months (July 2025) or when significant new requirements emerge.