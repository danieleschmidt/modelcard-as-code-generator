# modelcard-as-code-generator

Generate standardised ML Model Cards from YAML/JSON config files.

Model cards document what a model does, how it was trained, what it was evaluated on, and where it may fail — following the [HuggingFace](https://huggingface.co/docs/hub/model-cards) and [Google Model Card](https://modelcards.withgoogle.com/) formats.

This tool treats model cards as **code**: version-controlled, schema-validated, diffable config files that generate consistent documentation automatically.

---

## Features

- **YAML/JSON config → Markdown Model Card** following HuggingFace / Google / EU-CRA formats
- **Optional HTML output** for reports and dashboards
- **Schema validation** with completeness scoring
- **Drift detection** (`--check`) — diff your card's recorded metrics against new eval results
- **Snapshot history** — track metrics over time
- Sensible defaults via `--auto-populate`
- Zero heavy dependencies — just `click` and `pyyaml`

---

## Installation

```bash
pip install modelcard-as-code-generator
```

Or from source:

```bash
git clone https://github.com/danieleschmidt/modelcard-as-code-generator
cd modelcard-as-code-generator
pip install -e .
```

---

## Quick Start

**1. Print the config schema:**

```bash
mcg schema
```

**2. Generate a model card:**

```bash
mcg generate mymodel.yaml -o MODEL_CARD.md
mcg generate mymodel.yaml -o MODEL_CARD.md --html MODEL_CARD.html
```

**3. Validate an existing card:**

```bash
mcg validate MODEL_CARD.md
```

**4. Check for metric drift against new eval results:**

```bash
mcg check MODEL_CARD.md --new-eval eval_results.json
```

---

## Config File Format

A minimal config (YAML):

```yaml
model_details:
  name: my-bert-classifier
  version: "1.0.0"
  authors: [Alice Smith]
  license: Apache-2.0

intended_use: "Binary sentiment classification on product reviews."

evaluation_results:
  - name: accuracy
    value: 0.924
  - name: f1
    value: 0.918
```

A full example is in [`examples/bert_sentiment.yaml`](examples/bert_sentiment.yaml).
The generated output is in [`examples/bert_sentiment_card.md`](examples/bert_sentiment_card.md).

See all supported fields:

```bash
mcg schema          # YAML
mcg schema --format json  # JSON
```

### Top-level keys

| Key | Required | Description |
|-----|----------|-------------|
| `model_details` | Yes | Name, version, authors, license, base model |
| `intended_use` | Yes | Who the model is for and what it does |
| `training_details` | No | Framework, dataset, hyperparameters |
| `evaluation_results` | Yes | List of `{name, value, dataset?, confidence_interval?}` |
| `ethical_considerations` | No | Bias risks, mitigations, fairness metrics |
| `limitations` | No | Known limitations, out-of-scope uses |
| `custom_sections` | No | Freeform markdown sections (e.g. Carbon Footprint) |

---

## CLI Reference

```
mcg --help
mcg generate --help
mcg validate --help
mcg check --help
mcg schema --help
```

### `mcg generate`

```
mcg generate CONFIG_FILE [OPTIONS]

  -o, --output PATH       Output markdown file (default: stdout)
  --html PATH             Also write HTML to this path
  --format [huggingface|google|eu_cra]  Default: huggingface
  --no-auto-populate      Disable auto-fill of missing fields
```

### `mcg validate`

```
mcg validate CARD_FILE [OPTIONS]

  --format [huggingface|google|eu_cra]  Default: huggingface
  --strict                Exit 1 on warnings as well as errors
```

Exits 0 on success, 1 on validation failure.

### `mcg check` (drift detection)

```
mcg check CARD_FILE [OPTIONS]

  --new-eval PATH         New evaluation results (JSON/YAML) to compare
  --new-card PATH         New model card file to compare against
  --threshold FLOAT       Override default drift threshold (all metrics)
  --json-output           Print results as JSON
```

Exits 0 if no drift, 1 if significant drift detected (useful in CI).

---

## Python API

```python
from modelcard_generator import ModelCardGenerator, DriftDetector, Validator

# Generate from config file
gen = ModelCardGenerator()
card = gen.from_config("mymodel.yaml")
card.save("MODEL_CARD.md")

# Generate from individual sources
card = gen.generate(
    eval_results="eval.json",
    model_name="my-bert",
    intended_use="Sentiment classification",
)
markdown = card.render("markdown")
html = card.render("html")

# Validate
validator = Validator()
result = validator.validate_schema(card, "huggingface")
print(f"Valid: {result.is_valid}, Score: {result.score:.2f}")

# Drift detection
detector = DriftDetector()
report = detector.check(card, new_eval_results={"accuracy": 0.88, "f1": 0.87})
if report.has_drift:
    for change in report.significant_changes:
        print(f"{change.metric_name}: {change.old_value:.3f} → {change.new_value:.3f}")

# Save metric snapshot for future drift comparison
detector.save_snapshot(card, ".modelcard/snapshots.json")
```

---

## CI Integration

Add drift detection to your CI pipeline:

```yaml
# .github/workflows/model-card.yml
- name: Check for metric drift
  run: mcg check MODEL_CARD.md --new-eval eval_results.json
  # Exits 1 (failing the build) if significant drift is detected
```

Or keep the model card in sync with your code:

```yaml
- name: Regenerate model card
  run: mcg generate model_config.yaml -o MODEL_CARD.md

- name: Commit updated card
  run: |
    git add MODEL_CARD.md
    git diff --cached --quiet || git commit -m "chore: update model card"
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/test_modelcard_generator.py -v
```

---

## License

MIT — see [LICENSE](LICENSE).
