# bert-sentiment-amazon

BERT-base-uncased fine-tuned on Amazon product reviews for binary sentiment classification (positive / negative). Trained by the OpenClaw ML team as a reference example.


## Model Details
- **Version**: 1.2.0
- **Authors**: Daniel Schmidt
- **License**: Apache-2.0
- **Base Model**: bert-base-uncased

## Intended Use
Classify product reviews as positive or negative sentiment. Designed for use in e-commerce analytics pipelines. Target users: data science teams at retail companies. NOT intended for mental health, medical, or legal applications.


## Training Details
- **Framework**: PyTorch
- **Training Data**: amazon-reviews-v2 (2.4M reviews, balanced 50/50 split)
- **Hyperparameters**:
  - learning_rate: 2e-05
  - batch_size: 32
  - epochs: 4
  - warmup_ratio: 0.1
  - weight_decay: 0.01
  - max_length: 256

## Evaluation Results
- **accuracy**: 0.942 (CI: [0.939, 0.945])
- **f1**: 0.941
- **precision**: 0.938
- **recall**: 0.944
- **roc_auc**: 0.987
- **accuracy**: 0.891

## Ethical Considerations
### Bias Risks
- Model accuracy is 3–5 pp lower on reviews written in non-standard English.
- Negative-sentiment reviews by minority demographic groups may be under-detected.
### Fairness Metrics
- **demographic_parity_ratio**: 0.96
- **equalized_odds_difference**: 0.04

## Limitations
- Trained on Amazon reviews; may not generalise to other review domains.
- Accuracy drops ~5 pp on very short reviews (< 20 words).
- Does not handle mixed-sentiment or neutral reviews well.
- English-only; multilingual reviews will produce unreliable results.

## Carbon Footprint
Training was tracked with CodeCarbon.
Estimated emissions: **1.2 kg CO₂e** (equivalent to ~5 km driven).
Hardware: 2x A100, ~6 hours @ ~250W each.


## Citation
If you use this model, please cite:
```
@misc{bert-sentiment-amazon-2026,
  author = {Schmidt, Daniel},
  title  = {bert-sentiment-amazon},
  year   = {2026},
  url    = {https://github.com/danieleschmidt/modelcard-as-code-generator}
}
```
