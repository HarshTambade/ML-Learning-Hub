# Advanced Topics in Model Evaluation

## Calibration

Model calibration ensures predicted probabilities match actual outcomes.

- Predicted probability of 0.7 → ~70% occur in practice
- Important for decision-making
- Especially critical in high-stakes applications

### Calibration Methods
- Platt scaling: Logistic regression on predictions
- Isotonic regression: Non-parametric approach
- Temperature scaling: For neural networks

## Ranking Metrics

For ranking/recommendation systems:
- **NDCG (Normalized DCG)**: Position-aware ranking
- **MAP (Mean Average Precision)**: Quality of ranked results
- **Recall@k**: Fraction of positives in top k

## Fairness and Bias

Ensure models don't discriminate:
- Demographic parity: Equal treatment across groups
- Equalized odds: Equal TPR/FPR across groups
- Calibration across subgroups

## Cost-Sensitive Learning

When errors have different costs:
- Asymmetric loss functions
- Weighted metrics
- Threshold optimization

## Cold Start and Feedback Loops

- Cold start: Limited data on new items
- Feedback loops: Model predictions affect future data
- Exploration-exploitation trade-off

## A/B Testing

- Gold standard for online evaluation
- Compare control vs treatment
- Statistical significance testing
- Multi-armed bandits

## Adversarial Robustness

Model robustness against perturbations:
- Adversarial examples
- Robustness evaluation
- Defense mechanisms

## Few-Shot and Zero-Shot Learning

- Learning from limited examples
- Transfer learning evaluation
- Out-of-distribution generalization

## Meta-Learning

- Learning to learn
- Evaluation on new tasks
- Model-agnostic evaluation

## Summary

Advanced evaluation techniques address:
- Probability calibration
- Fairness and bias
- Cost-sensitive scenarios
- Online evaluation
- Robustness
- Domain shift

Choose techniques based on problem requirements.
