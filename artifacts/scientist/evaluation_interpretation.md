# ShopIntent AI · Evaluation interpretation

**Recommendation: ✅ Keep the main model** (decided by a fixed rule: all threshold checks pass and test PR-AUC ≥ 2× the random baseline)

_Written by the Model Evaluator agent (gpt-4.1-mini) from `evaluation.json` (27ef1452be25). All numbers: `evaluation_report.md`._

> Gradient boosting outperforms logistic regression with a PR-AUC of 0.346 on validation and 0.369 on test, supporting its use for purchase prediction.

## Why this algorithm

Gradient boosting was selected because it achieved the highest validation PR-AUC of 0.346 compared to logistic regression's 0.321, indicating better precision-recall performance. It also showed a higher ROC-AUC of 0.773 versus 0.764, demonstrating improved discrimination ability on the validation set. These metrics suggest gradient boosting better captures the purchase behavior in the data.

## The PageValues trade-off

Although adding PageValues improves test PR-AUC substantially to 0.753 and ROC-AUC to 0.935, it is excluded from the main model because PageValues are computed from pages that led to transactions, thus leaking the outcome. Using such features would compromise the model's validity and generalizability in real-world scenarios where future outcomes are unknown.

## Thresholds and segments

Thresholds were fixed in advance at 50.0% for buyers, 20.0%-50.0% for hesitant, and below 20.0% for unlikely, and were not tuned. All threshold checks passed on validation, confirming that each segment contains sessions and that purchase rates are ordered buyer > hesitant > unlikely. On test data, segments behave consistently: buyers (3.8% of sessions) have a 49.3% purchase rate, hesitant (27.6%) have 27.8%, and unlikely (68.5%) have 8.6%, confirming the thresholds' effectiveness on unseen data.

## What it means for the store

The model effectively segments visitors by purchase likelihood, enabling targeted interventions: coupons for hesitant returning visitors, chat invitations for hesitant new visitors, and no intervention for others. The main model's PR-AUC of 0.369 is more than twice the random baseline of 0.155, indicating meaningful predictive power. This supports using the model to optimize marketing resources and improve conversion rates in the online store.

## Next steps

- Monitor model performance over time to detect potential degradation.
- Explore additional features excluding outcome-leaking variables like PageValues.
- Evaluate intervention effectiveness by segment to refine marketing strategies.
- Consider retraining the model periodically with new data to maintain accuracy.
