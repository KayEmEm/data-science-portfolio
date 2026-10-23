# Experiment readout: email test of March 2008

| | |
| --- | --- |
| **To** | Marketing leadership |
| **From** | Analytics |
| **Data** | Hillstrom MineThatData email test: 64,000 customers, two weeks of outcomes |
| **Question** | Did the campaign raise spending, and who should get the next email? |

## Summary

1. **Both emails raised spending.** Over two weeks the mens email added $0.77 ($0.49 to $1.05) per customer against no email (+118%), the womens email $0.42 ($0.17 to $0.68) (+65%). Visits rose by 7.7 and 4.5 percentage points, conversions by 0.68 and 0.31 points. All six effects remain significant after Holm correction.
2. **Recommendation: send the mens email to every customer in this population.** Under the stated economics (30% margin, $0.10 per email) this earns $0.396 per customer on the test split against $0.188 with no email: $0.208 more per customer, about $20,794 per 100,000 customers. The policy was chosen on the validation split; it is also the best policy on the test split.
3. **Model based targeting does not pay yet.** No uplift model ranks customers by incremental spend better than chance: every spend Qini interval on the test split includes zero. The policy of the model chosen on the validation split (T learner) earns $0.288 per customer on the test split. At every email cost from $0.00 to $0.50, a simple policy earns at least as much as the model policy.
4. **Response differs by customer in visits, not measurably in spend.** For the womens email, ranking by the causal forest finds the customers whose visits rise most (Qini 0.82 points; random rankings fall between -0.32 and 0.32). Ranking simply by past womens purchases does as well (0.79).

## Was the test sound?

* Arm sizes (control 21,306, mens 21,307, womens 21,387) match the planned equal split (p = 0.90).
* Every pre treatment feature is balanced (largest standardized difference 0.014; threshold 0.1).
* Power: the test could detect a spend effect of $0.31 per customer with 80% power; both observed effects are larger.
* Variance reduction: past year spend barely predicts two week spend (correlation 0.022), so CUPED cuts the spend variance by 0.04%. Regression on all pre treatment features cuts the visit variance by 2.9%. Neither changes a conclusion.

## What if there had been no experiment?

A simulated targeted send (recent, high value web customers mailed more often) shows what an observational study would have reported for the mens email. The naive comparison gives 12.2 points of extra visits against the true 7.7. Adjusting for every confounder except past spend still gives 9.7. Only the full set recovers the truth (7.6 with propensity weighting, 7.6 with regression). DoWhy's refutation tests pass for the incomplete set as well, so they cannot reveal a missing confounder.

## Next steps

* Mail the full file, and keep a random holdout without email to keep measuring the effect.
* Before investing in targeting models, run a larger test or track a longer outcome window, so that differences in spend response become measurable.
* Replace the assumed margin and email cost with finance figures. At a 30% margin, mailing everyone beats mailing no one while an email costs less than $0.31.

Details: [A/B tables](ab_results.md), [uplift models](uplift_comparison.md), [policies](policy.md), [observational analysis](observational.md).
