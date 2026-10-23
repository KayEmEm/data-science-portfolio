-- Monthly credit card balances.
SELECT
    SK_ID_CURR,
    COUNT(*)                                                         AS CC_MONTHS,
    AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))            AS CC_UTILIZATION_MEAN,
    MAX(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))            AS CC_UTILIZATION_MAX,
    MAX(SK_DPD)                                                      AS CC_DPD_MAX
FROM credit_card_balance
GROUP BY SK_ID_CURR
