-- Monthly point of sale and cash loan balances.
SELECT
    SK_ID_CURR,
    COUNT(*)                                         AS POS_MONTHS,
    MAX(SK_DPD)                                      AS POS_DPD_MAX,
    AVG(CAST(SK_DPD_DEF > 0 AS DOUBLE))              AS POS_DPD_DEF_SHARE
FROM POS_CASH_balance
GROUP BY SK_ID_CURR
