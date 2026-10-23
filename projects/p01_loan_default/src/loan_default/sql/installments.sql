-- Repayment behaviour on earlier loans. A positive delay means the payment
-- came after its due day; a ratio below one means a partial payment.
SELECT
    SK_ID_CURR,
    COUNT(*)                                                         AS INST_COUNT,
    AVG(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)                        AS INST_DELAY_MEAN,
    MAX(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)                        AS INST_DELAY_MAX,
    AVG(CAST(DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT AS DOUBLE))        AS INST_LATE_SHARE,
    AVG(AMT_PAYMENT / NULLIF(AMT_INSTALMENT, 0))                     AS INST_PAYMENT_RATIO_MEAN,
    AVG(CAST(AMT_PAYMENT < AMT_INSTALMENT AS DOUBLE))                AS INST_UNDERPAID_SHARE
FROM installments_payments
GROUP BY SK_ID_CURR
