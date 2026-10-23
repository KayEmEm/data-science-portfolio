-- Earlier applications to the same lender.
SELECT
    SK_ID_CURR,
    COUNT(*)                                                        AS PREV_COUNT,
    AVG(CAST(NAME_CONTRACT_STATUS = 'Approved' AS DOUBLE))          AS PREV_APPROVED_SHARE,
    AVG(CAST(NAME_CONTRACT_STATUS = 'Refused' AS DOUBLE))           AS PREV_REFUSED_SHARE,
    AVG(AMT_APPLICATION / NULLIF(AMT_CREDIT, 0))                    AS PREV_APPLICATION_CREDIT_RATIO_MEAN,
    MAX(DAYS_DECISION)                                              AS PREV_DAYS_DECISION_LAST
FROM previous_application
GROUP BY SK_ID_CURR
