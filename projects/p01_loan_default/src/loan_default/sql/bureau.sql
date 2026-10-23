-- Credit bureau history per applicant. Monthly bureau statuses '1' to '5' mean
-- days past due buckets; their share per bureau credit is a delinquency signal.
WITH bb AS (
    SELECT
        SK_ID_BUREAU,
        COUNT(*)                                                      AS months,
        AVG(CAST(STATUS IN ('1', '2', '3', '4', '5') AS DOUBLE))     AS late_share
    FROM bureau_balance
    GROUP BY SK_ID_BUREAU
)
SELECT
    b.SK_ID_CURR,
    COUNT(*)                                                AS BUR_COUNT,
    SUM(CAST(b.CREDIT_ACTIVE = 'Active' AS INTEGER))        AS BUR_ACTIVE_COUNT,
    AVG(b.DAYS_CREDIT)                                      AS BUR_DAYS_CREDIT_MEAN,
    MAX(b.DAYS_CREDIT)                                      AS BUR_DAYS_CREDIT_MAX,
    MAX(b.CREDIT_DAY_OVERDUE)                               AS BUR_DAY_OVERDUE_MAX,
    SUM(b.AMT_CREDIT_SUM)                                   AS BUR_CREDIT_SUM,
    SUM(b.AMT_CREDIT_SUM_DEBT)                              AS BUR_DEBT_SUM,
    SUM(b.AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(b.AMT_CREDIT_SUM), 0) AS BUR_DEBT_CREDIT_RATIO,
    AVG(bb.late_share)                                      AS BUR_MONTHLY_LATE_SHARE_MEAN,
    MAX(bb.late_share)                                      AS BUR_MONTHLY_LATE_SHARE_MAX,
    SUM(bb.months)                                          AS BUR_MONTHS_OBSERVED
FROM bureau AS b
LEFT JOIN bb USING (SK_ID_BUREAU)
GROUP BY b.SK_ID_CURR
