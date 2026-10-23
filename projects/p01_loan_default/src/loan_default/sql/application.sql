-- One row per applicant: the application record plus ratio features.
-- DAYS_EMPLOYED holds 365243 as a placeholder for "not employed"; it becomes NULL
-- and a separate flag keeps the information.
SELECT
    a.* REPLACE (CASE WHEN a.DAYS_EMPLOYED = 365243 THEN NULL ELSE a.DAYS_EMPLOYED END AS DAYS_EMPLOYED),
    CAST(a.DAYS_EMPLOYED = 365243 AS INTEGER)            AS APP_NOT_EMPLOYED_FLAG,
    a.AMT_CREDIT  / NULLIF(a.AMT_INCOME_TOTAL, 0)         AS APP_CREDIT_INCOME_RATIO,
    a.AMT_ANNUITY / NULLIF(a.AMT_INCOME_TOTAL, 0)         AS APP_ANNUITY_INCOME_RATIO,
    a.AMT_ANNUITY / NULLIF(a.AMT_CREDIT, 0)               AS APP_ANNUITY_CREDIT_RATIO,
    a.AMT_GOODS_PRICE / NULLIF(a.AMT_CREDIT, 0)           AS APP_GOODS_CREDIT_RATIO,
    -a.DAYS_BIRTH / 365.25                                AS APP_AGE_YEARS,
    (COALESCE(a.EXT_SOURCE_1, 0) + COALESCE(a.EXT_SOURCE_2, 0) + COALESCE(a.EXT_SOURCE_3, 0))
        / NULLIF(
            CAST(a.EXT_SOURCE_1 IS NOT NULL AS INTEGER)
          + CAST(a.EXT_SOURCE_2 IS NOT NULL AS INTEGER)
          + CAST(a.EXT_SOURCE_3 IS NOT NULL AS INTEGER), 0)  AS APP_EXT_SOURCE_MEAN
FROM application_train AS a
