-- One row per client: the source columns plus repayment behaviour features.
-- PAY_0 and PAY_2 to PAY_6 are the repayment status from the latest month
-- (PAY_0) back to six months before; values of 1 or more are months of delay.
-- BILL_AMT1 and PAY_AMT1 are the latest month. Amounts are in NT dollars.
-- A bill can be negative when the client holds a credit balance.
SELECT
    CLIENT_ID,
    TARGET,
    LIMIT_BAL,
    AGE,
    CAST(SEX AS VARCHAR)       AS SEX,
    CAST(EDUCATION AS VARCHAR) AS EDUCATION,
    CAST(MARRIAGE AS VARCHAR)  AS MARRIAGE,
    PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
    BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
    PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6,

    GREATEST(PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)                         AS DELAY_MAX,
    CAST(PAY_0 >= 1 AS INTEGER) + CAST(PAY_2 >= 1 AS INTEGER) + CAST(PAY_3 >= 1 AS INTEGER)
      + CAST(PAY_4 >= 1 AS INTEGER) + CAST(PAY_5 >= 1 AS INTEGER) + CAST(PAY_6 >= 1 AS INTEGER) AS DELAY_MONTHS,

    BILL_AMT1 / NULLIF(LIMIT_BAL, 0)                                            AS UTIL_LATEST,
    (BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6)
      / 6.0 / NULLIF(LIMIT_BAL, 0)                                              AS UTIL_MEAN,
    GREATEST(BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6)
      / NULLIF(LIMIT_BAL, 0)                                                    AS UTIL_MAX,
    (BILL_AMT1 - BILL_AMT6) / NULLIF(LIMIT_BAL, 0)                              AS BILL_TREND,

    (PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6)
      / NULLIF(GREATEST(BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6, 0), 0)
                                                                                AS PAID_TO_BILLED_RATIO,
    CAST(PAY_AMT1 = 0 AND BILL_AMT1 > 0 AS INTEGER) + CAST(PAY_AMT2 = 0 AND BILL_AMT2 > 0 AS INTEGER)
      + CAST(PAY_AMT3 = 0 AND BILL_AMT3 > 0 AS INTEGER) + CAST(PAY_AMT4 = 0 AND BILL_AMT4 > 0 AS INTEGER)
      + CAST(PAY_AMT5 = 0 AND BILL_AMT5 > 0 AS INTEGER) + CAST(PAY_AMT6 = 0 AND BILL_AMT6 > 0 AS INTEGER)
                                                                                AS ZERO_PAYMENT_MONTHS
FROM clients
ORDER BY CLIENT_ID
