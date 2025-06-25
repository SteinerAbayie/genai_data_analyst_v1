WITH ChurnData AS (
  SELECT
    `City`,
    SUM(CASE WHEN `Churn_Label` THEN 1 ELSE 0 END) AS `churn_count`
  FROM
    `cosmic-kayak-462516-q8.test_cap.telco_main`
  WHERE
    `State` = 'California'
  GROUP BY
    `City`
)
SELECT
  `City`,
  `churn_count`
FROM
  ChurnData
ORDER BY
  `churn_count` DESC