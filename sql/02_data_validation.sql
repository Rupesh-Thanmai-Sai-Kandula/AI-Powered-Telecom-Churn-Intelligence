-- ============================================================
-- DATA VALIDATION
-- ============================================================

-- Total number of customers

SELECT COUNT(*) AS total_customers
FROM train_customers;


-- Target distribution

SELECT
    churn,
    COUNT(*) AS customer_count
FROM train_customers
GROUP BY churn
ORDER BY churn;


-- Missing values for important numerical variables

SELECT
    COUNT(*) FILTER (WHERE montant IS NULL) AS montant_missing,
    COUNT(*) FILTER (WHERE revenue IS NULL) AS revenue_missing,
    COUNT(*) FILTER (WHERE arpu_segment IS NULL) AS arpu_missing,
    COUNT(*) FILTER (WHERE data_volume IS NULL) AS data_volume_missing,
    COUNT(*) FILTER (WHERE on_net IS NULL) AS on_net_missing
FROM train_customers;


-- Basic numerical statistics

SELECT
    MIN(revenue) AS minimum_revenue,
    MAX(revenue) AS maximum_revenue,
    AVG(revenue) AS average_revenue,
    MIN(montant) AS minimum_montant,
    MAX(montant) AS maximum_montant,
    AVG(montant) AS average_montant
FROM train_customers;