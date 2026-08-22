-- ============================================================
-- EXPLORATORY DATA ANALYSIS
-- ============================================================


-- Churn distribution

SELECT
    churn,
    COUNT(*) AS customers
FROM train_customers
GROUP BY churn
ORDER BY churn;


-- Average revenue by churn status

SELECT
    churn,
    COUNT(*) AS customers,
    AVG(revenue) AS average_revenue,
    AVG(montant) AS average_montant
FROM train_customers
GROUP BY churn
ORDER BY churn;


-- Recharge behavior by churn status

SELECT
    churn,
    AVG(frequence_rech) AS average_recharge_frequency,
    AVG(freq_top_pack) AS average_top_pack_frequency
FROM train_customers
GROUP BY churn
ORDER BY churn;


-- Usage behavior by churn status

SELECT
    churn,
    AVG(data_volume) AS average_data_volume,
    AVG(on_net) AS average_on_net,
    AVG(orange) AS average_orange,
    AVG(tigo) AS average_tigo
FROM train_customers
GROUP BY churn
ORDER BY churn;


-- Customer distribution by region

SELECT
    region,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = 1 THEN 1 ELSE 0 END) AS churned_customers
FROM train_customers
GROUP BY region
ORDER BY customers DESC;