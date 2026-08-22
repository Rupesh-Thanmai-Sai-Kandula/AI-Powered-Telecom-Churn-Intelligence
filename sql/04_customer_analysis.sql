-- ============================================================
-- CUSTOMER-LEVEL ANALYSIS
-- ============================================================


-- Inspect individual customers

SELECT
    user_id,
    region,
    tenure,
    revenue,
    montant,
    frequence_rech,
    data_volume,
    on_net,
    orange,
    tigo,
    churn
FROM train_customers
LIMIT 20;


-- High-value customers

SELECT
    user_id,
    revenue,
    montant,
    data_volume,
    churn
FROM train_customers
WHERE revenue IS NOT NULL
ORDER BY revenue DESC
LIMIT 20;


-- Customers with high recharge activity

SELECT
    user_id,
    frequence_rech,
    revenue,
    montant,
    churn
FROM train_customers
WHERE frequence_rech IS NOT NULL
ORDER BY frequence_rech DESC
LIMIT 20;


-- Customers with high data usage

SELECT
    user_id,
    data_volume,
    revenue,
    churn
FROM train_customers
WHERE data_volume IS NOT NULL
ORDER BY data_volume DESC
LIMIT 20;