-- ============================================================
-- TELECOM CHURN INTELLIGENCE
-- DATABASE SETUP
-- ============================================================

-- Database used for the project:
-- expresso_churn

-- The following commands are examples of the database
-- structure used during the project.

-- ============================================================
-- CHECK DATABASE TABLES
-- ============================================================

SELECT
    table_schema,
    table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;


-- ============================================================
-- INSPECT TRAINING TABLE
-- ============================================================

SELECT *
FROM train_customers
LIMIT 10;


-- ============================================================
-- CHECK NUMBER OF RECORDS
-- ============================================================

SELECT COUNT(*) AS total_customers
FROM train_customers;


-- ============================================================
-- CHECK TABLE STRUCTURE
-- ============================================================

SELECT
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'train_customers'
ORDER BY ordinal_position;