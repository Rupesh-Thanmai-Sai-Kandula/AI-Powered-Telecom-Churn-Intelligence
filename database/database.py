from sqlalchemy import create_engine
from urllib.parse import quote_plus

# ==========================================
# DATABASE CONFIGURATION
# ==========================================

USERNAME = "postgres"
PASSWORD = "Rupesh@12"
HOST = "localhost"
PORT = "5432"
DATABASE = "expresso_churn"

# Encode password so special characters such as @ are handled correctly
ENCODED_PASSWORD = quote_plus(PASSWORD)

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{USERNAME}:{ENCODED_PASSWORD}"
    f"@{HOST}:{PORT}/{DATABASE}"
)

engine = create_engine(DATABASE_URL)


def get_engine():
    return engine