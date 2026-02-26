from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "Crypto Arbitrage MVP"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://arb:arb@localhost:5432/arbitrage"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Market data
    ORDERBOOK_DEPTH: int = 10
    SPREAD_HISTORY_SECONDS: int = 300   # 5-minute sliding window
    WS_BROADCAST_INTERVAL: float = 1.0  # seconds between broadcast ticks

    # Execution safety
    PRICE_SLIPPAGE_TOLERANCE: float = 0.005   # 0.5%
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_DELAY_SECONDS: float = 0.5

    # Supported connectors (exchange ids used by ccxt)
    SUPPORTED_CONNECTORS: List[str] = [
        "binance",
        "bybit",
        "okx",
        "gate",
        "hyperliquid",
    ]

    # Supported tokens (base assets)
    SUPPORTED_TOKENS: List[str] = [
        "BTC",
        "ETH",
        "SOL",
        "BNB",
        "XRP",
        "DOGE",
        "AVAX",
        "MATIC",
        "ARB",
        "OP",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
