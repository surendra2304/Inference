"""FastAPI Router for Cross-Market Intelligence, Correlations, Regime, and Venue Liquidity."""

from fastapi import APIRouter, Path, status
from pydantic import BaseModel, Field

from app.analysis.cross_asset import cross_asset_engine
from app.analysis.liquidity_intel import liquidity_intel
from app.analysis.market_regime_intel import regime_intel
from app.debate.market_debate import multi_market_debate
from app.integrations.multi_market_data import multi_market_data

multi_market_router = APIRouter(prefix="/v1/market", tags=["Cross-Market Intelligence"])


class PortfolioAnalysisRequest(BaseModel):
    positions: dict[str, float] = Field(
        # min_length=1 makes the contract explicit: an empty portfolio is not an
        # analysis request, and advertising that it is turned a client mistake into a
        # server crash (HTTP 500) instead of a 422 the caller can act on. A mutable
        # literal default is also replaced by a factory so the default can never be
        # shared between models.
        default_factory=lambda: {"BTCUSDT": 15000.0, "ETHUSDT": 8000.0, "SOLUSDT": 3000.0},
        min_length=1,
        description=(
            "Dictionary mapping asset symbol to USD notional value. "
            "Must contain at least one position."
        ),
    )


@multi_market_router.get("/cross-exchange/{asset}", status_code=status.HTTP_200_OK)
async def get_cross_exchange_view(asset: str = Path(..., description="Crypto asset symbol, e.g. BTC")):
    """Returns consolidated orderbook depth and price divergence across Binance, Bybit, and Coinbase."""
    symbol = f"{asset.upper()}USDT" if not asset.upper().endswith("USDT") else asset.upper()
    return await multi_market_data.get_cross_exchange_book(symbol)


@multi_market_router.get("/correlations", status_code=status.HTTP_200_OK)
async def get_correlation_matrix():
    """Returns cross-asset correlation matrix including crypto assets, S&P 500, Gold, and DXY."""
    return cross_asset_engine.get_correlation_matrix()


@multi_market_router.get("/regime", status_code=status.HTTP_200_OK)
async def get_market_regime():
    """Returns market regime classification, leading indicators, and 48h transition probabilities."""
    return regime_intel.classify_market_regime()


@multi_market_router.get("/liquidity/{asset}", status_code=status.HTTP_200_OK)
async def get_liquidity_analysis(asset: str = Path(..., description="Asset symbol, e.g. BTC")):
    """Returns liquidity depth scores, slippage models, and best execution venue recommendations."""
    symbol = f"{asset.upper()}USDT" if not asset.upper().endswith("USDT") else asset.upper()
    return liquidity_intel.analyze_asset_liquidity(symbol)


@multi_market_router.post("/portfolio-analysis", status_code=status.HTTP_200_OK)
async def analyze_portfolio_cross_market(req: PortfolioAnalysisRequest):
    """Executes multi-agent cross-market deliberation across the user's active portfolio holdings."""
    return multi_market_debate.conduct_cross_market_debate(req.positions)
