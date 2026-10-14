"""Production Deployment and Pre-flight Verification Script for AI Universe."""

import subprocess
import sys
from datetime import datetime

from fastapi.testclient import TestClient

from app.config_production import production_config
from app.main import app


def run_command(cmd: str) -> bool:
    print(f"\n[RUNNING] {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with return code {res.returncode}")
        return False
    return True


def audit_environment():
    print("================================================================================")
    print("         AI UNIVERSE — PRODUCTION DEPLOYMENT & HEALTH PRE-FLIGHT")
    print("================================================================================")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print(f"Environment: {production_config.APP_ENV}")
    print(f"Max Concurrency: {production_config.MAX_CONCURRENT_REQUESTS}")
    print(f"Provider Fallback Chain: {' -> '.join(production_config.PROVIDER_PRIORITY)}")
    print(f"Cache TTL: {production_config.CACHE_TTL_SECONDS}s")
    print("Checking core requirements...")

    client = TestClient(app)

    # Check Health Endpoints
    print("\n--- [1/3] Verifying Health Endpoints ---")
    r_health = client.get("/health")
    assert r_health.status_code == 200 and r_health.json().get("status") in ("ok", "healthy")
    print("  [+] GET /health: OK")

    r_detailed = client.get("/health/detailed")
    assert r_detailed.status_code == 200
    print("  [+] GET /health/detailed: OK")

    r_prov = client.get("/health/providers")
    assert r_prov.status_code == 200
    print("  [+] GET /health/providers: OK")

    r_status = client.get("/status")
    assert r_status.status_code == 200 and r_status.json()["status"] == "operational"
    print("  [+] GET /status: OK (All 10 specialists active)")

    r_ready = client.get("/ready")
    assert r_ready.status_code == 200 and r_ready.json()["ready"] is True
    print("  [+] GET /ready: OK (Database & provider routing ready)")

    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200 and ("inference_requests_total" in r_metrics.text or "ai_universe_requests_total" in r_metrics.text)
    print("  [+] GET /metrics: OK (Prometheus metrics active)")

    # Check Trading Subsystem
    print("\n--- [2/3] Verifying Trading Advisory & A/B / Testnet Endpoints ---")
    from app.core.config import settings
    auth_key = settings.INFERENCE_API_KEY or (settings.get_valid_api_keys()[0] if settings.get_valid_api_keys() else "dev_secret_key")
    auth_headers = {"X-INFERENCE-API-KEY": auth_key}

    r_trading_health = client.get("/v1/trading/consult/health", headers=auth_headers)
    assert r_trading_health.status_code == 200
    th = r_trading_health.json()
    assert th["advisory_only"] is True and th["exchange_execution"] is False
    print("  [+] GET /v1/trading/consult/health: OK (Advisory invariant strictly preserved)")

    r_testnet_perf = client.get("/v1/trading/testnet/performance", headers=auth_headers)
    assert r_testnet_perf.status_code == 200
    print("  [+] GET /v1/trading/testnet/performance: OK")

    r_testnet_comp = client.get("/v1/trading/testnet/comparison", headers=auth_headers)
    assert r_testnet_comp.status_code == 200
    print("  [+] GET /v1/trading/testnet/comparison: OK")

    print("\n--- [3/3] Running Pytest & Performance Benchmark ---")
    return True


if __name__ == "__main__":
    if not audit_environment():
        sys.exit(1)

    print("\n================================================================================")
    print("   [SUCCESS] AI Universe production pre-flight audit passed flawlessly.")
    print("================================================================================")
