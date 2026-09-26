import sys
import os

import requests

# Ensure UTF-8 output encoding for Windows PowerShell console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter question: ")
if not q.strip():
    print("Question cannot be empty.")
    sys.exit(0)

print("\n[Inference Gateway] Routing to Multi-Agent Specialist Cluster...")
api_key = os.getenv("INFERENCE_API_KEY") or os.getenv("FRIDAY_API_KEY")
if not api_key:
    print("Request not sent: configure INFERENCE_API_KEY or FRIDAY_API_KEY first.")
    sys.exit(2)
try:
    r = requests.post(
        "https://inference-r1sn.onrender.com/v1/friday/ask",
        headers={"X-FRIDAY-API-Key": api_key},
        json={"question": q},
        timeout=120,
    )
    if r.status_code == 200:
        data = r.json()
        print("\n" + "=" * 60)
        print("INFERENCE RESPONSE:")
        print("=" * 60)
        print(data.get("answer", data))
        print("=" * 60 + "\n")
    else:
        print(f"Error {r.status_code}: {r.text}")
except Exception as e:
    print(f"Request failed: {e}")
