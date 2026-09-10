import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

questions = [
    "What voting mechanism in multi-model consensus best resolves irreconcilable factual disagreements among foundation models?",
    "How can model disagreement entropy be quantified to determine when to trigger deeper debate rounds versus early exit?",
    "Evaluate cost-per-token efficiency curves across frontier models for multi-round deliberative reasoning.",
    "What prompt framing produces the highest calibration of internal uncertainty estimates across diverse LLM providers?",
    "How should token generation speed and TTFT variance be balanced in streaming multi-agent consensus pipelines?"
]

def main():
    print("=" * 80)
    print("AGENT [9/9]: INFERENCE -> INFERENCE GATEWAY (5 QUESTIONS)")
    print("Client: Native ask.py CLI within d:\\FRIDAY Universe\\Inference")
    print("=" * 80)
    
    results = []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        try:
            cmd = [sys.executable, "ask.py", q]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=90)
            lat = (time.perf_counter() - t0) * 1000
            out = proc.stdout
            
            if proc.returncode == 0 and "INFERENCE RESPONSE:" in out:
                ans_part = out.split("INFERENCE RESPONSE:")[1].split("=" * 60)[1].strip()
                ans_snip = ans_part[:120].replace("\n", " ")
                print(f"[INFERENCE Q{i}/5] HTTP 200 | {lat:>7.1f}ms | Ans: {ans_snip}...")
                results.append({"q_num": i, "status": 200, "latency_ms": round(lat, 1), "answer": ans_snip})
            else:
                print(f"[INFERENCE Q{i}/5] ERROR | {lat:>7.1f}ms | {proc.stderr[:100]}")
                results.append({"q_num": i, "status": "ERROR", "latency_ms": round(lat, 1)})
        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000
            print(f"[INFERENCE Q{i}/5] ERROR | {lat:>7.1f}ms | {e}")
            results.append({"q_num": i, "status": "ERROR", "latency_ms": round(lat, 1), "error": str(e)})
            
    print("-" * 80)
    lats = [r["latency_ms"] for r in results if r["status"] == 200]
    if lats:
        print(f"INFERENCE Batch Complete: Avg Latency = {sum(lats)/len(lats):.1f}ms (Min: {min(lats):.1f}ms, Max: {max(lats):.1f}ms)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
