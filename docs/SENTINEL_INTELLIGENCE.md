# Sentinel Cybersecurity Intelligence & Posture Analysis

Inference provides defensive security intelligence and posture analysis for **Sentinel**, supporting automated vulnerability assessment, adversarial attack path reasoning, remediation prioritization, threat intelligence correlation, and dynamic risk scoring.

---

## 1. Primary Analysis Endpoints

- **`POST /v1/sentinel/analyze`**: Executes security posture analysis with multi-agent panels, evidence mapping, and calibrated risk scoring.
- **`GET /v1/sentinel/analyze/{request_id}`**: Retrieves stored security intelligence audit records with full provenance ledger.

---

## 2. Analysis Types & Agent Composition

| Analysis Type | Specialist Agent Composition | Description |
| :--- | :--- | :--- |
| `vulnerability_assessment` | **Security Analyst + Data Analyst** | Evaluates findings, severity distribution, CVSS scores, and exposure context. |
| `attack_path_reasoning` | **Security Analyst + Strategist + Critic** *(Debate Mode)* | Models conceptual exploitation chains and calculates step likelihoods; Critic challenges false-positive paths. |
| `remediation_prioritization` | **Strategist + Security Analyst** | Ranks defensive fixes by risk reduction vs implementation effort (`QUICK_WIN`, `MODERATE`, `SIGNIFICANT_REFACTOR`). |
| `threat_intel_correlation` | **Researcher + Data Analyst** | Maps CVE matches against active in-the-wild exploitation and MITRE ATT&CK tactics. |
| `risk_scoring` | **Data Analyst + Critic** | Generates weighted $[0.0, 10.0]$ overall risk scores calibrated against target asset exposure level. |

---

## 3. Request & Response Contracts

### Request Payload (`SentinelAnalysisRequest`)
```json
{
  "request_id": "sent-req-001",
  "analysis_type": "attack_path_reasoning",
  "target_context": {
    "asset_type": "api_gateway",
    "technologies_detected": ["FastAPI", "Python", "Nginx"],
    "exposure_level": "public_internet"
  },
  "findings": [
    {
      "finding_id": "F-01",
      "severity": "HIGH",
      "title": "Missing Security Headers",
      "description": "Strict-Transport-Security header is not set on TLS listener.",
      "evidence_refs": ["header_audit_log_line_45"],
      "cvss_score": 7.5
    }
  ],
  "threat_intel": {
    "cve_matches": ["CVE-2026-GATE-01"],
    "exploit_availability": "poc",
    "threat_actor_activity": "low"
  },
  "constraints": {
    "scan_mode": "standard",
    "time_budget": 10
  }
}
```

### Response Payload (`SentinelAnalysisResponse`)
```json
{
  "request_id": "sent-req-001",
  "analysis": {
    "risk_assessment": {
      "overall_risk_score": 8.1,
      "risk_tier": "HIGH",
      "executive_summary": "Evaluated 1 finding(s) across api_gateway...",
      "key_vulnerability_factors": ["Missing Security Headers"]
    },
    "attack_paths": [
      {
        "chain_id": "PATH-001",
        "title": "External Public internet to api_gateway Boundary Breach",
        "overall_probability": 0.78,
        "criticality": "HIGH",
        "nodes": [
          {
            "step_number": 1,
            "vector": "Public Service Discovery (public_internet)",
            "preconditions": "Exposed public ingress endpoint...",
            "potential_impact": "Initial perimeter foothold",
            "likelihood_score": 0.85,
            "associated_finding_ids": ["F-01"]
          }
        ]
      }
    ],
    "prioritized_remediation": [
      {
        "priority_rank": 1,
        "finding_id": "F-01",
        "title": "Remediate: Missing Security Headers",
        "recommended_fix": "Apply vendor patch or configuration boundary...",
        "rationale": "High risk reduction (25.0%) against public_internet exposure.",
        "effort_estimate": "QUICK_WIN",
        "risk_reduction_pct": 25.0
      }
    ],
    "threat_context": {
      "active_in_the_wild": false,
      "trending_cves_for_stack": ["CVE-2026-GATE-01"],
      "mitre_attack_tactics": ["Initial Access", "Defense Evasion", "Lateral Movement"]
    },
    "confidence": 0.88,
    "dissent": [
      "Critic challenged reachability of secondary lateral movement step under strict VPC segmentation."
    ]
  },
  "evidence_references": {
    "F-01": ["header_audit_log_line_45"]
  },
  "safety_notes": [
    "Inference Sentinel Analysis is strictly defensive and advisory.",
    "Never executes active exploits, intrusive probing, or unauthorized network disruption.",
    "All findings and attack chain models are theoretical security posture assessments for defensive hardening."
  ],
  "provenance": {
    "request_id": "sent-req-001",
    "analysis_type": "attack_path_reasoning",
    "agents_consulted": ["security_analyst", "strategist", "critic"],
    "latency_ms": 1.25,
    "findings_evaluated": 1,
    "timestamp": 1724880000.0
  }
}
```

---

## 4. Consumer Isolation & Rate Limiting

- **Rate Limit**: $100\text{ reqs/hour}$ dedicated quota for the `sentinel` consumer API key.
- **Dedicated Queue**: Security intelligence workloads execute independently without blocking or being blocked by FORGE or Trading Bot traffic.
- **Deduplication & Idempotency**: 5-minute deduplication window keyed on `request_id`.
