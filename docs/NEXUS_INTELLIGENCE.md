# Nexus Intelligence Service & Mode-Based Routing

Inference implements high-speed, calibrated multi-mode intelligence endpoints adhering to the **Nexus Intelligence Contract**.

---

## 1. Mode-Based Routing & Agent Allocation

| Mode | Target Latency | Max Rounds | Specialist Panel | Use Case |
| :--- | :---: | :---: | :--- | :--- |
| **FAST** | $\le 3\text{s}$ | 1 | Single best-matching specialist agent | Low-risk deterministic decisions |
| **REVIEW** | $\le 8\text{s}$ | 1 | Primary agent + Critic adversarial review | Moderate ambiguity & risk mitigation |
| **DEBATE** | $\le 20\text{s}$ | Up to 6 | Multi-round panel (Primary + Critic + Domain Experts) | Strategic high-impact decisions |

---

## 2. Task Type to Specialist Mapping

- `lead_qualification` $\rightarrow$ **Strategist** + **Data Analyst**
- `conversion_diagnosis` $\rightarrow$ **Data Analyst** + **Debugger** + **Critic**
- `incident_analysis` $\rightarrow$ **Debugger** + **Security Analyst** + **Critic**
- `strategic_decision` $\rightarrow$ **Strategist** + **Critic** + **Fact Checker**
- `intervention_planning` $\rightarrow$ **Strategist** + **Debugger** + **Critic**
- `copy_optimization` $\rightarrow$ **Synthesizer**
- `churn_analysis` $\rightarrow$ **Data Analyst** + **Strategist**

---

## 3. Confidence Calibration & Evidence Trust Hierarchy

Confidence scores ($0.0 - 1.0$) are dynamically calibrated based on evidence provenance:
- **System Fact** ($1.0\times$ weight)
- **Verified Telemetry** ($0.9\times$ weight)
- **Inferred Profile** ($0.6\times$ weight)
- **Untrusted User Input** ($0.5\times$ weight)

> [!IMPORTANT]
> **Preservation of Disagreements**: When specialist agents disagree, the confidence score is penalized proportionally, and unresolved dissenting views are **always preserved** in `unresolved_disagreements` (never silently flattened).

---

## 4. Endpoints Reference

- `POST /v1/nexus/intelligence`: Execute FAST, REVIEW, or DEBATE intelligence queries.
- `GET /v1/nexus/intelligence/{request_id}`: Retrieve stored request/response records with full provenance ledger.
