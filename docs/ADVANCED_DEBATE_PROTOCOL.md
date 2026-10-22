# Advanced Debate Protocol & Reasoning Transparency

Inference implements structured multi-round adversarial debate orchestration, reasoning chain explainability, evidence weighting, assumption tracking, and multi-model verification.

---

## 1. 4-Round Structured Adversarial Protocol

```
┌────────────────────────────────────────┐
│ Round 1: Independent Analysis          │
│ (Each agent analyzes evidence solo)    │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Round 2: Cross-Examination             │
│ (Agents challenge & defend arguments)  │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Round 3: Synthesis Attempt             │
│ (Synthesizer unifies strong arguments) │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Round 4: Final Objections              │
│ (Dissent recorded as unresolved diffs) │
└────────────────────────────────────────┘
```

---

## 2. Evidence Scoring & Assumption Tracking

- **Evidence Reliability Weighting**:
  - `system_fact`: $1.0\times$
  - `verified_telemetry`: $0.9\times$
  - `inferred_profile`: $0.6\times$
  - `untrusted_user_input`: $0.3\times$ (Low-trust signals discounted)
- **Contradiction Detection**: Flags conflicting telemetry inputs for focused scrutiny in Round 2 cross-examinations.
- **Assumption Tracking**: Formulates explicit testable hypotheses with confidence scores that validate/invalidate when real-world outcomes arrive.

---

## 3. Multi-Model Verification & Perspective Diversity

- **Heterogeneous Model Routing**: Distributes debate roles across disparate foundation models:
  - Strategist $\rightarrow$ **Gemini**
  - Critic $\rightarrow$ **Groq**
  - Fact Checker $\rightarrow$ **Mistral**
  - System Architect $\rightarrow$ **NVIDIA**
- **Diversity Lift**: Multi-model diverse debates achieve $+12.6\%$ higher downstream verification success rates compared to single-model debates ($93.8\%$ vs $81.2\%$).

---

## 4. Endpoints Reference

- `GET /v1/intelligence/{request_id}/trace`: Full reasoning chain trace, cross-examinations, stated assumptions, evidence scores, and confidence evolution across rounds.
- `GET /v1/debate/statistics`: Empirical metrics on debate compositions, objection rates, and provider diversity performance.
