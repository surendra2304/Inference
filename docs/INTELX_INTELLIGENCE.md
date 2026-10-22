# IntelX Deep Research Intelligence & Verification API

Inference provides specialized research reasoning and claim verification for **IntelX**, enabling structured multi-agent decomposition, verbatim span extraction, adversarial verification debates, and coherent research synthesis.

---

## 1. Primary Endpoints

- **`POST /v1/intelx/research`**: Executes agent-role research requests with strict quality controls.
- **`GET /v1/intelx/research/{request_id}`**: Retrieves stored research audit records with complete provenance ledger.

---

## 2. Research Roles & Specialized Agent Mappings

| Role | Agent Composition | Execution & Focus |
| :--- | :--- | :--- |
| `planner` | **Strategist** | Decomposes overarching research questions into structured sub-questions and investigative tracks. |
| `extractor` | **Coder** | Performs literal, verbatim text span extraction with strict avoidance of hallucination or paraphrasing. |
| `verifier` | **Fact Checker + Critic** *(REVIEW Mode)* | Two-agent verification debate: Fact Checker tests evidence sufficiency; Critic challenges invalidating assumptions. Disagreements become `dissent`. |
| `analyst` | **Data Analyst + Researcher** | Identifies quantitative and qualitative patterns across peer-reviewed and official documents. |
| `critic` | **Critic** | Adversarial review challenging methodological premises and publication bias. |
| `synthesizer` | **Synthesizer** | Assembles findings into a coherent report with direct span citations. |

---

## 3. Research Quality Controls

1. **Verbatim Span Verification**: Ensures conclusions explicitly reference exact source spans.
2. **Source Independence Detection**: Automatically flags syndicated or copied text spans across distinct domains.
3. **Credibility Weighting**: Calculates an overall credibility score $[0.0, 1.0]$ based on source trust labels (`peer_reviewed`, `official_doc`, `news_wire`, `blog_post`, `unverified_social`).

---

## 4. Rate Limiting & Multi-Tenant Budgeting

- **Consumer Rate Limit**: $200\text{ reqs/hour}$ dedicated queue for the `intelx` consumer key.
- **Monthly Budget**: $\$100.00$ monthly allocation with soft ($80\%$) and hard ($100\%$) limit enforcement.
