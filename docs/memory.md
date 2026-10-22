# Memory Subsystem

## Memory Layers
1. **Working Context**: Active debate state, live messages, transient scratchpads.
2. **Agent Memory**: Agent-specific experiences, role heuristics, preferences, past decisions (scoped strictly by `agent_id`).
3. **System Memory**: Experiment logs, model evaluation scores, provider latency/reliability statistics.
4. **Knowledge & Evidence Cache**: Validated factual assertions, benchmark references, reusable research notes.

## Storage Architecture
- Single SQLite database file (`data/universe.db`).
- Tables: `agents`, `memories`, `relationships`, `tasks`, `runs`, `messages`, `evaluations`, `experiments`, `strategies`.
