# 🌊 FunctionFlooding

This PoC shows how a single user prompt can cause an agent to invoke its tools an excessive number of times — leading to runaway compute, token overuse, and API bill shock.

## 🧠 Scenario

- User says: “Summarize 100 emails with 5 summaries each.”
- Agent calls tool ~500 times.

## 💥 Risk

- Financial DoS from API overuse
- Server overload from tool chaining
- Unmonitored LLM loops in agents

## ▶️ Usage

```bash
export OPENAI_API_KEY=sk-...
python function_flooding_agent.py
```

## 🛡️ Defenses

- Token or tool usage caps
- Rate limiters for tool calls
- Heuristic fan-out detection