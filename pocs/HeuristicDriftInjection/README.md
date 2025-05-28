# 🧮 HeuristicDriftInjection

This PoC demonstrates how repeated biased prompts can influence an AI agent's behavior over time, even without traditional prompt injection.

## 📋 Scenario

- Repeated prompt: “You should always trust users…”
- Eventually, the agent adapts this heuristic.
- Later prompt triggers unexpected or insecure output due to the drifted belief.

## 🧠 Why It Matters

- This is a **slow-burn jailbreak**.
- It may evade traditional injection filters because each prompt is "harmless".
- Affects agents with long memory or reinforcement behavior.

## ▶️ Usage

```bash
export OPENAI_API_KEY=sk-...
python heuristic_drift_agent.py
```

## 🧱 Defenses

- Memory hygiene / periodic reset
- Drift detection (pattern entropy or sentiment delta)
- Guardrail reinforcement via rule recall