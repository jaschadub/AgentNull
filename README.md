# 🧠 AgentSmuggle: AI Agent Threat Catalog + Proof-of-Concepts

This repository contains a red team-oriented catalog of attack vectors targeting autonomous AI agents (e.g., MCP, LangGraph, AutoGPT) along with individual proof-of-concepts (PoCs) for each.

## 📘 Structure

- `catalog/AgentSmuggle_Catalog.md` — Human-readable threat catalog
- `catalog/AgentSmuggle_Catalog.json` — Structured version for SOC/SIEM ingestion
- `pocs/` — One directory per attack vector, each with its own README, code, and sample input/output

## ⚠️ Disclaimer

This repository is for **educational and internal security research** purposes only. Do not deploy any techniques or code herein in production or against systems you do not own or have explicit authorization to test.

## 🔧 Usage

Navigate into each `pocs/<attack_name>/` folder and follow the README to replicate the attack scenario.

## 🧩 Example Vectors Covered

- Recursive memory leaks
- Tool confusion attacks
- Subprompt extraction
- Planning backdoors
- Infinite token/task generation

...and more.