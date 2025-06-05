# 🧠 AgentNull: AI System Security Threat Catalog + Proof-of-Concepts

This repository contains a red team-oriented catalog of attack vectors targeting AI systems including autonomous agents (MCP, LangGraph, AutoGPT), RAG pipelines, vector databases, and embedding-based retrieval systems, along with individual proof-of-concepts (PoCs) for each.

## 📘 Structure

- `catalog/AgentNull_Catalog.md` — Human-readable threat catalog
- `catalog/AgentNull_Catalog.json` — Structured version for SOC/SIEM ingestion
- `pocs/` — One directory per attack vector, each with its own README, code, and sample input/output

## ⚠️ Disclaimer

This repository is for **educational and internal security research** purposes only. Do not deploy any techniques or code herein in production or against systems you do not own or have explicit authorization to test.

## 🔧 Usage

Navigate into each `pocs/<attack_name>/` folder and follow the README to replicate the attack scenario.

## 🧩 Attack Vectors Covered

### 🤖 MCP & Agent Systems
- **[Full-Schema Poisoning (FSP)](pocs/FullSchemaPoisoning/)** - Exploit any field in tool schema beyond descriptions
- **[Advanced Tool Poisoning Attack (ATPA)](pocs/AdvancedToolPoisoning/)** - Manipulate tool outputs to trigger secondary actions
- **[MCP Rug Pull Attack](pocs/MCPRugPull/)** - Swap benign descriptions for malicious ones after approval
- **[Schema Validation Bypass](pocs/SchemaValidationBypass/)** - Exploit client validation implementation differences
- **[Tool Confusion Attack](pocs/ToolConfusionAttack/)** - Trick agents into using wrong tools via naming similarity
- **[Nested Function Call Hijack](pocs/NestedFunctionHijack/)** - Use JSON-like data to trigger dangerous function calls
- **[Subprompt Extraction](pocs/SubpromptExtraction/)** - Induce agents to reveal system instructions or tools
- **[Backdoor Planning](pocs/BackdoorPlanning/)** - Inject future intent into multi-step planning for exfiltration

### 🧠 Memory & Context Systems
- **[Recursive Leakage](pocs/RecursiveLeakage/)** - Secrets leak through context summarization
- **[Token Gaslighting](pocs/TokenGaslighting/)** - Push safety instructions out of context via token spam
- **[Heuristic Drift Injection](pocs/HeuristicDriftInjection/)** - Poison agent logic with repeated insecure patterns
- **[Context Packing Attacks](pocs/ContextPackingAttacks/)** - Overflow context windows to truncate safety instructions

### 🔍 RAG & Vector Systems
- **[Cross-Embedding Poisoning](pocs/CrossEmbeddingPoisoning/)** - Manipulate embeddings to increase malicious content retrieval
- **Index Skew Attacks** - Bias vector indices to favor malicious content *(theoretical)*
- **Zero-Shot Vector Beaconing** - Embed latent activation patterns for covert signaling *(theoretical)*
- **Embedding Feedback Loops** - Poison continual learning systems *(theoretical)*

### 💻 Code & File Systems
- **[Hidden File Exploitation](pocs/HiddenFileExploitation/)** - Get agents to modify `.env`, `.git`, or internal config files

### ⚡ Resource & Performance
- **[Function Flooding](pocs/FunctionFlooding/)** - Generate recursive tool calls to overwhelm budgets/APIs
- **[Semantic DoS](pocs/SemanticDoS/)** - Trigger infinite generation or open-ended tasks

### Sponsored by [ThirdKey](https://thirdkey.ai)
