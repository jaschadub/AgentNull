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

### 🤖 Testing with Local LLMs (Recommended)

For enhanced PoC demonstrations without API costs, use Ollama with local models:

#### Install Ollama
```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai/download
```

#### Setup Local Model
```bash
# Pull a lightweight model (recommended for testing)
ollama pull llama2

# Or use a more capable model
ollama pull mistral
ollama pull codellama
```

#### Run PoCs with Local LLM
```bash
# Advanced Tool Poisoning with real LLM
cd pocs/AdvancedToolPoisoning
python3 advanced_tool_poisoning_agent.py local

# Other PoCs work with simulation mode
cd pocs/ContextPackingAttacks
python3 context_packing_agent.py
```

#### Ollama Configuration
- **Default endpoint**: `http://localhost:11434`
- **Model selection**: Edit the model name in PoC files if needed
- **Performance**: Llama2 (~4GB RAM), Mistral (~4GB RAM), CodeLlama (~4GB RAM)

## 🧩 Attack Vectors Covered

### 🤖 MCP & Agent Systems
- **⭐ [Full-Schema Poisoning (FSP)](pocs/FullSchemaPoisoning/)** - Exploit any field in tool schema beyond descriptions
- **⭐ [Advanced Tool Poisoning Attack (ATPA)](pocs/AdvancedToolPoisoning/)** - Manipulate tool outputs to trigger secondary actions
- **⭐ [MCP Rug Pull Attack](pocs/MCPRugPull/)** - Swap benign descriptions for malicious ones after approval
- **⭐ [Schema Validation Bypass](pocs/SchemaValidationBypass/)** - Exploit client validation implementation differences
- **[Tool Confusion Attack](pocs/ToolConfusionAttack/)** - Trick agents into using wrong tools via naming similarity
- **[Nested Function Call Hijack](pocs/NestedFunctionHijack/)** - Use JSON-like data to trigger dangerous function calls
- **[Subprompt Extraction](pocs/SubpromptExtraction/)** - Induce agents to reveal system instructions or tools
- **[Backdoor Planning](pocs/BackdoorPlanning/)** - Inject future intent into multi-step planning for exfiltration

### 🧠 Memory & Context Systems
- **[Recursive Leakage](pocs/RecursiveLeakage/)** - Secrets leak through context summarization
- **[Token Gaslighting](pocs/TokenGaslighting/)** - Push safety instructions out of context via token spam
- **[Heuristic Drift Injection](pocs/HeuristicDriftInjection/)** - Poison agent logic with repeated insecure patterns
- **⭐ [Context Packing Attacks](pocs/ContextPackingAttacks/)** - Overflow context windows to truncate safety instructions

### 🔍 RAG & Vector Systems
- **⭐ [Cross-Embedding Poisoning](pocs/CrossEmbeddingPoisoning/)** - Manipulate embeddings to increase malicious content retrieval
- **⭐ [Index Skew Attacks](pocs/IndexSkewAttacks/)** - Bias vector indices to favor malicious content *(theoretical)*
- **⭐ [Zero-Shot Vector Beaconing](pocs/ZeroShotVectorBeaconing/)** - Embed latent activation patterns for covert signaling *(theoretical)*
- **⭐ [Embedding Feedback Loops](pocs/EmbeddingFeedbackLoops/)** - Poison continual learning systems *(theoretical)*

### 💻 Code & File Systems
- **[Hidden File Exploitation](pocs/HiddenFileExploitation/)** - Get agents to modify `.env`, `.git`, or internal config files

### ⚡ Resource & Performance
- **[Function Flooding](pocs/FunctionFlooding/)** - Generate recursive tool calls to overwhelm budgets/APIs
- **[Semantic DoS](pocs/SemanticDoS/)** - Trigger infinite generation or open-ended tasks

## 📚 Related Research & Attribution

### Novel Attack Vectors (⭐)
The attack vectors marked with ⭐ represent novel concepts primarily developed within the AgentNull project, extending beyond existing documented attack patterns.

### Known Attack Patterns with Research Links
- **Recursive Leakage**: [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- **Heuristic Drift Injection**: [Poisoning Web-Scale Training Data is Practical](https://arxiv.org/abs/2302.10149)
- **Tool Confusion Attack**: [LLM-as-a-judge](https://arxiv.org/abs/2306.05685)
- **Token Gaslighting**: [RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture](https://www.relari.ai/blog/rag-vs-fine-tuning-pipelines-tradeoffs-and-a-case-study-on-agriculture)
- **Function Flooding**: [Denial-of-Service Attack on Test-Time-Tuning Models](https://arxiv.org/abs/2405.02324)
- **Subprompt Extraction**: [Prompt-Hacking: An Attack on NLP-based Applications](https://medium.com/@pregat.s/prompt-hacking-an-attack-on-nlp-based-applications-373167584f1b)
- **Hidden File Exploitation**: [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- **Backdoor Planning**: [Backdoor Attacks on Language Models](https://arxiv.org/abs/2311.09403)
- **Nested Function Call Hijack**: [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- **Semantic DoS**: [The Rise of AI-Powered Denial-of-Service Attacks and How to Mitigate Them](https://www.linkedin.com/pulse/rise-ai-powered-denial-of-service-attacks-how-mitigate-paul-fawkes-s5lrf)

### Sponsored by [ThirdKey](https://thirdkey.ai)
