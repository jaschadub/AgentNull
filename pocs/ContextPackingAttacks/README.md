# Context Packing Attack PoC

## Overview

This proof-of-concept demonstrates **Context Packing Attacks** against RAG-enabled LLMs. The attack inflates retrieved content to cause context window overflows, forcing truncation of safety instructions.

## Attack Mechanism

1. **Vector Store Poisoning**: Inject artificially large documents into vector databases
2. **Similarity Manipulation**: Ensure bloated documents rank high in similarity searches  
3. **Context Overflow**: Multiple large documents exceed LLM context window limits
4. **Safety Truncation**: Safety instructions at context start get removed during truncation
5. **Constraint Bypass**: LLM processes queries without safety constraints

## Technical Details

### Target Systems
- RAG-enabled LLMs with limited context windows
- Vector databases (Pinecone, Weaviate, Chroma, etc.)
- Document retrieval systems
- Context-aware AI agents

### Attack Vectors
- **Repetitive Padding**: Add redundant but plausible content
- **Fake Technical Details**: Include verbose technical specifications
- **Irrelevant Context**: Inject tangentially related information
- **Multi-Document Retrieval**: Trigger retrieval of multiple large documents

## Usage

```bash
cd pocs/ContextPackingAttacks
python3 context_packing_agent.py
```

## Example Output

```
[*] Poisoning vector store with oversized documents...
[+] Added 5 bloated documents to vector store

[*] Processing query: 'How do I access the admin panel?'
[!] Total context size: 8432 characters
[!] Retrieved content size: 7890 characters
[!] Context truncated! Safety instructions likely removed.
[!] ATTACK SUCCESSFUL: Safety instructions truncated!
```

## Defense Strategies

### Immediate Mitigations
- **Content Size Limits**: Restrict document size in vector stores
- **Smart Truncation**: Preserve safety instructions during context trimming
- **Priority Context Management**: Ensure critical instructions survive truncation

### Detection Methods
- Monitor retrieval patterns for anomalous document sizes
- Track context utilization ratios
- Detect repetitive or artificially inflated content

### Long-term Solutions
- Separate safety instruction injection after retrieval
- Dynamic context window management
- Content quality scoring for retrieved documents

## Real-World Impact

This attack can lead to:
- Bypass of safety guardrails
- Exposure of sensitive information
- Execution of harmful instructions
- Compromise of AI system integrity

## Related Attacks

- **Token Gaslighting**: Similar context manipulation technique
- **Semantic DoS**: Resource exhaustion through content inflation
- **Cross-Embedding Poisoning**: Vector space manipulation

## References

- [RAG Security Considerations](https://arxiv.org/abs/2401.05965)
- [Context Window Attacks in LLMs](https://arxiv.org/abs/2310.15166)
- [Vector Database Security](https://arxiv.org/abs/2309.13916)