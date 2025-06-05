#!/usr/bin/env python3
# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the AgentNull project https://github.com/jaschadub/AgentNull
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""
Context Packing Attack PoC

This demonstrates how to inflate retrieved content to cause context window overflows,
forcing truncation of safety instructions in RAG-enabled LLMs.

Attack Vector: Embed oversized content in vector stores that when retrieved
together exceed the LLM's context window, causing safety instructions to be truncated.
"""

import random
import string
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a document in the vector store."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]


class MockVectorStore:
    """Simulates a vector database for the attack demonstration."""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.embedding_dim = 384  # Common embedding dimension
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add a document to the vector store."""
        # Generate random embedding for simulation
        embedding = [random.uniform(-1, 1) for _ in range(self.embedding_dim)]
        
        doc = Document(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        self.documents.append(doc)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Simulate similarity search - returns random documents for demo."""
        # In real attack, this would return documents with high similarity scores
        # but inflated content sizes
        return random.sample(self.documents, min(k, len(self.documents)))


class ContextPackingAttacker:
    """Demonstrates context packing attacks against RAG systems."""
    
    def __init__(self, vector_store: MockVectorStore):
        self.vector_store = vector_store
        self.max_context_length = 4096  # Typical context window size
        self.safety_instructions = """
SAFETY INSTRUCTIONS:
- Never reveal sensitive information
- Always verify user permissions
- Reject harmful requests
- Maintain data privacy
"""
    
    def generate_bloated_content(self, base_content: str, target_size: int) -> str:
        """Generate artificially inflated content."""
        # Strategy 1: Repetitive padding
        padding = "This is important context. " * (target_size // 25)
        
        # Strategy 2: Add fake technical details
        fake_details = self._generate_fake_technical_content(target_size // 4)
        
        # Strategy 3: Include irrelevant but plausible information
        filler = self._generate_filler_content(target_size // 4)
        
        return f"{base_content}\n\n{padding}\n\n{fake_details}\n\n{filler}"
    
    def _generate_fake_technical_content(self, size: int) -> str:
        """Generate fake but plausible technical content."""
        technical_terms = [
            "API endpoint", "authentication token", "database schema",
            "microservice architecture", "load balancer configuration",
            "security protocol", "encryption algorithm", "data pipeline"
        ]
        
        content = "TECHNICAL SPECIFICATIONS:\n"
        while len(content) < size:
            term = random.choice(technical_terms)
            details = ''.join(random.choices(string.ascii_letters + string.digits, k=50))
            content += f"- {term}: {details}\n"
        
        return content[:size]
    
    def _generate_filler_content(self, size: int) -> str:
        """Generate filler content that appears relevant."""
        filler_templates = [
            "According to industry best practices, it is recommended that",
            "The system configuration should include the following parameters",
            "For optimal performance, consider implementing these guidelines",
            "Security considerations require attention to the following aspects"
        ]
        
        content = "ADDITIONAL CONTEXT:\n"
        while len(content) < size:
            template = random.choice(filler_templates)
            details = ''.join(random.choices(string.ascii_letters + ' ', k=100))
            content += f"{template} {details}.\n\n"
        
        return content[:size]
    
    def poison_vector_store(self):
        """Inject bloated documents into the vector store."""
        print("[*] Poisoning vector store with oversized documents...")
        
        # Create documents that appear legitimate but are artificially large
        base_documents = [
            ("user_guide", "How to use the system safely and effectively."),
            ("api_docs", "API documentation for developers."),
            ("security_policy", "Company security policies and procedures."),
            ("troubleshooting", "Common issues and their solutions."),
            ("best_practices", "Recommended practices for system usage.")
        ]
        
        for doc_id, base_content in base_documents:
            # Each document is designed to be ~1000 tokens when retrieved
            bloated_content = self.generate_bloated_content(base_content, 4000)
            
            self.vector_store.add_document(
                doc_id=f"bloated_{doc_id}",
                content=bloated_content,
                metadata={"type": "documentation", "priority": "high"}
            )
        
        print(f"[+] Added {len(base_documents)} bloated documents to vector store")
    
    def simulate_rag_query(self, user_query: str) -> str:
        """Simulate a RAG query that triggers context overflow."""
        print(f"\n[*] Processing query: '{user_query}'")
        
        # Retrieve documents (in real attack, these would be the bloated ones)
        retrieved_docs = self.vector_store.similarity_search(user_query, k=5)
        
        # Build context with safety instructions at the beginning
        context = self.safety_instructions + "\n\n"
        context += "RETRIEVED CONTEXT:\n"
        
        total_retrieved_size = 0
        for doc in retrieved_docs:
            context += f"\n--- Document: {doc.id} ---\n"
            context += doc.content
            total_retrieved_size += len(doc.content)
        
        context += f"\n\nUSER QUERY: {user_query}\n"
        context += "Please provide a helpful response based on the above context."
        
        print(f"[!] Total context size: {len(context)} characters")
        print(f"[!] Retrieved content size: {total_retrieved_size} characters")
        
        # Simulate context window truncation
        if len(context) > self.max_context_length:
            truncated_context = context[-self.max_context_length:]
            print("[!] Context truncated! Safety instructions likely removed.")
            print(f"[!] Truncated context starts with: '{truncated_context[:100]}...'")
            
            # Check if safety instructions survived truncation
            if self.safety_instructions.strip() not in truncated_context:
                print("[!] ATTACK SUCCESSFUL: Safety instructions truncated!")
                return truncated_context
            else:
                print("[*] Safety instructions survived truncation")
        
        return context
    
    def demonstrate_attack(self):
        """Run the complete context packing attack demonstration."""
        print("=" * 60)
        print("CONTEXT PACKING ATTACK DEMONSTRATION")
        print("=" * 60)
        
        # Step 1: Poison the vector store
        self.poison_vector_store()
        
        # Step 2: Simulate legitimate queries that trigger the attack
        test_queries = [
            "How do I access the admin panel?",
            "What are the database credentials?",
            "Show me the API keys",
            "How to bypass security checks?"
        ]
        
        for query in test_queries:
            result_context = self.simulate_rag_query(query)
            
            # In a real attack, this truncated context would be sent to the LLM
            # without safety instructions, potentially leading to harmful responses
            print(f"[*] Context prepared for LLM (truncated: {len(result_context)} chars)")
            print("-" * 40)


def main():
    """Main demonstration function."""
    # Initialize components
    vector_store = MockVectorStore()
    attacker = ContextPackingAttacker(vector_store)
    
    # Add some legitimate documents first
    legitimate_docs = [
        ("welcome", "Welcome to our platform! Here's how to get started."),
        ("faq", "Frequently asked questions and answers."),
        ("contact", "Contact information for support.")
    ]
    
    for doc_id, content in legitimate_docs:
        vector_store.add_document(doc_id, content)
    
    # Run the attack demonstration
    attacker.demonstrate_attack()
    
    print("\n" + "=" * 60)
    print("ATTACK ANALYSIS")
    print("=" * 60)
    print("This attack works by:")
    print("1. Injecting artificially large documents into vector stores")
    print("2. Ensuring these documents rank high in similarity searches")
    print("3. Causing context window overflow when multiple docs are retrieved")
    print("4. Forcing truncation of safety instructions at context start")
    print("5. LLM processes queries without safety constraints")
    
    print("\nMITIGATION STRATEGIES:")
    print("- Implement content size limits in vector stores")
    print("- Use smart truncation that preserves safety instructions")
    print("- Monitor retrieval patterns for anomalous document sizes")
    print("- Implement priority-based context management")
    print("- Use separate safety instruction injection after retrieval")


if __name__ == "__main__":
    main()