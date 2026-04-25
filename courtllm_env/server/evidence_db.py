# server/evidence_db.py — Locked corpus loader

import json
from typing import List, Dict
from pathlib import Path

class EvidenceDatabase:
    """
    Manages the locked evidence corpus.
    Sources are read-only with exact-ID access only.
    """

    def __init__(self, corpus_path: str = None):
        self.corpus_path = corpus_path or self._default_path()
        self.corpus = []
        self._load_corpus()

    def _default_path(self) -> str:
        """Get default path to evidence corpus"""
        return str(Path(__file__).parent.parent / "data" / "evidence_corpus.jsonl")

    def _load_corpus(self):
        """Load evidence corpus from JSONL file"""
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                self.corpus = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            # Generate minimal corpus if file doesn't exist
            self.corpus = self._generate_minimal_corpus()

    def _generate_minimal_corpus(self) -> List[Dict]:
        """Generate a minimal corpus for testing"""
        minimal = []
        domains = ["biomedical", "historical", "legal", "financial", "scientific"]

        for domain_idx, domain in enumerate(domains):
            for i in range(20):  # 20 entries per domain = 100 total
                source_id = f"{domain.upper()}_{i:03d}"
                minimal.append({
                    "source_id": source_id,
                    "title": f"{domain.title()} Source {i}",
                    "domain": domain,
                    "snippet": f"This is a factual statement about {domain} topic {i}.",
                    "url": f"https://example.com/{domain}/{i}",
                    "is_verified": True
                })

        return minimal

    def get_by_id(self, source_id: str) -> Dict:
        """Get source by exact ID match"""
        for source in self.corpus:
            if source["source_id"] == source_id:
                return source
        return None

    def get_by_domain(self, domain: str, limit: int = 50) -> List[Dict]:
        """Get sources filtered by domain"""
        return [s for s in self.corpus if s.get("domain") == domain][:limit]

    def get_all(self) -> List[Dict]:
        """Get entire corpus"""
        return self.corpus

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Simple keyword search in corpus"""
        query_lower = query.lower()
        results = []

        for source in self.corpus:
            snippet = source.get("snippet", "").lower()
            title = source.get("title", "").lower()

            if query_lower in snippet or query_lower in title:
                results.append(source)
                if len(results) >= limit:
                    break

        return results

    def get_random_sample(self, n: int, domain: str = None) -> List[Dict]:
        """Get random sample of N sources, optionally filtered by domain"""
        import random

        pool = self.corpus
        if domain:
            pool = [s for s in pool if s.get("domain") == domain]

        return random.sample(pool, min(n, len(pool)))
