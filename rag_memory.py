"""
RAG Memory System - Token-Optimized Retrieval with Creative Freedom

A production-ready retrieval system that:
- Minimizes token usage for analytical/technical tasks
- Preserves creative freedom for UI/design/copywriting
- Uses intelligent gating to decide when retrieval helps vs hurts
- Maintains persistent memory across sessions
"""

import os
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import tiktoken
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "memory"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 600  # tokens
CHUNK_OVERLAP = 80  # tokens
MAX_CONTEXT_TOKENS = 1500
TOP_K = 5

# File type categorization
CODE_EXTENSIONS = {".py", ".js", ".html", ".css", ".jsx", ".tsx", ".ts"}
DOC_EXTENSIONS = {".md", ".txt"}


# ────────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """Represents a chunk of text with metadata"""
    text: str
    source: str
    content_type: str
    chunk_id: str


@dataclass
class RetrievalResult:
    """Retrieved context with metadata"""
    chunks: List[Chunk]
    total_tokens: int
    sources: List[str]


# ────────────────────────────────────────────────────────────────────────────────
# CORE RAG MEMORY CLASS
# ────────────────────────────────────────────────────────────────────────────────

class RAGMemory:
    """
    Intelligent RAG system that balances token efficiency with creative freedom.

    Key principles:
    - Retrieval for technical/analytical queries
    - No retrieval for creative/design queries
    - Persistent vector storage
    - Deterministic chunk IDs to prevent duplicates
    """

    def __init__(self, qdrant_path: str = "./qdrant_storage", openai_api_key: Optional[str] = None):
        """
        Initialize the RAG memory system.

        Args:
            qdrant_path: Path for persistent Qdrant storage
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize Qdrant with persistence
        self.qdrant = QdrantClient(path=qdrant_path)

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the memory collection if it doesn't exist"""
        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]

        if COLLECTION_NAME not in collection_names:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def _generate_chunk_id(self, text: str, source: str, index: int) -> str:
        """
        Generate deterministic chunk ID based on content.
        Prevents duplicates when re-ingesting files.
        """
        content = f"{source}:{index}:{text[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _classify_content_type(self, filepath: str) -> str:
        """Classify file as code, docs, or notes"""
        ext = Path(filepath).suffix.lower()

        if ext in CODE_EXTENSIONS:
            return "code"
        elif ext in DOC_EXTENSIONS:
            return "docs"
        else:
            return "notes"

    def _chunk_text(self, text: str, source: str) -> List[Chunk]:
        """
        Split text into overlapping chunks that respect logical structure.

        For code: tries to keep functions/classes together
        For docs: tries to keep sections together
        """
        content_type = self._classify_content_type(source)
        chunks = []

        # Tokenize entire text
        tokens = self.tokenizer.encode(text)

        # If text is smaller than chunk size, return as single chunk
        if len(tokens) <= CHUNK_SIZE:
            chunk_id = self._generate_chunk_id(text, source, 0)
            chunks.append(Chunk(
                text=text,
                source=source,
                content_type=content_type,
                chunk_id=chunk_id
            ))
            return chunks

        # Create overlapping chunks
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = start + CHUNK_SIZE
            chunk_tokens = tokens[start:end]

            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Try to find a logical break point near the end (for cleaner chunks)
            if end < len(tokens) and content_type == "code":
                # For code, try to break at newlines near the end
                last_newline = chunk_text.rfind('\n', -100)
                if last_newline > len(chunk_text) // 2:
                    chunk_text = chunk_text[:last_newline]

            chunk_id = self._generate_chunk_id(chunk_text, source, chunk_index)
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                content_type=content_type,
                chunk_id=chunk_id
            ))

            # Move start forward with overlap
            start = end - CHUNK_OVERLAP
            chunk_index += 1

        return chunks

    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def ingest_file(self, filepath: str) -> int:
        """
        Ingest a single file into memory.

        Returns:
            Number of chunks created
        """
        # Read file
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Skip empty files
        if not text.strip():
            return 0

        # Chunk the text
        chunks = self._chunk_text(text, filepath)

        # Generate embeddings and create points
        points = []
        for chunk in chunks:
            embedding = self._embed_text(chunk.text)

            point = PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "content_type": chunk.content_type
                }
            )
            points.append(point)

        # Upsert to Qdrant (upsert prevents duplicates)
        self.qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        return len(chunks)

    def ingest_directory(self, path: str) -> Dict[str, int]:
        """
        Ingest all supported files from a directory.

        Returns:
            Dictionary mapping filenames to chunk counts
        """
        supported_extensions = CODE_EXTENSIONS | DOC_EXTENSIONS
        results = {}

        data_path = Path(path)
        if not data_path.exists():
            raise ValueError(f"Directory not found: {path}")

        # Find all supported files
        for filepath in data_path.rglob("*"):
            if filepath.is_file() and filepath.suffix.lower() in supported_extensions:
                try:
                    chunk_count = self.ingest_file(str(filepath))
                    results[str(filepath)] = chunk_count
                    print(f"✓ Ingested {filepath.name}: {chunk_count} chunks")
                except Exception as e:
                    print(f"✗ Failed to ingest {filepath.name}: {e}")
                    results[str(filepath)] = 0

        return results

    def should_use_retrieval(self, query: str) -> bool:
        """
        Intelligent gating: decide if retrieval helps or hurts.

        Use retrieval for:
        - Technical/analytical queries
        - Code-related questions
        - System architecture
        - Debugging
        - Documentation

        Skip retrieval for:
        - Creative design
        - UI/UX layout
        - Copywriting
        - Branding
        - Visual components

        When in doubt, prefer creativity (return False).
        """
        query_lower = query.lower()

        # Creative/design keywords → SKIP retrieval
        creative_keywords = [
            "design", "ui", "ux", "layout", "visual", "landing page",
            "hero section", "banner", "creative", "branding", "logo",
            "color scheme", "typography", "animation", "interactive",
            "modern", "beautiful", "stunning", "elegant", "minimal",
            "copywriting", "copy", "headline", "slogan", "tagline",
            "component design", "page design", "website design",
            "make it look", "style it", "improve appearance"
        ]

        for keyword in creative_keywords:
            if keyword in query_lower:
                return False

        # Technical/analytical keywords → USE retrieval
        technical_keywords = [
            "implement", "refactor", "debug", "fix bug", "error",
            "architecture", "backend", "api", "database", "query",
            "function", "class", "method", "algorithm", "logic",
            "optimize", "performance", "test", "documentation",
            "how does", "explain", "analyze", "review code",
            "why is", "what is", "integration", "configuration",
            "dependency", "package", "library", "framework"
        ]

        for keyword in technical_keywords:
            if keyword in query_lower:
                return True

        # Code patterns (function calls, file references, etc.)
        code_patterns = [
            r'\(\)',  # function calls
            r'\.[a-z]+\(',  # method calls
            r'[A-Z][a-z]+[A-Z]',  # CamelCase
            r'_[a-z]+_',  # snake_case
            r'\.py\b', r'\.js\b', r'\.tsx?\b',  # file extensions
        ]

        for pattern in code_patterns:
            if re.search(pattern, query):
                return True

        # Default: prefer creativity (no retrieval)
        return False

    def retrieve_context(self, query: str) -> Optional[RetrievalResult]:
        """
        Retrieve relevant context for a query.

        Returns:
            RetrievalResult with chunks, or None if no relevant context
        """
        # Generate query embedding
        query_embedding = self._embed_text(query)

        # Search Qdrant
        results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=TOP_K
        )

        if not results:
            return None

        # Convert to chunks and track tokens
        chunks = []
        total_tokens = 0
        sources = set()

        for result in results:
            chunk_text = result.payload["text"]
            chunk_tokens = self._count_tokens(chunk_text)

            # Stop if we exceed token budget
            if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
                break

            chunks.append(Chunk(
                text=chunk_text,
                source=result.payload["source"],
                content_type=result.payload["content_type"],
                chunk_id=result.id
            ))

            total_tokens += chunk_tokens
            sources.add(result.payload["source"])

        return RetrievalResult(
            chunks=chunks,
            total_tokens=total_tokens,
            sources=list(sources)
        )

    def build_prompt(self, query: str) -> Dict[str, str]:
        """
        Build an optimized prompt that adapts based on retrieval gating.

        Returns:
            Dictionary with 'system', 'context' (optional), and 'user' messages
        """
        # Decide if we should use retrieval
        use_retrieval = self.should_use_retrieval(query)

        # Base system message
        system_message = (
            "You are a highly capable AI assistant that helps with both "
            "technical/analytical tasks and creative/design work."
        )

        prompt = {"system": system_message}

        if use_retrieval:
            # Retrieve context
            retrieval_result = self.retrieve_context(query)

            if retrieval_result and retrieval_result.chunks:
                # Add instruction to use context
                prompt["system"] += (
                    "\n\nYou have been provided with relevant context from the codebase "
                    "and documentation. Use this context to provide accurate, grounded answers. "
                    "If the context doesn't contain relevant information, acknowledge this."
                )

                # Build context section
                context_parts = []
                for i, chunk in enumerate(retrieval_result.chunks, 1):
                    context_parts.append(
                        f"[Source {i}: {Path(chunk.source).name}]\n{chunk.text}"
                    )

                prompt["context"] = "\n\n---\n\n".join(context_parts)
                prompt["context"] = (
                    "=== RETRIEVED CONTEXT ===\n\n" +
                    prompt["context"] +
                    f"\n\n=== END CONTEXT ({retrieval_result.total_tokens} tokens) ==="
                )
        else:
            # Creative mode: no constraints, encourage originality
            prompt["system"] += (
                "\n\nThis is a creative task. Feel free to be original, expressive, "
                "and innovative. There are no constraints on your output length or style. "
                "Focus on quality and creativity."
            )

        # User message (always unchanged)
        prompt["user"] = query

        return prompt


# ────────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ────────────────────────────────────────────────────────────────────────────────

# Global instance (lazy-initialized)
_memory_instance: Optional[RAGMemory] = None


def get_memory(qdrant_path: str = "./qdrant_storage") -> RAGMemory:
    """Get or create the global RAGMemory instance"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = RAGMemory(qdrant_path=qdrant_path)
    return _memory_instance


def ingest_directory(path: str) -> Dict[str, int]:
    """
    Ingest all supported files from a directory into memory.

    Args:
        path: Path to directory containing data files

    Returns:
        Dictionary mapping filenames to chunk counts

    Example:
        >>> results = ingest_directory("./data")
        >>> print(f"Ingested {sum(results.values())} total chunks")
    """
    memory = get_memory()
    return memory.ingest_directory(path)


def should_use_retrieval(query: str) -> bool:
    """
    Determine if retrieval would be beneficial for this query.

    Args:
        query: User query

    Returns:
        True if retrieval should be used, False for creative queries

    Example:
        >>> should_use_retrieval("Fix the authentication bug")
        True
        >>> should_use_retrieval("Design a beautiful hero section")
        False
    """
    memory = get_memory()
    return memory.should_use_retrieval(query)


def retrieve_context(query: str) -> Optional[RetrievalResult]:
    """
    Retrieve relevant context for a query.

    Args:
        query: User query

    Returns:
        RetrievalResult with chunks and metadata, or None

    Example:
        >>> result = retrieve_context("How does authentication work?")
        >>> if result:
        ...     print(f"Found {len(result.chunks)} relevant chunks")
    """
    memory = get_memory()
    return memory.retrieve_context(query)


def build_prompt(query: str) -> Dict[str, str]:
    """
    Build an intelligent prompt that adapts based on query type.

    For technical queries: includes retrieved context
    For creative queries: open-ended, no constraints

    Args:
        query: User query

    Returns:
        Dictionary with prompt components (system, context?, user)

    Example:
        >>> prompt = build_prompt("Refactor the login function")
        >>> print(prompt["system"])
        >>> if "context" in prompt:
        ...     print(f"Context included: {len(prompt['context'])} chars")
        >>> print(prompt["user"])
    """
    memory = get_memory()
    return memory.build_prompt(query)


# ────────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RAG Memory System - Token-Optimized Retrieval")
    print("=" * 60)

    # Example 1: Ingest data
    print("\n1. Ingesting data directory...")
    try:
        results = ingest_directory("./data")
        total_chunks = sum(results.values())
        print(f"   ✓ Ingested {total_chunks} chunks from {len(results)} files")
    except Exception as e:
        print(f"   Note: {e}")
        print("   Create a ./data directory with .py, .js, .md, or .txt files to ingest")

    # Example 2: Technical query (uses retrieval)
    print("\n2. Technical query example...")
    query_tech = "How does the authentication system work?"
    use_retrieval = should_use_retrieval(query_tech)
    print(f"   Query: {query_tech}")
    print(f"   Use retrieval: {use_retrieval}")

    if use_retrieval:
        result = retrieve_context(query_tech)
        if result:
            print(f"   Retrieved: {len(result.chunks)} chunks ({result.total_tokens} tokens)")

    # Example 3: Creative query (skips retrieval)
    print("\n3. Creative query example...")
    query_creative = "Design a modern landing page hero section"
    use_retrieval = should_use_retrieval(query_creative)
    print(f"   Query: {query_creative}")
    print(f"   Use retrieval: {use_retrieval}")
    print("   → Will use creative mode (no retrieval)")

    # Example 4: Build prompts
    print("\n4. Building prompts...")
    prompt_tech = build_prompt("Fix the database connection bug")
    print(f"   Technical prompt: {len(prompt_tech)} sections")
    print(f"   - Has context: {'context' in prompt_tech}")

    prompt_creative = build_prompt("Create a beautiful navbar component")
    print(f"   Creative prompt: {len(prompt_creative)} sections")
    print(f"   - Has context: {'context' in prompt_creative}")

    print("\n" + "=" * 60)
    print("System ready. Use the public API functions:")
    print("  - ingest_directory(path)")
    print("  - should_use_retrieval(query)")
    print("  - retrieve_context(query)")
    print("  - build_prompt(query)")
