"""Global application constants."""

# File handling constants
SUPPORTED_FILE_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt"}

# Ingestion & chunking defaults
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 150

# Retrieval defaults
DEFAULT_RETRIEVAL_TOP_K: int = 4

# Ollama model identifiers
DEFAULT_EMBED_MODEL: str = "nomic-embed-text"
DEFAULT_CHAT_MODEL: str = "llama3.2:3b"

# Vectorstore defaults (Supabase pgvector)
DEFAULT_VECTOR_TABLE: str = "document_embeddings"
