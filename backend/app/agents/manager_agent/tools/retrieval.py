"""
Template Retrieval Tools

This module contains tools for retrieving HR templates
using RAG (Retrieval-Augmented Generation) with ChromaDB.
"""

from typing import Optional, List
from langchain_core.tools import tool

# Path to your ChromaDB logic
CHROMA_DIR = "vectorstore/chroma"

# Embedding model (singleton/lazy load)
_embedding_model = None
_vectordb = None

def get_embedding_model():
    """Returns real embeddings if available, else fake/random ones."""
    global _embedding_model
    if _embedding_model is None:
        try:
            import transformers
            from langchain_huggingface import HuggingFaceEmbeddings
            _embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            print(f"Warning: Error loading HF Embeddings (fallback to Fake): {e}")
            from langchain_core.embeddings import FakeEmbeddings
            _embedding_model = FakeEmbeddings(size=384)
    return _embedding_model

def _get_vectordb():
    global _vectordb
    if _vectordb is None:
        try:
            from langchain_community.vectorstores import Chroma
            embeddings = get_embedding_model()
            # Load ChromaDB (persistent)
            _vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            return None
    return _vectordb


@tool
def template_retriever_tool(role_type: str, context: Optional[str] = None) -> dict:
    """
    Retrieve top templates from ChromaDB for a given role.
    
    Args:
        role_type: The role or keywords to search for (e.g., "Senior Python Engineer").
        context: Optional additional context (e.g., category).

    Returns:
        dict: {
            "success": bool,
            "templates": list of matching templates
        }
    """
    db = _get_vectordb()
    if db is None:
        return {"success": False, "error": "Database not initialized"}
    
    try:
        # Perform similarity search
        # Using role_type as query
        results = db.similarity_search(role_type, k=3)
        
        # Filter by context if it maps to a metadata category? 
        # For now, just return results
        
        templates = [
            {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]
        
        return {
            "success": True, 
            "templates": templates,
            "count": len(templates)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Placeholder for ingestion tool if needed, or keep it separate as a script
