"""
Hiring Manager Agent Tools - Group 2 Workspace

This package contains all tools for the Hiring Manager Agent:
- retrieval.py: ChromaDB template retrieval
- generation.py: Offer generation and validation
"""

# Only import what actually exists in retrieval.py
from .retrieval import template_retriever_tool
from .generation import job_offer_generator, offer_validator_tool, market_salary_check

__all__ = [
    # Retrieval
    "template_retriever_tool",
    # Generation & Validation
    "job_offer_generator",
    "offer_validator_tool",
    "market_salary_check",
]
