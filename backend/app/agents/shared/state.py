"""
Global State Protocol for the ATIA-HR Multi-Agent System.
This defines the shared state that flows between all agents in the hierarchy.

ATIA-HR State includes:
- Conversation history (messages)
- Routing decision (next)
- Job/recruitment context (job_context)
- Intent detection results (current_task, filters)
- User preferences (language, focus region)
- Cached results (search, ranking)
"""

from typing import Annotated, Any, Optional, TypedDict
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    The global state shared across all agents in the ATIA-HR system.

    Attributes:
        messages: Conversation history (auto-accumulated via operator.add).
        next: The next agent/node to route to (used by conditional edges).
        job_context: Shared dict for passing data between agents
                     (extracted skills, candidate IDs, job requirements, etc.).
        current_task: Detected intent/task type (e.g., "job_search", "cv_ranking",
                      "offer_generation", "salary_check", "clarification_needed").
        filters: Extracted entities from user query
                 (e.g., {"skill": "python", "location": "remote", "level": "junior"}).
        user_preferences: Persistent user preferences
                          (e.g., {"language": "FR", "focus": "Tunisie/remote"}).
        search_results: Cached job search results from the latest query.
        ranking_results: Cached LLM ranking results from the latest analysis.
    """
    messages: Annotated[list[BaseMessage], operator.add]
    next: str
    job_context: dict[str, Any]
    current_task: Optional[str]
    filters: Optional[dict[str, Any]]
    user_preferences: Optional[dict[str, Any]]
    search_results: Optional[list]
    ranking_results: Optional[list]
    reasoning_log: Annotated[list[str], operator.add]
