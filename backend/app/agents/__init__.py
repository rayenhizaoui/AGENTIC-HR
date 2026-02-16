"""
Agents Module - Multi-Agent System for HR Recruitment

This package contains:
- shared/: Common state protocol and utilities
- recruiter_agent/: Lead Recruiter agent (CV parsing, ranking)
- manager_agent/: Hiring Manager agent (templates, offers)
- supervisor.py: Main routing supervisor
"""

from .supervisor import supervisor_graph, run_supervisor

__all__ = ["supervisor_graph", "run_supervisor"]
