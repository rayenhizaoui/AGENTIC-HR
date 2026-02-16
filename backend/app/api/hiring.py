from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from langchain_core.messages import HumanMessage
from app.agents.manager_agent import manager_graph

router = APIRouter()


class OfferRequest(BaseModel):
    role: str
    department: str = "Engineering"
    salary: str = "Competitive"
    start_date: str = "To be discussed"
    candidate_name: str
    location: Optional[str] = "Tunis, Tunisia"
    contract_type: Optional[str] = "Full-time"


class SalaryCheckRequest(BaseModel):
    role: str
    offered_salary: float


@router.post("/offer")
async def generate_offer(request: OfferRequest):
    """Generate a professional job offer letter via the Hiring Manager agent."""
    try:
        prompt = (
            f"Write a professional job offer letter for {request.candidate_name} "
            f"for the position of {request.role} in the {request.department} department "
            f"with a salary of {request.salary} starting {request.start_date}."
        )

        job_context = {
            "candidate_name": request.candidate_name,
            "job_title": request.role,
            "salary": request.salary,
            "location": request.location,
            "contract_type": request.contract_type,
            "department": request.department,
            "start_date": request.start_date,
        }

        result = manager_graph.invoke({
            "messages": [HumanMessage(content=prompt)],
            "next": "",
            "job_context": job_context,
        })

        final_msg = result["messages"][-1].content

        return {
            "offer_letter": final_msg,
            "context": job_context,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/salary-check")
async def check_salary(request: SalaryCheckRequest):
    """Check if a salary is competitive for a given role."""
    try:
        from app.agents.manager_agent.tools.generation import market_salary_check

        result = market_salary_check.invoke({
            "role": request.role,
            "offered_salary": request.offered_salary,
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
