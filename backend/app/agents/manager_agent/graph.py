"""
ATIA-HR Hiring Manager Agent Graph

This agent handles:
- RAG-based template retrieval (ChromaDB)
- Job offer generation and validation
- Salary market checks
- Email drafting for candidates
- Multilingual support (FR/EN/Arabic)
"""

import re
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.shared.state import AgentState
from app.agents.shared.utils import logger, extract_last_message

# Import tools from the tools folder
from .tools.retrieval import template_retriever_tool
from .tools.generation import (
    job_offer_generator,
    offer_validator_tool,
    market_salary_check,
    DEFAULT_OFFER_TEMPLATE,
)

# List of all available tools for this agent
MANAGER_TOOLS = [
    template_retriever_tool,
    job_offer_generator,
    offer_validator_tool,
    market_salary_check,
]


def agent_node(state: AgentState) -> dict:
    """
    Main processing node for the Hiring Manager Agent.

    Routes to the appropriate tool based on keyword detection in the user query.
    """
    # Extract the last HumanMessage
    user_query = "No query provided"
    messages = state.get("messages", [])
    if messages:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        if user_query == "No query provided" and messages:
            last_message = messages[-1]
            user_query = last_message.content if hasattr(last_message, 'content') else str(last_message)

    job_context = state.get("job_context", {})
    query_lower = user_query.lower()
    
    # ─── Context Extraction ──────────────────────────────────────
    # Try to extract salary from prompt if present (e.g. "salary of 50000")
    salary_match = re.search(r'salary of\s+([$\d,\.]+)', user_query, re.I) or \
                   re.search(r'salary\s+([$\d,\.]+)', user_query, re.I)
    if salary_match:
        raw_sal = salary_match.group(1).replace(",", "").replace("$", "")
        try:
            job_context["salary"] = str(raw_sal)
            # Also store numeric for checks
            job_context["salary_numeric"] = float(raw_sal)
        except:
            pass
            
    # Try to extract role
    role_match = re.search(r'position of\s+(.+?)(?:\s+with|\s+at|\.|$)', user_query, re.I) or \
                 re.search(r'role\s+(.+?)(?:\s+with|\s+at|\.|$)', user_query, re.I)
    if role_match:
        job_context["job_title"] = role_match.group(1).strip()
        
    response_content = ""

    # ---------------------------------------------------------
    # ROUTE: Job Offer Generation (Prioritized)
    # ---------------------------------------------------------
    if "offer" in query_lower or "generate" in query_lower or "draft" in query_lower or "write" in query_lower:
        try:
            # Get candidate data from context or use defaults
            candidate_data = {
                "name": job_context.get("candidate_name", "[CANDIDATE NAME]"),
                "skills": job_context.get("extracted_skills", {}).get("skills", []),
                "experience_years": job_context.get("extracted_skills", {}).get("experience_years", 0),
            }

            job_data = {
                "title": job_context.get("job_title", "Software Engineer"),
                "company": "ATIA Club ESB",
                "location": job_context.get("location", "Tunis, Tunisia"),
                "salary": job_context.get("salary", "Competitive"),
                "currency": "TND",
                "contract_type": "Full-time",
                "start_date": "To be discussed",
                "response_deadline": "2 weeks from receipt",
                "hiring_manager": "HR Department",
                "date": "As of today",
                "department": "Engineering",
            }

            # Try to retrieve a matching template from ChromaDB
            template = DEFAULT_OFFER_TEMPLATE
            try:
                retrieval_result = template_retriever_tool.invoke({
                    "role_type": job_data["title"]
                })
                if retrieval_result.get("success") and retrieval_result.get("templates"):
                    template = retrieval_result["templates"][0]["text"]
            except Exception:
                pass  # Use default template

            # Generate the offer
            offer_result = job_offer_generator.invoke({
                "template": template,
                "candidate_data": candidate_data,
                "job_data": job_data,
            })

            if offer_result.get("success"):
                offer_text = offer_result["offer_text"]

                # Validate the offer
                validation = offer_validator_tool.invoke({
                    "generated_text": offer_text
                })

                response_content = (
                    f"### 📝 Generated Job Offer\n\n"
                    f"{offer_text}\n\n"
                    f"---\n"
                    f"### ✅ Validation Report\n"
                    f"- **Valid:** {'Yes ✅' if validation.get('valid') else 'No ⚠️'}\n"
                )
                if validation.get("unfilled_placeholders"):
                    response_content += f"- **Unfilled Placeholders:** {', '.join(validation['unfilled_placeholders'])}\n"
                if validation.get("warnings"):
                    response_content += f"- **Warnings:** {', '.join(validation['warnings'])}\n"
                if validation.get("suggestions"):
                    response_content += "\n**Suggestions:**\n"
                    for s in validation["suggestions"]:
                        response_content += f"  - {s}\n"
            else:
                response_content = f"❌ Could not generate offer: {offer_result.get('error', 'Unknown error')}"

        except Exception as e:
            response_content = f"❌ Error generating offer: {str(e)}"
            import traceback
            print(traceback.format_exc())

    # ---------------------------------------------------------
    # ROUTE: Salary Check
    # ---------------------------------------------------------
    elif "salary" in query_lower or "market" in query_lower or "compensation" in query_lower:
        # Try to extract role and salary from the message if not already done
        salary_match = re.search(r'(\d[\d,\.]+)', user_query)
        offered_salary = job_context.get("salary_numeric", 0)
        if not offered_salary and salary_match:
            try:
                offered_salary = float(salary_match.group(1).replace(",", ""))
            except:
                offered_salary = 0

        role = job_context.get("job_title", "Software Engineer")
        
        if offered_salary > 0:
            try:
                result = market_salary_check.invoke({
                    "role": role,
                    "offered_salary": offered_salary,
                })
                response_content = (
                    f"### 💰 Salary Market Check\n\n"
                    f"**Role:** {role}\n"
                    f"**Offered Salary:** {offered_salary:,.0f}\n\n"
                    f"**Result:** {result.get('recommendation', 'No data')}\n\n"
                )
                if result.get("market_median"):
                    response_content += (
                        f"| Metric | Value |\n"
                        f"|--------|-------|\n"
                        f"| Market Min | {result['market_min']:,} |\n"
                        f"| Market Median | {result['market_median']:,} |\n"
                        f"| Market Max | {result['market_max']:,} |\n"
                        f"| Deviation | {result['deviation_percent']:+.1f}% |\n"
                    )
                
                # Add a nudge that they can still generate the offer
                response_content += "\n\n💡 *To generate the offer letter anyway, click 'Generate Offer Letter' again or adjust the salary.*"
                
            except Exception as e:
                response_content = f"❌ Error checking salary: {str(e)}"
        else:
            response_content = (
                "💰 **Salary Check**\n\n"
                "Please specify a salary amount and role. Example:\n"
                "\"Check salary 80000 for Software Engineer\""
            )

    # ---------------------------------------------------------
    # ROUTE: Template Retrieval
    # ---------------------------------------------------------
    elif "template" in query_lower or "retrieve" in query_lower or "get" in query_lower:
        try:
            role = job_context.get("job_title", "Software Engineer")
            result = template_retriever_tool.invoke({"role_type": role})

            if result.get("success") and result.get("templates"):
                response_content = f"### 📋 Retrieved Templates for '{role}'\n\n"
                for i, tmpl in enumerate(result["templates"], 1):
                    preview = tmpl["text"][:300] + "..." if len(tmpl["text"]) > 300 else tmpl["text"]
                    response_content += f"**Template {i}** (source: {tmpl.get('metadata', {}).get('source', 'N/A')}):\n```\n{preview}\n```\n\n"
            else:
                response_content = (
                    f"⚠️ No templates found for '{role}'.\n\n"
                    f"Make sure the ChromaDB knowledge base has been ingested.\n"
                    f"Run: `python scripts/ingest_knowledge.py`"
                )
        except Exception as e:
            response_content = f"❌ Error retrieving templates: {str(e)}"

    # ---------------------------------------------------------
    # ROUTE: Email Drafting
    # ---------------------------------------------------------
    elif "email" in query_lower or "invitation" in query_lower or "interview" in query_lower:
        candidate_name = job_context.get("candidate_name", "[Candidate Name]")
        job_title = job_context.get("job_title", "[Position]")

        response_content = (
            f"### ✉️ Interview Invitation Email\n\n"
            f"**Subject:** Interview Invitation — {job_title} at ATIA Club ESB\n\n"
            f"---\n\n"
            f"Dear {candidate_name},\n\n"
            f"Thank you for your application for the **{job_title}** position at ATIA Club ESB.\n\n"
            f"We were impressed by your profile and would like to invite you for an interview.\n\n"
            f"**Interview Details:**\n"
            f"- 📅 Date: [To be confirmed]\n"
            f"- 🕐 Time: [To be confirmed]\n"
            f"- 📍 Location: [Office / Video Call link]\n\n"
            f"Please confirm your availability by replying to this email.\n\n"
            f"Best regards,\n"
            f"HR Team — ATIA Club ESB"
        )

    # ---------------------------------------------------------
    # Default / Fallback — ATIA-HR branded
    # ---------------------------------------------------------
    else:
        # Detect language from state
        filters = state.get("filters", {})
        lang = filters.get("language", "EN") if filters else "EN"

        if lang == "FR":
            response_content = (
                f"📝 **ATIA-HR — Agent Responsable RH**\n\n"
                f"Je peux vous aider avec :\n"
                f"- **Générer une offre d'emploi** — dites \"rédiger une offre\"\n"
                f"- **Récupérer des modèles** — dites \"chercher un modèle\"\n"
                f"- **Vérifier un salaire** — dites \"vérifier salaire 80000 pour Data Scientist\"\n"
                f"- **Rédiger des emails** — dites \"écrire une invitation d'entretien\"\n\n"
                f"Que souhaitez-vous faire ?"
            )
        else:
            response_content = (
                f"📝 **ATIA-HR — Hiring Manager Agent**\n\n"
                f"I can help you with:\n"
                f"- **Generate a job offer** — say \"draft an offer\"\n"
                f"- **Retrieve templates** — say \"get templates\"\n"
                f"- **Check salary** — say \"check salary 80000 for Data Scientist\"\n"
                f"- **Draft emails** — say \"write interview invitation email\"\n\n"
                f"What would you like to do?"
            )

    return {
        "messages": [AIMessage(content=response_content)],
        "job_context": job_context
    }


def build_manager_graph() -> StateGraph:
    """
    Builds and compiles the Hiring Manager Agent graph.

    Returns:
        A compiled StateGraph ready for execution.
    """
    graph = StateGraph(AgentState)
    graph.add_node("manager_process", agent_node)
    graph.set_entry_point("manager_process")
    graph.add_edge("manager_process", END)
    return graph.compile()


# Expose the compiled graph for import by the supervisor
manager_graph = build_manager_graph()
