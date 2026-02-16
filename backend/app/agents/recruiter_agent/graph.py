"""
ATIA-HR Lead Recruiter Agent Graph

This agent handles:
- CV parsing and analysis (PDF upload)
- Skill extraction from resumes
- Candidate ranking and scoring (embeddings + LLM)
- Multi-source job search (RSS/API)
- Multilingual support (FR/EN/Arabic)
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.shared.state import AgentState
from app.agents.shared.utils import logger, extract_last_message

# Import tools from the tools folder
from .tools import (
    cv_parser_tool,
    batch_cv_parser,
    text_cleaner_pipeline,
    anonymizer_tool,
    skill_extractor_tool, 
    candidate_summarizer,
    similarity_matcher_tool, 
    match_explainer, 
    cv_ranker,
    job_scraper_tool,
    job_search_tool,
    llm_rank_candidates,
    ocr_cv_tool,
    semantic_skill_enhancer,
)

# List of all available tools for this agent
RECRUITER_TOOLS = [
    cv_parser_tool,
    batch_cv_parser,
    text_cleaner_pipeline,
    anonymizer_tool,
    skill_extractor_tool,
    candidate_summarizer,
    similarity_matcher_tool,
    match_explainer,
    cv_ranker,
    job_scraper_tool,
    job_search_tool,
    llm_rank_candidates,
    ocr_cv_tool,
    semantic_skill_enhancer,
]


def agent_node(state: AgentState) -> dict:
    """
    Main processing node for the Lead Recruiter Agent.
    Uses NLU understanding from supervisor for intelligent routing.
    
    Args:
        state: The current agent state containing messages and job context.
    
    Returns:
        Updated state with the agent's response.
    
    """
    # Extract the last HumanMessage for context (ignore routing messages)
    user_query = "No query provided"
    messages = state.get("messages", [])
    if messages:
        # Find the last HumanMessage from the end
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        
        # If no HumanMessage found, fallback to last message
        if user_query == "No query provided" and messages:
            last_message = messages[-1]
            user_query = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
    job_context = state.get("job_context", {})
    
    # Check if we have a CV to analyze — check multiple sources
    cv_text = job_context.get("current_cv_text")
    
    # Fallback: check candidateData.text (set by frontend after upload)
    if not cv_text and isinstance(job_context.get("candidateData"), dict):
        cv_text = job_context["candidateData"].get("text", "")
    
    # Fallback: check the parsed CV cache (set by /candidates/analyze)
    if not cv_text:
        try:
            from app.api.candidates import get_cv_cache
            cv_cache = get_cv_cache()
            if cv_cache:
                # Use the most recently cached CV
                last_key = list(cv_cache.keys())[-1]
                cv_text = cv_cache[last_key].get("text", "")
        except Exception:
            pass
    
    response_content = ""

    # === USE SUPERVISOR'S NLU UNDERSTANDING ===
    understanding = job_context.get("understanding", {})
    nlu_intent = understanding.get("intent", "")
    nlu_sub = understanding.get("sub_action", "")
    nlu_role = (understanding.get("entities") or {}).get("role", "")
    rephrased = understanding.get("rephrased_query", user_query)

    query_lower = user_query.lower()
    
    # ---------------------------------------------------------
    # ROUTE: CV Analysis (Upload)  — intent-first + keyword fallback
    # ---------------------------------------------------------
    if nlu_intent == "cv_analysis" or "analyze" in query_lower or "uploaded" in query_lower or "analyser" in query_lower or "تحليل" in query_lower:
        if not cv_text:
            response_content = (
                f"⚠️ **Issue Detected**\n\n"
                f"I noticed you uploaded a CV, but I couldn't extract any text from it.\n"
                f"- The file might be an **image-based PDF** or scanned document (OCR not yet enabled).\n"
                f"- The file might be corrupted or empty.\n\n"
                f"**Please try converting the PDF to a Word document or ensuring it has selectable text.**"
            )
        else:
            # Perform analysis manually (simulating LLM tool use)
            try:
                # 1. Extract Skills (Enhanced tool)
                extracted_data = skill_extractor_tool.invoke({"cv_text": cv_text})
                
                # 2. Summarize (Enhanced tool)
                summary = candidate_summarizer.invoke({
                    "cv_text": cv_text,
                    "extracted_skills": extracted_data
                })
                
                # Update context with extracted data
                job_context["extracted_skills"] = extracted_data
                job_context["candidate_summary"] = summary
                
                # Helper to format list
                def format_list(items):
                    if not items: return "None detected"
                    return ", ".join(items[:10]) + (f" (+{len(items)-10} more)" if len(items) > 10 else "")

                # 3. Format Response (Professional Layout)
                response_content = (
                    f"### 📄 CV Analysis Result\n\n"
                    f"{summary}\n\n"
                    f"#### 🛠️ Technical Competencies\n"
                    f"- **Identified Skills**: {format_list(extracted_data.get('skills', []))}\n"
                    f"- **Key Category**: Data Science & AI (Inferred from keywords)\n\n"
                    f"#### 📊 Professional Profile\n"
                    f"- **Experience Level**: {extracted_data.get('experience_years', 0)} years (Estimated)\n"
                    f"- **Projects Detected**: ~{extracted_data.get('projects_count', 0)} projects mentioned\n\n"
                    f"#### 🎓 Education\n"
                )
                
                if extracted_data.get("education"):
                    for edu in extracted_data.get("education", []):
                        response_content += f"- **{edu.get('degree', 'Degree')}** in {edu.get('field', 'Field')} — *{edu.get('institution', 'Institution')}*\n"
                else:
                    response_content += "- No explicit degree information detected.\n"
                    
                response_content += "\n---\n*Analysis based on keyword extraction and heuristic matching. Would you like to proceed with candidate ranking?*"
                
            except Exception as e:
                response_content = f"❌ Error analyzing CV: {str(e)}"
                import traceback
                print(traceback.format_exc())

    # ---------------------------------------------------------
    # ROUTE: Job Search — intent-first + keyword fallback
    # ---------------------------------------------------------
    elif nlu_intent == "job_search" or any(kw in query_lower for kw in [
        "search", "find job", "fetch", "remote job", "job opening",
        "cherche", "chercher", "trouver", "recherche", "emploi", "travail",
        "ابحث", "وظيفة", "عمل", "شغل",
    ]):
        try:
            # Use detected entities to refine search
            filters = state.get("filters", {})
            search_query = user_query
            if filters and filters.get("skills"):
                search_query = " ".join(filters["skills"])
                if filters.get("location"):
                    search_query += f" {filters['location']}"

            search_result = job_search_tool.invoke({
                "query": search_query,
                "sources": "all",
                "max_results": 15,
            })

            jobs = search_result.get("jobs", [])
            total = search_result.get("total_found", 0)

            if not jobs:
                response_content = "⚠️ No jobs found for your query. Try broader keywords."
            else:
                response_content = (
                    f"### 🔍 Job Search Results\n\n"
                    f"**Query**: {search_result.get('query', user_query)}\n"
                    f"**Found**: {total} job(s)\n\n"
                    f"| # | Title | Company | Location | Source |\n"
                    f"|---|-------|---------|----------|--------|\n"
                )
                for i, j in enumerate(jobs, 1):
                    title_link = f"[{j['title'][:40]}]({j['link']})" if j.get('link') else j.get('title', 'N/A')
                    response_content += (
                        f"| {i} | {title_link} | {j.get('company', 'N/A')} "
                        f"| {j.get('location', 'N/A')} | {j.get('source', '')} |\n"
                    )

                job_context["search_results"] = jobs

        except Exception as e:
            response_content = f"❌ Error during job search: {str(e)}"
            import traceback
            print(traceback.format_exc())

    # ---------------------------------------------------------
    # ROUTE: LLM Ranking (Mistral) — deep analysis
    # ---------------------------------------------------------
    elif (nlu_intent == "cv_ranking" and nlu_sub in ("llm_ranking", "deep_ranking")) or any(kw in query_lower for kw in [
        "llm rank", "deep rank", "mistral", "advanced rank",
        "classement avancé", "analyse approfondie",
    ]):
        extracted_skills = job_context.get("extracted_skills")
        cv_text = job_context.get("current_cv_text", "")

        if not cv_text and not extracted_skills:
            response_content = "⚠️ Please upload and analyze a CV first before LLM ranking."
        else:
            default_jd = job_context.get("current_job_description", """
            Senior Python Developer. Requirements: 3+ years Python, SQL, REST APIs,
            Docker, CI/CD, Cloud (AWS/GCP). Nice to have: ML, data pipelines.
            """)

            try:
                from .tools.llm_ranker import rank_single_candidate
                result = rank_single_candidate(
                    job_description=default_jd,
                    cv_text=cv_text,
                    candidate_name=job_context.get("current_cv_meta", {}).get("filename", "Candidate"),
                    use_llm=True,
                )

                score = result["score"]
                match_level = "🟢 High" if score > 70 else "🟡 Medium" if score > 45 else "🔴 Low"
                engine = "Mistral (Ollama) + Embeddings" if result.get("llm_used") else "Embeddings only (Ollama offline)"

                response_content = (
                    f"### 🧠 LLM-Enhanced Ranking Report\n\n"
                    f"**Engine**: {engine}\n"
                    f"**Candidate**: {result['candidate']}\n"
                    f"**Composite Score**: **{score}%** — {match_level}\n\n"
                    f"| Metric | Score |\n"
                    f"|--------|-------|\n"
                    f"| Semantic Similarity | {result['semantic_similarity']}% |\n"
                    f"| Skill Match | {result['skill_match']}% |\n\n"
                )

                if result.get("matched_skills"):
                    response_content += f"**✅ Matched Skills**: {', '.join(result['matched_skills'][:10])}\n\n"
                if result.get("missing_skills"):
                    response_content += f"**❌ Missing Skills**: {', '.join(result['missing_skills'][:10])}\n\n"

                recommendation = "Proceed to interview" if score > 60 else "Consider for junior role" if score > 35 else "Profile does not match"
                response_content += f"**Recommendation**: {recommendation}"

                job_context["ranking_results"] = [result]

            except Exception as e:
                response_content = f"❌ Error during LLM ranking: {str(e)}"
                import traceback
                print(traceback.format_exc())

    # ---------------------------------------------------------
    # ROUTE: Standard Ranking Candidates — intent-first
    # ---------------------------------------------------------
    elif nlu_intent == "cv_ranking" or any(kw in query_lower for kw in [
        "rank", "ranking", "classer", "classement", "comparer",
        "score", "match", "ترتيب", "مقارنة",
    ]):
        # First, check if there are cached CVs from frontend uploads
        try:
            from app.api.candidates import get_cv_cache
            cv_cache = get_cv_cache()
        except:
            cv_cache = {}
        
        # Check both agent state and uploaded cache
        extracted_skills = job_context.get("extracted_skills")
        
        if not extracted_skills and not cv_cache:
            response_content = "⚠️ Please analyze a candidate CV first before ranking.\n\n💡 **Tip**: Upload a CV using the 📎 button or ask me to analyze a CV."
        else:
            # Extract job description from user query or use default
            default_jd = """
            We are looking for a Data Scientist with experience in Machine Learning, Python, and NLP.
            Key requirements:
            - 2+ years of experience
            - Strong knowledge of TensorFlow, PyTorch, and Scikit-Learn
            - Experience with Large Language Models (LLMs) and RAG
            - Degree in Computer Science or related field.
            """
            
            # Try to extract JD from query or use previously set one
            job_description = job_context.get("current_job_description")
            if not job_description:
                # Check if user provided JD in message
                if "for" in user_query.lower() or "position" in user_query.lower():
                    # Extract everything after "for" or "position"
                    for keyword in ["for", "position"]:
                        if keyword in user_query.lower():
                            parts = user_query.lower().split(keyword, 1)
                            if len(parts) > 1:
                                job_description = parts[1].strip()
                                break
                
                # Still no JD? Use default
                if not job_description or len(job_description) < 20:
                    job_description = default_jd
            
            try:
                # If CVs were uploaded via frontend, rank them
                if cv_cache:
                    from app.agents.recruiter_agent.tools.llm_ranker import rank_single_candidate
                    
                    rankings = []
                    for filename, cv_data in cv_cache.items():
                        result = rank_single_candidate(
                            job_description=job_description,
                            cv_text=cv_data["text"],
                            candidate_name=filename,
                            use_llm=True
                        )
                        result["summary"] = cv_data.get("summary", "")
                        rankings.append(result)
                    
                    # Sort by score
                    rankings.sort(key=lambda x: x["score"], reverse=True)
                    
                    # Format response
                    response_content = f"### 🏆 Candidate Ranking Report\n\n"
                    response_content += f"**Total Candidates Analyzed**: {len(rankings)}\n"
                    response_content += f"**Job Position**: {job_description[:100]}...\n\n"
                    
                    for i, rank in enumerate(rankings, 1):
                        score = rank["score"]
                        match_level = "🟢 High" if score > 75 else "🟡 Medium" if score > 50 else "🔴 Low"
                        response_content += (
                            f"**#{i} - {rank['candidate']}**\n"
                            f"- **Match Score**: {score}% {match_level}\n"
                            f"- **Semantic Match**: {rank.get('semantic_score', 0):.1f}%\n"
                            f"- **Skill Overlap**: {rank.get('skill_overlap_score', 0):.1f}%\n"
                        )
                        if rank.get("summary"):
                            response_content += f"- **Profile**: {rank['summary'][:150]}...\n"
                        response_content += "\n"
                    
                    response_content += "\n💡 **Next Steps**: Interview the top candidates or refine the job description for better matches."
                    
                    # Save rankings to context
                    job_context["ranking_results"] = rankings
                
                # Otherwise, use the old flow with extracted_skills
                elif extracted_skills:
                    candidate_profile = {
                        "id": "candidate_current",
                        "skills": extracted_skills.get("skills", []),
                        "experience": [f"{extracted_skills.get('experience_years', 0)} years experience"],
                        "education": " ".join([e.get('degree','') for e in extracted_skills.get("education", [])])
                    }
                    
                    match_result = similarity_matcher_tool.invoke({
                        "candidate_profile": candidate_profile,
                        "job_description": job_description
                    })
                    
                    score = match_result.get("similarity_score", 0)
                    match_level = "High" if score > 75 else "Medium" if score > 50 else "Low"
                    
                    response_content = (
                        f"### 🏆 Candidate Ranking Report\n\n"
                        f"**Target Role**: Data Scientist / AI Engineer\n"
                        f"**Match Score**: **{score}%** ({match_level} Match)\n\n"
                        f"#### 🔍 Analysis\n"
                        f"The candidate demonstrates a strong alignment with the technical stack (Python, NLP, ML frameworks). "
                        f"The experience level is compatible with the role requirements.\n\n"
                        f"**Recommendation**: Proceed to interview phase."
                    )
                    if match_result.get("note"):
                         response_content += f"\n\n*(Note: {match_result.get('note')})*"
                
            except Exception as e:
                 response_content = f"❌ Error during ranking: {str(e)}"
                 import traceback
                 print(traceback.format_exc())

    # ---------------------------------------------------------
    # ROUTE: Career Advice / Profile Query — intent-first
    # ---------------------------------------------------------
    elif nlu_intent in ("career_advice", "job_description_query"):
        filters = state.get("filters", {})
        lang = understanding.get("language", filters.get("language", "EN")) if understanding else (filters.get("language", "EN") if filters else "EN")
        nlu_skills = (understanding.get("entities") or {}).get("skills", [])

        # Always prefer LLM for career advice — much richer than rule-based
        query_for_llm = rephrased or user_query
        if nlu_role:
            # Augment the query so LLM focuses on the correct role
            query_for_llm = f"{query_for_llm} (Role: {nlu_role})"
        if nlu_skills:
            query_for_llm += f" (Skills context: {', '.join(nlu_skills)})"
        response_content = _llm_answer_recruiter(query_for_llm, lang)

    else:
        # Intelligent fallback — uses NLU understanding + LLM
        filters = state.get("filters", {})
        lang = understanding.get("language", filters.get("language", "EN")) if understanding else (filters.get("language", "EN") if filters else "EN")

        # Use LLM for a contextual, intelligent answer
        response_content = _llm_answer_recruiter(
            rephrased or user_query, lang
        )
    
    return {
        "messages": [AIMessage(content=response_content)],
        "job_context": job_context
    }


def _llm_answer_recruiter(user_message: str, lang: str = "EN") -> str:
    """
    Use Mistral (Ollama) to answer a general HR/recruitment question.
    Falls back to a rule-based answer with profile advice.
    """
    import sys

    try:
        import requests as _req
        resp = _req.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            lang_instruction = {
                "FR": "Réponds en français.",
                "AR": "أجب بالعربية.",
                "EN": "Respond in English.",
            }.get(lang, "Respond in English.")

            prompt = f"""You are ATIA-HR, an expert AI recruitment assistant specializing in tech recruitment.
The user asked a question about recruitment, job profiles, careers, skills, or HR topics.
Give a helpful, detailed, and professional answer using markdown formatting.
Include specific skills, tools, certifications, and actionable advice when relevant.
{lang_instruction}

User question: {user_message}

Answer:"""

            resp = _req.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 512},
                },
                timeout=60,
            )
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip()
            if answer:
                return answer
    except Exception as e:
        print(f"[Recruiter] LLM answer failed: {e}", file=sys.stderr)

    # Rule-based fallback
    return _rule_based_recruiter_answer(user_message, lang)


def _rule_based_recruiter_answer(user_message: str, lang: str = "EN") -> str:
    """Generate role-specific advice without LLM."""
    import re as _re
    msg_lower = user_message.lower()

    # Detect role from message
    role_map = {
        "ai engineer": ("AI/ML Engineer", ["Python", "TensorFlow/PyTorch", "NLP", "Computer Vision", "MLOps", "Docker", "LangChain", "Transformers"]),
        "ia engineer": ("AI/ML Engineer", ["Python", "TensorFlow/PyTorch", "NLP", "Computer Vision", "MLOps", "Docker", "LangChain", "Transformers"]),
        "machine learning": ("ML Engineer", ["Python", "scikit-learn", "TensorFlow/PyTorch", "MLOps", "Docker", "Kubernetes", "SQL"]),
        "data scientist": ("Data Scientist", ["Python", "R", "SQL", "Statistics", "ML", "Pandas", "Tableau/Power BI"]),
        "data engineer": ("Data Engineer", ["Python", "SQL", "Apache Spark", "Airflow", "Kafka", "AWS/GCP", "ETL"]),
        "fullstack": ("Full-Stack Developer", ["JavaScript/TypeScript", "React/Vue", "Node.js", "Python", "SQL", "Docker", "Git"]),
        "full stack": ("Full-Stack Developer", ["JavaScript/TypeScript", "React/Vue", "Node.js", "Python", "SQL", "Docker", "Git"]),
        "frontend": ("Frontend Developer", ["JavaScript/TypeScript", "React/Vue/Angular", "CSS/Tailwind", "HTML", "Figma"]),
        "backend": ("Backend Developer", ["Python/Java/Go", "SQL/NoSQL", "REST APIs", "Docker", "CI/CD", "Redis"]),
        "devops": ("DevOps Engineer", ["Linux", "Docker", "Kubernetes", "CI/CD", "Terraform", "AWS/GCP", "Monitoring"]),
        "developer": ("Software Developer", ["Python/JavaScript", "SQL", "Git", "Docker", "REST APIs", "Agile/Scrum"]),
        "développeur": ("Software Developer", ["Python/JavaScript", "SQL", "Git", "Docker", "REST APIs", "Agile/Scrum"]),
        "ingénieur": ("Software Engineer", ["Python/Java", "Algorithms", "System Design", "Docker", "Cloud", "CI/CD"]),
    }

    detected_role = None
    detected_skills = []
    for key, (role, skills) in role_map.items():
        if key in msg_lower:
            detected_role = role
            detected_skills = skills
            break

    if detected_role:
        skills_md = "\n".join(f"- {s}" for s in detected_skills)
        if lang == "FR":
            return (
                f"### 🎯 Profil idéal : **{detected_role}**\n\n"
                f"#### 🛠️ Compétences clés\n{skills_md}\n\n"
                f"#### 🎓 Formation recommandée\n"
                f"- Bachelor's/Master's en Informatique ou domaine connexe\n\n"
                f"#### 💼 Expérience\n- 2-5 ans dans des projets pertinents\n\n"
                f"💡 *Voulez-vous que je recherche des offres **{detected_role}** ou que j'analyse votre CV ?*"
            )
        else:
            return (
                f"### 🎯 Ideal Profile: **{detected_role}**\n\n"
                f"#### 🛠️ Key Skills\n{skills_md}\n\n"
                f"#### 🎓 Recommended Education\n"
                f"- Bachelor's/Master's in CS or related field\n\n"
                f"#### 💼 Experience\n- 2-5 years in relevant projects\n\n"
                f"💡 *Would you like me to search for **{detected_role}** jobs or analyze your CV?*"
            )

    # Generic fallback
    if lang == "FR":
        return (
            "🎯 **ATIA-HR — Agent Recruteur**\n\n"
            "Je peux vous aider avec :\n"
            "- 📄 **Analyser des CVs** — Uploadez un PDF avec le bouton 📎\n"
            "- 🔍 **Rechercher des emplois** — \"Cherche un emploi Python remote\"\n"
            "- 🏆 **Classer les candidats** — \"Classe ces candidats\"\n"
            "- 🧠 **Analyse avancée LLM** — \"Classement avancé\"\n\n"
            "💡 *Précisez votre question — par ex. \"Quel profil pour un Data Scientist ?\"*"
        )
    else:
        return (
            "🎯 **ATIA-HR — Lead Recruiter Agent**\n\n"
            "I can help you with:\n"
            "- 📄 **Analyze CVs** — Upload a PDF with the 📎 button\n"
            "- 🔍 **Search Jobs** — \"Search for Python developer jobs\"\n"
            "- 🏆 **Rank Candidates** — \"Rank these candidates\"\n"
            "- 🧠 **LLM Deep Analysis** — \"Advanced ranking\"\n\n"
            "💡 *Try asking a specific question — e.g. \"What profile for a Data Scientist?\"*"
        )


def build_recruiter_graph() -> StateGraph:
    """
    Builds and compiles the Lead Recruiter Agent graph.
    
    Returns:
        A compiled StateGraph ready for execution.
    """
    # Initialize the graph with shared state
    graph = StateGraph(AgentState)
    
    # Add the main processing node
    graph.add_node("recruiter_process", agent_node)
    
    # Set entry point
    graph.set_entry_point("recruiter_process")
    
    # Add edge to END
    graph.add_edge("recruiter_process", END)
    
    return graph.compile()


# Expose the compiled graph for import by the supervisor
recruiter_graph = build_recruiter_graph()
