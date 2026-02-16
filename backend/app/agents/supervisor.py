"""
ATIA-HR Supervisor Agent - Hierarchical Router for Multi-Agent System

This module implements the ATIA-HR Supervisor Pattern that:
1. Detects user intent (job_search, cv_ranking, offer_generation, etc.)
2. Extracts entities (skills, location, level, salary)
3. Routes to specialized agents:
   - Lead Recruiter Agent (CV parsing, skill extraction, ranking, job search)
   - Hiring Manager Agent (RAG templates, job offers, emails, salary checks)
4. Maintains state across conversations
5. Supports multilingual interactions (FR/EN/Arabic)

Uses Mistral (Ollama) for LLM routing with rule-based fallback.
"""

import json
import re
import sys
from typing import Literal, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.agents.shared.state import AgentState
from app.agents.recruiter_agent import recruiter_graph
from app.agents.manager_agent import manager_graph

# HTTP for Ollama
try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# Define the possible routing destinations
TEAM_MEMBERS = ["Lead_Recruiter", "Hiring_Manager"]
FINISH = "FINISH"


class RouteDecision(BaseModel):
    """Structured routing decision from LLM."""
    next: Literal["Lead_Recruiter", "Hiring_Manager", "FINISH"] = Field(
        description="The next agent to route to, or FINISH if the task is complete."
    )
    reasoning: str = Field(
        description="Brief explanation of why this routing decision was made."
    )


# ============================================================
# ATIA-HR SYSTEM PROMPT
# ============================================================
ATIA_HR_SYSTEM_PROMPT = """You are ATIA-HR, an advanced AI-powered Recruitment Assistant built on a multi-agent architecture.
Your core mission is to assist users in job searching, CV matching, hiring processes, and RH tasks.
You support multilingual interactions (FR/EN/Arabic) and focus on Tunisia/remote jobs.

You manage a team of two specialized agents:

1. **Lead_Recruiter**: Handles all CV/candidate and job search tasks:
   - Parsing and analyzing CVs/resumes (PDF upload)
   - Extracting skills from candidates
   - Ranking and scoring candidates against job descriptions
   - Searching for jobs across RSS feeds & APIs (Remote OK, Arbeitnow, etc.)
   - Screening applications
   - Comparing candidates

2. **Hiring_Manager**: Handles communication, offers, and documentation:
   - Writing/generating job offers and descriptions
   - Retrieving templates (RAG via ChromaDB)
   - Drafting emails to candidates (interview invitations, rejections)
   - Salary market checks and comparisons
   - Creating offer letters

Based on the user's request, decide which agent should handle the task.
If the task is a simple greeting, general question, or appears complete, respond with FINISH.

Always respond with a JSON object:
{"next": "Lead_Recruiter" | "Hiring_Manager" | "FINISH", "reasoning": "..."}
"""


# ============================================================
# INTENT DETECTION & ENTITY EXTRACTION
# ============================================================

# Intent keywords — multilingual (FR/EN/Arabic)
INTENT_PATTERNS = {
    "job_search": {
        "keywords": [
            # English
            "search", "find job", "look for job", "fetch job", "remote job",
            "job opening", "vacancy", "position available", "hiring",
            # French
            "cherche", "chercher", "trouver", "recherche", "emploi",
            "travail", "poste", "offre d'emploi", "opportunité",
            # Arabic
            "ابحث", "وظيفة", "عمل", "شغل",
            # common verbs
            "postuler", "candidater", "looking for", "apply",
        ],
        "route": "Lead_Recruiter",
    },
    "cv_analysis": {
        "keywords": [
            "analyze", "parse", "upload", "cv", "resume", "curriculum",
            "analyser", "télécharger", "fichier",
            "سيرة ذاتية", "تحليل",
        ],
        "route": "Lead_Recruiter",
    },
    "cv_ranking": {
        "keywords": [
            "rank", "ranking", "score", "match", "compare", "best candidate",
            "classer", "classement", "comparer", "meilleur candidat",
            "ترتيب", "مقارنة",
        ],
        "route": "Lead_Recruiter",
    },
    "offer_generation": {
        "keywords": [
            "offer", "generate offer", "draft offer", "write offer",
            "job description", "create offer",
            "offre", "rédiger", "générer", "créer offre", "lettre",
            "rédige", "rédigez", "redige", "rediger",
            "عرض عمل", "كتابة",
        ],
        "route": "Hiring_Manager",
    },
    "email_drafting": {
        "keywords": [
            "email", "mail", "invitation", "interview", "draft email",
            "courriel", "entretien", "invitation",
            "بريد", "مقابلة",
        ],
        "route": "Hiring_Manager",
    },
    "salary_check": {
        "keywords": [
            "salary", "compensation", "market", "pay", "wage",
            "salaire", "rémunération", "marché",
            "راتب", "أجر",
        ],
        "route": "Hiring_Manager",
    },
    "template_retrieval": {
        "keywords": [
            "template", "retrieve", "get template", "model",
            "modèle", "gabarit",
            "قالب", "نموذج",
        ],
        "route": "Hiring_Manager",
    },
    "career_advice": {
        "keywords": [
            # English
            "profile", "profil", "best profile", "ideal candidate",
            "skills needed", "requirements", "qualification",
            "career", "career path", "roadmap", "how to become",
            "what skills", "what experience", "tips",
            "advice", "recommend", "suggestion",
            "prepare", "interview prep", "certification",
            "competenc", "aptitude",
            "meilleur", "best", "ideal",
            # French  
            "profil idéal", "compétences requises", "quel profil",
            "quelle compétence", "conseil", "recommandation",
            "comment devenir", "parcours", "formation",
            "préparer", "entretien", "mieux", "better",
            "idéal", "requis", "nécessaire", "faut",
            # Arabic
            "مهارات", "ملف", "نصيحة", "مؤهلات",
        ],
        "route": "Lead_Recruiter",
    },
    "job_description_query": {
        "keywords": [
            # Roles that indicate HR/job questions
            "engineer", "developer", "analyst", "scientist",
            "manager", "designer", "architect", "consultant",
            "technician", "administrator", "coordinator",
            "devops", "fullstack", "frontend", "backend",
            "data engineer", "data scientist", "ml engineer",
            "ai engineer", "cloud engineer", "sre",
            # French roles
            "ingénieur", "développeur", "analyste", "technicien",
            "concepteur", "responsable", "chef de projet",
            # Arabic
            "مهندس", "مطور", "محلل",
        ],
        "route": "Lead_Recruiter",
    },
}


def _is_greeting_only(message: str) -> bool:
    """Check if the message is just a greeting with no substantive question."""
    greetings = [
        "bonjour", "salut", "bonsoir", "hello", "hi", "hey",
        "yo", "coucou", "salam", "مرحبا", "السلام", "ahla",
        "good morning", "good evening", "bonne journée",
    ]
    cleaned = re.sub(r'[^\w\s]', '', message.lower()).strip()
    words = cleaned.split()
    # A greeting is <=3 words that are all greeting words or very common filler
    if len(words) <= 3:
        filler = {"à", "tous", "tout", "le", "la", "monde", "there", "everyone", "all"}
        return all(w in greetings or w in filler for w in words)
    return False


# ============================================================
# PROMPT NORMALIZATION
# ============================================================

# Common typos in tech recruitment context (case-insensitive)
_TYPO_MAP = {
    # Developer variants
    "developper": "developer", "developpeur": "développeur", "developeur": "développeur",
    "devloper": "developer", "devlopper": "developer",
    # Engineer variants
    "engeneer": "engineer", "engineeer": "engineer", "engenieur": "engineer",
    "ingenieur": "ingénieur", "ingeniuer": "ingénieur",
    # Common French typos
    "analiste": "analyste", "annaliste": "analyste",
    "recherhce": "recherche", "recher": "recherche", "cherhe": "cherche",
    "rechrche": "recherche", "recherce": "recherche",
    "kompetences": "compétences", "competences": "compétences",
    "competance": "compétence", "conpetence": "compétence",
    "redige": "rédige", "rediger": "rédiger", "redigez": "rédigez",
    "generer": "générer", "genere": "génère",
    "offre d emploi": "offre d'emploi",
    # Salary/tech
    "salry": "salary", "salaray": "salary", "salaire": "salaire",
    "pyhton": "python", "pytohn": "python", "javasript": "javascript",
    "typesript": "typescript", "kuberntes": "kubernetes",
    "expereince": "experience", "experiance": "experience",
    "candiate": "candidate", "condidat": "candidat",
    "clasement": "classement", "classsement": "classement",
    # Common mixed-language phrases
    "postuler": "postuler",  # keep but ensure recognized
}

# IT abbreviations to expand (unambiguous only)
_ABBREV_MAP = {
    r'\bdev\b': 'developer', r'\bdevs\b': 'developers',
    r'\bjs\b': 'JavaScript', r'\bts\b': 'TypeScript',
    r'\bpy\b': 'Python', r'\bml\b': 'machine learning',
    r'\bdl\b': 'deep learning', r'\bk8s\b': 'kubernetes',
    r'\bsre\b': 'site reliability engineer',
    r'\bjd\b': 'job description', r'\brh\b': 'ressources humaines',
    r'\bqa\b': 'quality assurance', r'\bpm\b': 'project manager',
    r'\bux\b': 'user experience', r'\bui\b': 'user interface',
}


def normalize_prompt(message: str) -> str:
    """
    Normalize user message: fix common typos, expand abbreviations.
    Preserves original language and meaning.
    """
    text = message.strip()
    if not text:
        return text

    # Fix common typos (case-preserving)
    for typo, fix in _TYPO_MAP.items():
        pattern = re.compile(re.escape(typo), re.IGNORECASE)
        text = pattern.sub(fix, text)

    # Expand unambiguous abbreviations
    for pattern, expansion in _ABBREV_MAP.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)

    return text


def detect_intent(message: str) -> dict:
    """
    Detect user intent using keyword matching.
    Returns {"intent": str, "confidence": float, "route": str}.
    """
    msg_lower = message.lower()

    # Fast path: pure greetings
    if _is_greeting_only(message):
        return {"intent": "greeting", "confidence": 1.0, "route": "FINISH"}

    best_intent = None
    best_score = 0
    best_route = "FINISH"

    for intent_name, config in INTENT_PATTERNS.items():
        matches = sum(1 for kw in config["keywords"] if kw in msg_lower)
        if matches > best_score:
            best_score = matches
            best_intent = intent_name
            best_route = config["route"]

    confidence = min(best_score / 3.0, 1.0) if best_score > 0 else 0.0

    # === DISAMBIGUATION PASS ===
    # When multiple intents match with similar scores, use context to pick the right one
    if best_score > 0 and best_intent:
        # Collect all intents with a score close to the best
        competing = {}
        for iname, iconf in INTENT_PATTERNS.items():
            s = sum(1 for kw in iconf["keywords"] if kw in msg_lower)
            if s > 0:
                competing[iname] = (s, iconf["route"])

        # Career advice vs job_description_query—prefer career_advice for "quel profil", "skills", "how to become"
        profile_words = ["profil", "profile", "compétence", "skill", "career", "become",
                         "meilleur", "idéal", "mieux", "better", "best", "advice", "conseil",
                         "roadmap", "parcours", "faut", "nécessaire", "requis"]
        if any(pw in msg_lower for pw in profile_words):
            if "career_advice" in competing:
                best_intent = "career_advice"
                best_route = "Lead_Recruiter"
                best_score = max(best_score, competing["career_advice"][0] + 1)

        # Offer generation vs job_search—prefer offer when "offre", "rédiger", "draft", "write", "générer" present
        offer_words = ["offre", "rédige", "rédiger", "draft", "write", "générer", "create offer",
                       "lettre", "offer letter"]
        if any(ow in msg_lower for ow in offer_words):
            if "offer_generation" in competing:
                best_intent = "offer_generation"
                best_route = "Hiring_Manager"
                best_score = max(best_score, competing["offer_generation"][0] + 1)

        # Job search—boost when "cherche", "trouver", "search", "find", "postuler" present
        search_words = ["cherche", "trouver", "search", "find", "postuler", "looking",
                        "ابحث", "recherche un"]
        if any(sw in msg_lower for sw in search_words):
            if "job_search" in competing:
                best_intent = "job_search"
                best_route = "Lead_Recruiter"
                best_score = max(best_score, competing["job_search"][0] + 1)

        confidence = min(best_score / 3.0, 1.0)

    # If no keyword matched but the message is a real question (>4 words),
    # treat it as a general HR question routed to Lead_Recruiter
    if best_score == 0 and len(msg_lower.split()) >= 4:
        # Check for question indicators
        question_indicators = [
            "?", "quel", "quelle", "comment", "pourquoi", "combien",
            "what", "how", "why", "which", "where", "when", "who",
            "est-ce", "c'est quoi", "donne", "explain", "tell me",
            "je veux", "i want", "i need", "j'ai besoin",
            "liste", "list", "describe", "décris",
            "كيف", "ماذا", "لماذا", "ما هو", "ما هي",
        ]
        is_question = any(ind in msg_lower for ind in question_indicators)
        if is_question:
            return {
                "intent": "hr_question",
                "confidence": 0.5,
                "route": "Lead_Recruiter",
            }
        # Even without question markers, if the message is substantive (>5 words),
        # route to agent instead of dead-ending at FINISH
        if len(msg_lower.split()) >= 5:
            return {
                "intent": "hr_question",
                "confidence": 0.3,
                "route": "Lead_Recruiter",
            }

    return {
        "intent": best_intent or "general",
        "confidence": round(confidence, 2),
        "route": best_route if best_score > 0 else "FINISH",
    }


def extract_entities(message: str) -> dict:
    """
    Extract key entities from user message:
    - skills, locations, experience level, salary, language
    """
    msg_lower = message.lower()
    entities = {}

    # Skills extraction
    tech_skills = [
        "python", "java", "javascript", "typescript", "react", "angular", "vue",
        "node", "nodejs", "django", "flask", "fastapi", "spring", "docker",
        "kubernetes", "aws", "azure", "gcp", "sql", "nosql", "mongodb",
        "postgresql", "redis", "tensorflow", "pytorch", "scikit-learn",
        "machine learning", "deep learning", "nlp", "llm", "rag",
        "langchain", "langgraph", "data science", "data engineer",
        "devops", "ci/cd", "git", "linux", "agile", "scrum",
        "figma", "ui/ux", "html", "css", "tailwind",
        "c++", "c#", ".net", "rust", "go", "kotlin", "swift",
    ]
    found_skills = [s for s in tech_skills if s in msg_lower]
    if found_skills:
        entities["skills"] = found_skills

    # Location extraction
    locations = [
        "remote", "tunis", "tunisia", "tunisie", "paris", "france",
        "berlin", "germany", "london", "uk", "usa", "canada",
        "dubai", "uae", "maroc", "morocco", "algérie", "algeria",
        "sfax", "sousse", "monastir", "bizerte", "nabeul",
        "à distance", "télétravail", "hybride", "hybrid", "on-site",
    ]
    found_locations = [loc for loc in locations if loc in msg_lower]
    if found_locations:
        entities["location"] = found_locations[0]

    # Experience level
    levels = {
        "junior": ["junior", "débutant", "entry", "stage", "intern", "مبتدئ"],
        "mid": ["mid", "intermédiaire", "middle", "2-5 ans", "متوسط"],
        "senior": ["senior", "expert", "lead", "principal", "خبير", "5+ ans"],
    }
    for level, keywords in levels.items():
        if any(kw in msg_lower for kw in keywords):
            entities["level"] = level
            break

    # Salary extraction
    salary_match = re.search(r'(\d[\d,\.]*)\s*(?:tnd|dt|eur|usd|\$|€|دينار)', msg_lower)
    if not salary_match:
        salary_match = re.search(r'(?:salary|salaire|راتب)\s*(?:of|de|:)?\s*(\d[\d,\.]*)', msg_lower)
    if salary_match:
        try:
            entities["salary"] = float(salary_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Language detection
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', message))
    french_indicators = ["je", "tu", "nous", "vous", "le", "la", "les",
                         "un", "une", "des", "est", "sont", "pour",
                         "dans", "avec", "sur", "bonjour", "salut",
                         "merci", "oui", "non", "cherche", "emploi",
                         "travail", "offre", "candidat", "poste",
                         "salaire", "entretien", "veuillez", "s'il"]
    french_words = sum(1 for w in french_indicators if w in msg_lower)
    if arabic_chars > 3:
        entities["language"] = "AR"
    elif french_words >= 1:
        entities["language"] = "FR"
    else:
        entities["language"] = "EN"

    return entities


# ============================================================
# LLM ROUTING (Ollama/Mistral)
# ============================================================

def _ollama_available() -> bool:
    """Quick health check on Ollama server."""
    if not _HAS_REQUESTS:
        return False
    try:
        resp = _requests.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _llm_answer_general(user_message: str, lang: str = "EN") -> str:
    """
    Use Mistral (Ollama) to answer a general HR/recruitment question.
    Returns empty string if Ollama is unavailable.
    """
    if not _ollama_available():
        # Rule-based fallback for common question patterns
        return _rule_based_answer(user_message, lang)

    lang_instruction = {
        "FR": "Réponds en français.",
        "AR": "أجب بالعربية.",
        "EN": "Respond in English.",
    }.get(lang, "Respond in English.")

    prompt = f"""You are ATIA-HR, an expert AI recruitment assistant.
The user asked a question about recruitment, careers, or HR.
Answer helpfully, concisely, and professionally. Use markdown formatting.
Focus on practical, actionable information.
{lang_instruction}

User question: {user_message}

Answer:"""

    try:
        resp = _requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
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
        print(f"[ATIA-HR] LLM general answer failed: {e}", file=sys.stderr)

    return _rule_based_answer(user_message, lang)


def _rule_based_answer(user_message: str, lang: str = "EN") -> str:
    """
    Generate a helpful answer without LLM for common HR questions.
    Uses keyword matching to provide structured advice.
    """
    msg_lower = user_message.lower()

    # Detect job role mentioned
    role_keywords = {
        "ai engineer": "AI/ML Engineer",
        "ia engineer": "AI/ML Engineer",
        "machine learning": "ML Engineer",
        "data scientist": "Data Scientist",
        "data engineer": "Data Engineer",
        "full stack": "Full-Stack Developer",
        "fullstack": "Full-Stack Developer",
        "frontend": "Frontend Developer",
        "backend": "Backend Developer",
        "devops": "DevOps Engineer",
        "cloud": "Cloud Engineer",
        "cybersecurity": "Cybersecurity Analyst",
        "web developer": "Web Developer",
        "mobile": "Mobile Developer",
        "ingénieur ia": "AI/ML Engineer",
        "développeur": "Software Developer",
    }

    detected_role = None
    for key, role in role_keywords.items():
        if key in msg_lower:
            detected_role = role
            break

    # Role-specific profile advice
    role_profiles = {
        "AI/ML Engineer": {
            "skills": ["Python", "TensorFlow/PyTorch", "NLP", "Computer Vision",
                       "MLOps", "Docker", "AWS/GCP", "LangChain", "Transformers"],
            "education": "Master's or PhD in CS, AI, or related field",
            "experience": "2-5 years in ML/AI projects",
            "certifications": ["AWS ML Specialty", "Google ML Engineer", "Deep Learning Specialization (Coursera)"],
        },
        "ML Engineer": {
            "skills": ["Python", "scikit-learn", "TensorFlow/PyTorch", "MLOps",
                       "Docker", "Kubernetes", "SQL", "Feature Engineering"],
            "education": "Bachelor's/Master's in CS or Mathematics",
            "experience": "2-4 years in data/ML pipelines",
            "certifications": ["AWS ML Specialty", "MLflow Certification"],
        },
        "Data Scientist": {
            "skills": ["Python", "R", "SQL", "Statistics", "Machine Learning",
                       "Pandas", "Visualization (Tableau/Power BI)", "A/B Testing"],
            "education": "Master's in Data Science, Statistics, or CS",
            "experience": "1-3 years in data analysis/ML",
            "certifications": ["Google Data Analytics", "IBM Data Science"],
        },
        "Data Engineer": {
            "skills": ["Python", "SQL", "Apache Spark", "Airflow", "Kafka",
                       "AWS/GCP/Azure", "ETL", "Data Warehousing", "dbt"],
            "education": "Bachelor's in CS or Engineering",
            "experience": "2-4 years in data pipelines",
            "certifications": ["AWS Data Analytics", "GCP Data Engineer"],
        },
        "Full-Stack Developer": {
            "skills": ["JavaScript/TypeScript", "React/Vue/Angular", "Node.js",
                       "Python/Java", "SQL", "REST APIs", "Docker", "Git"],
            "education": "Bachelor's in CS or self-taught with portfolio",
            "experience": "1-3 years full-stack projects",
            "certifications": ["AWS Cloud Practitioner", "Meta Full-Stack Certificate"],
        },
        "DevOps Engineer": {
            "skills": ["Linux", "Docker", "Kubernetes", "CI/CD (Jenkins/GitHub Actions)",
                       "Terraform", "AWS/GCP/Azure", "Monitoring (Grafana/Prometheus)", "Bash"],
            "education": "Bachelor's in CS or Systems Engineering",
            "experience": "2-4 years in infrastructure/DevOps",
            "certifications": ["AWS DevOps Professional", "CKA (Kubernetes)", "Terraform Associate"],
        },
    }

    if detected_role and detected_role in role_profiles:
        p = role_profiles[detected_role]
        if lang == "FR":
            answer = (
                f"### 🎯 Profil idéal : **{detected_role}**\\n\\n"
                f"#### 🛠️ Compétences clés\\n"
                f"{chr(10).join('- ' + s for s in p['skills'])}\\n\\n"
                f"#### 🎓 Formation\\n- {p['education']}\\n\\n"
                f"#### 💼 Expérience\\n- {p['experience']}\\n\\n"
                f"#### 📜 Certifications recommandées\\n"
                f"{chr(10).join('- ' + c for c in p['certifications'])}\\n\\n"
                f"💡 *Voulez-vous que je recherche des offres **{detected_role}** ou que j'analyse votre CV ?*"
            )
        else:
            answer = (
                f"### 🎯 Ideal Profile: **{detected_role}**\\n\\n"
                f"#### 🛠️ Key Skills\\n"
                f"{chr(10).join('- ' + s for s in p['skills'])}\\n\\n"
                f"#### 🎓 Education\\n- {p['education']}\\n\\n"
                f"#### 💼 Experience\\n- {p['experience']}\\n\\n"
                f"#### 📜 Recommended Certifications\\n"
                f"{chr(10).join('- ' + c for c in p['certifications'])}\\n\\n"
                f"💡 *Would you like me to search for **{detected_role}** jobs or analyze your CV?*"
            )
        return answer

    # Generic HR question fallback
    if lang == "FR":
        return (
            "🎯 **ATIA-HR** — Je peux vous aider !\\n\\n"
            "Pourriez-vous préciser votre question ? Par exemple :\\n"
            "- *\"Quel est le profil idéal pour un Data Scientist ?\"*\\n"
            "- *\"Cherche un emploi Python remote\"*\\n"
            "- *\"Quelles compétences pour un DevOps Engineer ?\"*\\n"
            "- Ou uploadez un CV avec le bouton 📎\\n\\n"
            "Je suis spécialisé en recrutement tech et je connais le marché tunisien et international."
        )
    else:
        return (
            "🎯 **ATIA-HR** — I can help!\\n\\n"
            "Could you clarify your question? For example:\\n"
            "- *\"What's the ideal profile for a Data Scientist?\"*\\n"
            "- *\"Search for Python remote jobs\"*\\n"
            "- *\"What skills does a DevOps Engineer need?\"*\\n"
            "- Or upload a CV with the 📎 button\\n\\n"
            "I specialize in tech recruitment across Tunisia and international markets."
        )


def llm_route(user_message: str) -> str:
    """
    Use Mistral (Ollama) to decide routing based on user intent.
    Falls back to rule-based routing if Ollama is unavailable.
    """
    if not _ollama_available():
        print("[ATIA-HR Supervisor] Ollama unavailable — using rule-based routing", file=sys.stderr)
        return rule_based_route(user_message)

    prompt = f"""{ATIA_HR_SYSTEM_PROMPT}

USER MESSAGE: {user_message}

Respond ONLY with valid JSON. No explanation, no markdown fences."""

    try:
        resp = _requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 128},
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        # Parse JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > 0:
            data = json.loads(raw[start:end])
            next_agent = data.get("next", "FINISH")
            reasoning = data.get("reasoning", "")
            print(f"[ATIA-HR LLM] → {next_agent} | {reasoning}", file=sys.stderr)
            if next_agent in ("Lead_Recruiter", "Hiring_Manager", "FINISH"):
                return next_agent

    except Exception as e:
        print(f"[ATIA-HR] LLM routing failed: {e} — falling back to rules", file=sys.stderr)

    return rule_based_route(user_message)


def rule_based_route(user_message: str) -> str:
    """
    Enhanced rule-based routing with multilingual support (FR/EN/Arabic).
    Fallback when Ollama is unavailable.
    """
    intent_result = detect_intent(user_message)

    if intent_result["confidence"] > 0.0:
        print(f"[ATIA-HR Rules] Intent: {intent_result['intent']} "
              f"(confidence: {intent_result['confidence']}) → {intent_result['route']}",
              file=sys.stderr)
        return intent_result["route"]

    # Even at 0% confidence, if the message is long enough to be a real question,
    # route to Lead_Recruiter rather than dead-ending at FINISH
    if len(user_message.split()) >= 4 and not _is_greeting_only(user_message):
        print(f"[ATIA-HR Rules] No keywords matched but message is substantive — routing to Lead_Recruiter",
              file=sys.stderr)
        return "Lead_Recruiter"

    return "FINISH"


# ============================================================
# UNIFIED LLM UNDERSTANDING
# ============================================================

_NLU_PROMPT = """You are the NLU engine for ATIA-HR, an AI recruitment platform.
Analyze the user's message comprehensively.
The user may write in French, English, Arabic, or a mix (common in Tunisia).
They may use abbreviations (dev, ML, AI, JS, k8s…), typos, or informal language.

Return a JSON object with:
{
  "intent": one of ["greeting", "job_search", "cv_analysis", "cv_ranking", "career_advice", "offer_generation", "email_drafting", "salary_check", "template_retrieval", "general_hr"],
  "sub_action": more specific action (e.g. "profile_query", "skills_query", "roadmap_query", "search_jobs", "rank_candidates", "llm_ranking", "write_offer", "interview_invitation", "market_rate", null if unclear),
  "entities": {
    "role": "detected job role if any (e.g. AI Engineer, Data Scientist, DevOps…)" or null,
    "skills": ["skill1", "skill2"] or [],
    "location": "location if mentioned" or null,
    "level": "junior" or "mid" or "senior" or null,
    "salary": "salary if mentioned" or null
  },
  "language": "FR" or "EN" or "AR",
  "rephrased_query": "the user's message rephrased clearly, fixing typos and expanding abbreviations, in the same language",
  "confidence": 0.0 to 1.0,
  "route": "Lead_Recruiter" or "Hiring_Manager" or "FINISH",
  "reasoning": "brief explanation of your analysis"
}

Routing rules:
- Lead_Recruiter: CV analysis, job search, ranking, career advice, skills questions, profile questions
- Hiring_Manager: Job offers, emails, salary checks, templates, offer letters
- FINISH: ONLY for pure greetings with absolutely NO question ("bonjour", "hi", "hello")

Intent guide:
- career_advice: asking about ideal profiles, required skills, career paths, how to become X, best profile for a role
- job_search: actively looking for job listings or openings
- cv_analysis: analyzing/parsing an uploaded CV document
- cv_ranking: ranking, comparing, or scoring candidates
- general_hr: any other HR/recruitment question

IMPORTANT: If in doubt, NEVER route to FINISH. Route to Lead_Recruiter instead.
IMPORTANT: "career_advice" includes questions like "quel profil pour X", "quelles compétences", "what skills for", "how to become", "c'est quoi le meilleur profil".

Respond ONLY with valid JSON. No markdown, no explanation.

User message: """


def understand_user_message(message: str) -> dict:
    """
    Comprehensive LLM-powered understanding of user message.
    Extracts intent, entities, language, sub-action, and routing in ONE call.
    Falls back to rule-based detection if Ollama is unavailable.
    """
    # Normalize first
    normalized = normalize_prompt(message)

    # Always compute rule-based results as baseline / fallback
    rule_intent = detect_intent(normalized)
    rule_entities = extract_entities(normalized)
    rule_route = rule_based_route(normalized)

    # === Try LLM understanding ===
    if _ollama_available():
        try:
            resp = _requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": _NLU_PROMPT + normalized + '"',
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 500},
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")

            # Parse JSON from LLM response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])

                # Validate fields
                valid_intents = {
                    "greeting", "job_search", "cv_analysis", "cv_ranking",
                    "career_advice", "offer_generation", "email_drafting",
                    "salary_check", "template_retrieval", "general_hr",
                }
                valid_routes = {"Lead_Recruiter", "Hiring_Manager", "FINISH"}

                intent = data.get("intent", "general_hr")
                if intent not in valid_intents:
                    intent = "general_hr"

                route = data.get("route", "Lead_Recruiter")
                if route not in valid_routes:
                    route = "Lead_Recruiter"

                # Safety net: never FINISH for non-greetings
                if route == "FINISH" and intent != "greeting":
                    route = "Lead_Recruiter"
                if route == "FINISH" and not _is_greeting_only(message):
                    route = "Lead_Recruiter"

                # Inverse safety net: force FINISH for pure greetings
                if intent == "greeting" and _is_greeting_only(message):
                    route = "FINISH"

                entities = data.get("entities", {})
                if not isinstance(entities, dict):
                    entities = {}

                # Merge rule-based entities for completeness
                if not entities.get("skills") and rule_entities.get("skills"):
                    entities["skills"] = rule_entities["skills"]
                if not entities.get("location") and rule_entities.get("location"):
                    entities["location"] = rule_entities["location"]
                if not entities.get("level") and rule_entities.get("level"):
                    entities["level"] = rule_entities["level"]

                # Language detection: trust LLM but validate
                lang = data.get("language", rule_entities.get("language", "EN"))
                if lang not in ("FR", "EN", "AR"):
                    lang = rule_entities.get("language", "EN")
                entities["language"] = lang

                confidence = data.get("confidence", 0.8)
                try:
                    confidence = min(float(confidence), 1.0)
                except (TypeError, ValueError):
                    confidence = 0.8

                print(
                    f"[ATIA-HR NLU] LLM understanding: intent={intent}, "
                    f"route={route}, role={entities.get('role')}, "
                    f"lang={lang}, conf={confidence:.0%}",
                    file=sys.stderr,
                )

                return {
                    "intent": intent,
                    "sub_action": data.get("sub_action"),
                    "entities": entities,
                    "language": lang,
                    "confidence": confidence,
                    "route": route,
                    "rephrased_query": data.get("rephrased_query", normalized),
                    "reasoning": data.get("reasoning", ""),
                    "llm_used": True,
                }

        except Exception as e:
            print(f"[ATIA-HR NLU] LLM understanding failed: {e}", file=sys.stderr)

    # === Fallback: enhanced rule-based ===
    print("[ATIA-HR NLU] Using rule-based fallback", file=sys.stderr)
    route = rule_route
    if route == "FINISH" and not _is_greeting_only(message) and len(message.split()) >= 3:
        route = "Lead_Recruiter"

    return {
        "intent": rule_intent["intent"],
        "sub_action": None,
        "entities": rule_entities,
        "language": rule_entities.get("language", "EN"),
        "confidence": rule_intent["confidence"],
        "route": route,
        "rephrased_query": normalized,
        "reasoning": f"Rule-based: {rule_intent['intent']} ({rule_intent['confidence']:.0%})",
        "llm_used": False,
    }


# ============================================================
# GRAPH NODES
# ============================================================

def supervisor_node(state: AgentState) -> dict:
    """
    ATIA-HR Supervisor node — uses LLM-powered NLU for intelligent routing.
    Understands intent, entities, language, and sub-action in one pass.
    """
    messages = state.get("messages", [])

    if not messages:
        return {
            "next": "FINISH",
            "messages": [AIMessage(content="No input received. Please provide a task.")],
        }

    # Get the last user message
    last_message = messages[-1]
    user_input = last_message.content if hasattr(last_message, "content") else str(last_message)

    # === COMPREHENSIVE LLM UNDERSTANDING ===
    understanding = understand_user_message(user_input)

    # Build detailed reasoning log
    reasoning_steps: list[str] = []
    reasoning_steps.append(
        f'📥 \"{user_input[:100]}{"..." if len(user_input) > 100 else ""}\"'
    )

    rephrased = understanding.get("rephrased_query", "")
    if rephrased and rephrased.lower().strip() != user_input.lower().strip():
        reasoning_steps.append(f'📝 Understood as: "{rephrased[:100]}"')

    engine_tag = "🤖 Mistral LLM" if understanding.get("llm_used") else "📋 Rule-based"
    reasoning_steps.append(
        f"🧠 Intent: **{understanding['intent']}** "
        f"(confidence: {understanding['confidence']:.0%}) [{engine_tag}]"
    )
    if understanding.get("sub_action"):
        reasoning_steps.append(f"🎯 Sub-action: {understanding['sub_action']}")

    entities = understanding["entities"]
    if entities.get("role"):
        reasoning_steps.append(f"👤 Role: {entities['role']}")
    if entities.get("skills") and isinstance(entities["skills"], list):
        reasoning_steps.append(f"🔧 Skills: {', '.join(str(s) for s in entities['skills'][:6])}")
    if entities.get("location"):
        reasoning_steps.append(f"📍 Location: {entities['location']}")
    if entities.get("level"):
        reasoning_steps.append(f"📊 Level: {entities['level']}")
    if entities.get("salary"):
        reasoning_steps.append(f"💰 Salary: {entities['salary']}")

    lang = understanding["language"]
    reasoning_steps.append(f"🌍 Language: {lang}")

    route = understanding["route"]
    reasoning_steps.append(f"🔀 Routing: **{route}**")
    if understanding.get("reasoning"):
        reasoning_steps.append(f"💡 {understanding['reasoning']}")
    reasoning_steps.append(f"🚀 Dispatching to **{route}**...")

    # Update job_context with understanding for downstream agents
    job_context = state.get("job_context", {})
    job_context["understanding"] = understanding  # full NLU result

    if entities.get("skills") and isinstance(entities["skills"], list):
        job_context["detected_skills"] = entities["skills"]
    if entities.get("location"):
        job_context["detected_location"] = entities["location"]
    if entities.get("level"):
        job_context["detected_level"] = entities["level"]
    if entities.get("salary"):
        job_context["salary_numeric"] = entities["salary"]
        job_context["salary"] = str(entities["salary"])
    if entities.get("role"):
        job_context["detected_role"] = entities["role"]

    routing_message = AIMessage(
        content=f"🔀 **ATIA-HR Supervisor**: Routing to `{route}` "
                f"(intent: {understanding['intent']}, lang: {lang})"
    )

    # Build filters from entities
    filters = {k: v for k, v in entities.items() if v}
    if "language" not in filters:
        filters["language"] = lang

    return {
        "next": route,
        "messages": [routing_message],
        "job_context": job_context,
        "current_task": understanding["intent"],
        "filters": filters,
        "user_preferences": {"language": lang, "focus": "Tunisie/remote"},
        "reasoning_log": reasoning_steps,
    }


def recruiter_node(state: AgentState) -> dict:
    """Wrapper node that invokes the Lead Recruiter sub-graph."""
    result = recruiter_graph.invoke(state)
    return {
        "messages": result.get("messages", []),
        "job_context": result.get("job_context", state.get("job_context", {})),
    }


def manager_node(state: AgentState) -> dict:
    """Wrapper node that invokes the Hiring Manager sub-graph."""
    result = manager_graph.invoke(state)
    return {
        "messages": result.get("messages", []),
        "job_context": result.get("job_context", state.get("job_context", {})),
    }


def finish_node(state: AgentState) -> dict:
    """
    ATIA-HR terminal node — handles greetings and general HR questions.
    Uses LLM for substantive questions, canned welcome for greetings.
    """
    messages = state.get("messages", [])
    filters = state.get("filters", {})
    lang = filters.get("language", "EN") if filters else "EN"

    # Get original user message
    user_input = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break

    # If the user asked a real question (not just greeting), try to answer with LLM
    if user_input and not _is_greeting_only(user_input) and len(user_input.split()) >= 3:
        llm_answer = _llm_answer_general(user_input, lang)
        if llm_answer:
            return {"messages": [AIMessage(content=llm_answer)]}

    # Check if we already have a substantive response
    if len(messages) > 2:
        return {}

    # Multilingual welcome for greetings / empty input
    if lang == "FR":
        welcome = (
            "👋 Bonjour ! Je suis **ATIA-HR**, votre assistant intelligent de recrutement.\n\n"
            "Je peux vous aider avec :\n"
            "- 📄 **Analyser des CVs** — Uploadez un PDF avec le bouton 📎\n"
            "- 🔍 **Rechercher des emplois** — Tunisie, remote, international\n"
            "- 🏆 **Classer les candidats** — Scoring par IA (embeddings + LLM)\n"
            "- 📝 **Générer des offres** — Lettres d'embauche personnalisées\n"
            "- 💰 **Vérifier les salaires** — Comparaison au marché\n"
            "- ✉️ **Rédiger des emails** — Invitations d'entretien\n\n"
            "💡 *Essayez : \"Cherche un emploi Python remote\" ou uploadez un CV !*"
        )
    elif lang == "AR":
        welcome = (
            "👋 مرحبا! أنا **ATIA-HR**، مساعدك الذكي للتوظيف.\n\n"
            "يمكنني مساعدتك في:\n"
            "- 📄 **تحليل السيرة الذاتية** — حمّل ملف PDF بالزر 📎\n"
            "- 🔍 **البحث عن وظائف** — تونس، عن بعد، دولي\n"
            "- 🏆 **ترتيب المترشحين** — تقييم بالذكاء الاصطناعي\n"
            "- 📝 **إنشاء عروض عمل** — رسائل توظيف مخصصة\n"
            "- 💰 **فحص الرواتب** — مقارنة بالسوق\n\n"
            "💡 *جرّب: \"ابحث عن وظيفة Python عن بعد\" أو حمّل سيرة ذاتية!*"
        )
    else:
        welcome = (
            "👋 Hello! I'm **ATIA-HR**, your intelligent recruitment assistant.\n\n"
            "I can help you with:\n"
            "- 📄 **Analyze CVs** — Upload a PDF with the 📎 button\n"
            "- 🔍 **Search jobs** — Tunisia, remote, international\n"
            "- 🏆 **Rank candidates** — AI-powered scoring (embeddings + LLM)\n"
            "- 📝 **Generate offers** — Personalized offer letters\n"
            "- 💰 **Check salaries** — Market comparison\n"
            "- ✉️ **Draft emails** — Interview invitations\n\n"
            "💡 *Try: \"Search for Python remote jobs\" or upload a CV!*"
        )

    return {
        "messages": [AIMessage(content=welcome)]
    }


def route_to_agent(state: AgentState) -> str:
    """Conditional edge function that returns the next node based on state."""
    next_step = state.get("next", "FINISH")

    if next_step == "Lead_Recruiter":
        return "recruiter"
    elif next_step == "Hiring_Manager":
        return "manager"
    else:
        return "finish"


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_supervisor_graph() -> StateGraph:
    """Builds and compiles the ATIA-HR supervisor graph."""
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("recruiter", recruiter_node)
    graph.add_node("manager", manager_node)
    graph.add_node("finish", finish_node)

    # Set the entry point
    graph.set_entry_point("supervisor")

    # Add conditional edges from supervisor
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "recruiter": "recruiter",
            "manager": "manager",
            "finish": "finish",
        },
    )

    # All agents route back to END after processing
    graph.add_edge("recruiter", END)
    graph.add_edge("manager", END)
    graph.add_edge("finish", END)

    return graph.compile()


# Export the compiled supervisor graph
supervisor_graph = build_supervisor_graph()


# Convenience function for running the graph
def run_supervisor(user_input: str, job_context: dict = None) -> dict:
    """
    Convenience function to run the ATIA-HR supervisor graph.

    Args:
        user_input: The user's message/request.
        job_context: Optional shared context dictionary.

    Returns:
        The final state after graph execution.
    """
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "next": "",
        "job_context": job_context or {},
        "current_task": None,
        "filters": None,
        "user_preferences": None,
        "reasoning_log": [],
    }

    return supervisor_graph.invoke(initial_state)
