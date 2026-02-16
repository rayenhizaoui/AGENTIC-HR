from fastapi import APIRouter
from app.api import chat, search, candidates, hiring

router = APIRouter()

@router.get("/api")
async def api_root():
    return {"message": "API Router Active"}

router.include_router(chat.router, prefix="/chat", tags=["Chat"])
router.include_router(search.router, prefix="/search", tags=["Search"])
router.include_router(candidates.router, prefix="/candidates", tags=["Candidates"])
router.include_router(hiring.router, prefix="/hiring", tags=["Hiring"])
