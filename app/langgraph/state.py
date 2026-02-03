from typing import List, Optional
from langchain_core.documents import Document
from pydantic import BaseModel

class ResearchState(BaseModel):
    question: str
    task_type: Optional[str] = None
    documents: List[Document] = []
    answer: Optional[str] = None
    sources: List[str] = []
    confidence: float = 0.0
    report_md: Optional[str] = None