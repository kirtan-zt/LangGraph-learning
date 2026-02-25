from typing import List, Optional
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class ResearchState(BaseModel):
    question: str

    # Routing
    task_type: Optional[str] = None
    force_no_context: bool = False  

    # Retrieved documents
    documents: List[Document] = Field(default_factory=list)

    # Output
    answer: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    confidence: float = 0.0

    # Optional markdown report (for summarize / insights nodes)
    report_md: Optional[str] = None