from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime

class DocumentRead(BaseModel):
    """
    Schema representing the public metadata of an uploaded document.
    """
    id: UUID
    name: str
    upload_date: datetime

    class Config:
        from_attributes = True
