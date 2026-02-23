from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID
from datetime import datetime

class DocumentRead(BaseModel):
    """
    Schema representing the public metadata of an uploaded document.
    """
    id: UUID
    name: str
    upload_date: datetime

    model_config = ConfigDict(from_attributes=True)
