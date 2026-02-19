from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

# Create a TypeVar to represent "Any Pydantic Model"
T = TypeVar("T")

class StandardResponse(BaseModel, Generic[T]):
    status: int 
    message: str 
    data: Optional[T] = None 
