from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class RAGResult(BaseModel):
    """Stores structured LLM response for a given question"""
    answer: str
    sources: List[str]
    confidence: float

# Prompt template with context ingestion
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a company knowledge assistant.\n"
            "Answer the user's question using document context AND conversation history.\n"
            "You may receive context from MULTIPLE documents.\n\n"
            "If a user asks random question that is not in context, Do not hallucinate. Respond with: Enter a question from the uploaded document.\n"
            "Rules:\n"
            "- Do NOT use prior knowledge.\n"
            "- If multiple documents are present, you MUST cover EACH document.\n"
            "- Clearly separate the answer per document.\n"
            "- If the answer cannot be found, say:\n"
            "-'No matching information identified. Please upload relevant documents.'\n"
            "- Cite sources using file name AND page number.\n"
            "- If multiple sources are used, list them all.\n"
            "- If information is missing for any document, explicitly say so.\n"
            "- Be concise and factual.\n"
        ),
        (
            "human",
            "Conversation history:\n{history}\n\n"
            "Document context:\n{context}\n\n"
            "Question:\n{question}",
        ),
    ]
)


class AIService:
    """Business logic for AI response"""
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def generate_rag_answer(
        self,
        question: str,
        documents: List[Document],
        history: str,
    ) -> RAGResult:
        """Generates answers from LLM for a given question 

        Args:
            question (str): Question from the document uploaded
            documents (List[Document]): Reference document objects for generating answers.

        Returns:
            RAGResult: Response model to store feedback
        """
        context_blocks = []
        sources = []

        for doc in documents:
            file_name = doc.metadata.get("file_name", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            page = doc.metadata.get("page_number", "?")

            context_blocks.append(
                f"[Source: {file_name}, Page {page}, Chunk {chunk_idx}]\n"
                f"{doc.page_content}"
            )

            sources.append(f"{file_name} – Page {page}")

        context = "\n\n".join(context_blocks)

        prompt = RAG_PROMPT.invoke(
            {
                "context": context,
                "question": question,
                "history": history,
            }
        )

        response = await self.llm.ainvoke(prompt)
        answer = response.content.strip()

        if answer.startswith("No matching information identified"):
            return RAGResult(
                answer=answer,
                sources=[],
                confidence=0.0,
            )

        confidence = min(1.0, 0.6 + 0.1 * len(documents))

        return RAGResult(
            answer=answer,
            sources=list(set(sources)),
            confidence=confidence,
        )

    async def generate_chat_title(self, question: str, answer: str) -> str:
        prompt = f"""
        Generate a short 3–6 word title summarizing this conversation.

        Question: {question}
        Answer: {answer}

        Title:
        """

        response = await self.llm.ainvoke(prompt)
        return response.content.strip().strip('"')