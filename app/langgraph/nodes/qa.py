from app.langgraph.state import ResearchState


def make_qa_node(llm):
    async def qa_node(state: ResearchState):

        # Hard stop if router marked irrelevant
        if state.force_no_context:
            state.answer = "No context found."
            state.sources = []
            state.confidence = 0.0
            return state

        # If retrieval returned nothing
        if not state.documents:
            state.answer = "No context found."
            state.sources = []
            state.confidence = 0.0
            return state

        # Build context safely
        context = "\n\n".join(
            f"(Page {doc.metadata.get('page_number')})\n{doc.page_content}"
            for doc in state.documents
            if doc.page_content
        )

        prompt = f"""
You are a STRICT DOCUMENT-BASED QUESTION ANSWERING system.

You MUST follow these rules:

1. Use ONLY the provided context.
2. DO NOT use external knowledge.
3. DO NOT infer beyond what is written.
4. DO NOT fabricate missing facts.
5. If the answer is not explicitly present in the context,
   return EXACTLY:
   "No context found."
6. Every factual statement MUST include citation in format:
   (Page X)
7. DO NOT add explanations outside the answer.

--------------------------------------------------

<QUESTION>
{state.question}
</QUESTION>

--------------------------------------------------

<CONTEXT>
{context}
</CONTEXT>

--------------------------------------------------

Return ONLY the final answer.
"""

        response = await llm.ainvoke(prompt)
        answer_text = response.content.strip()

        # Hard hallucination guard
        if "no context found" in answer_text.lower():
            state.answer = "No context found."
            state.sources = []
            state.confidence = 0.0
            return state

        # Additional guard — if model gives generic answer
        if not answer_text or len(answer_text) < 5:
            state.answer = "No context found."
            state.sources = []
            state.confidence = 0.0
            return state

        # Build clean source list
        sources = sorted(
            {
                f"{doc.metadata.get('file_name')} – Page {doc.metadata.get('page_number')}"
                for doc in state.documents
                if doc.metadata.get("file_name") and doc.metadata.get("page_number")
            }
        )

        state.answer = answer_text

        state.report_md = (
            "# Answer\n\n"
            f"{answer_text}\n\n"
            "## Sources\n\n"
            + "\n".join(f"- {src}" for src in sources)
        )

        state.sources = sources

        # Smarter confidence scoring
        coverage_factor = min(1.0, len(state.documents) / 5)
        citation_bonus = 0.2 if "(page" in answer_text.lower() else 0.0

        state.confidence = round(min(1.0, 0.5 + 0.3 * coverage_factor + citation_bonus), 2)

        return state

    return qa_node