from app.langgraph.state import ResearchState


def make_summarize_node(llm):
    async def summarize_node(state: ResearchState):

        # Non-context safety check
        if state.force_no_context or not state.documents:
            state.answer = "No context found."
            state.sources = []
            state.confidence = 0.0
            return state

        # Build context safely
        context = "\n\n".join(
            f"(Page {doc.metadata.get('page_number')})\n{doc.page_content}"
            for doc in state.documents
        )

        prompt = f"""
You are a DOCUMENT SUMMARIZATION assistant.

Your task is to generate a concise, structured summary using ONLY the provided context.

You MUST:
- Use ONLY the given context.
- DO NOT add external knowledge.
- DO NOT infer beyond the text.
- DO NOT fabricate information.
- Each bullet point MUST include its source page in format: (Page X).
- Use clear bullet points.
- DO NOT use markdown headings.
- DO NOT add commentary outside the summary.

If the context does not contain meaningful content,
return EXACTLY:
"No context found."

--------------------------------------------------

<CONTEXT>
{context}
</CONTEXT>

--------------------------------------------------

Return only the bullet summary.
"""

        response = await llm.ainvoke(prompt)
        summary_text = response.content.strip()

        # Hard hallucination guard
        if "no context found" in summary_text.lower():
            state.answer = "No context found."
            state.sources = []
            state.confidence = 0.0
            return state

        # Generate clean source list
        sources = sorted(
            {
                f"{doc.metadata.get('file_name')} – Page {doc.metadata.get('page_number')}"
                for doc in state.documents
                if doc.metadata.get("file_name") and doc.metadata.get("page_number")
            }
        )

        state.answer = summary_text

        # Structured markdown report
        state.report_md = (
            "# Document Summary\n\n"
            "## Key Findings\n\n"
            f"{summary_text}\n\n"
            "## Sources\n\n"
            + "\n".join(f"- {src}" for src in sources)
        )

        state.sources = sources

        # Improved confidence calculation
        coverage_factor = min(1.0, len(state.documents) / 5)
        state.confidence = round(0.5 + 0.4 * coverage_factor, 2)

        return state

    return summarize_node