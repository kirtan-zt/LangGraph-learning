from app.langgraph.state import ResearchState

def make_summarize_node(llm):
    async def summarize_node(state: ResearchState):
        context = "\n\n".join(
            f"(Page {doc.metadata.get('page_number')})\n{doc.page_content}"
            for doc in state.documents
        )

        prompt = f"""
Summarize the following content into concise bullet points.
Each bullet must reference the page number it comes from.
DO NOT use markdown headings.

Content:
{context}
"""
        response = await llm.ainvoke(prompt)
        summary_text = response.content.strip()

        sources = sorted(
            {
                f"{doc.metadata.get('file_name')} – Page {doc.metadata.get('page_number')}"
                for doc in state.documents
            }
        )
        state.answer = summary_text
        state.report_md = (
            "# Document Summary\n\n"
            "## Key Findings\n\n"
            f"{summary_text}\n\n"
            "## Sources\n\n"
            + "\n".join(f"- {src}" for src in sources)
        )

        state.sources = sources
        state.confidence = min(1.0, 0.6 + 0.05 * len(state.documents))
        return state

    return summarize_node