from app.langgraph.state import ResearchState

def make_insights_node(llm):
    async def insights_node(state: ResearchState):
        prompt = f"""
Based on the documents, generate insights and recommendations.
Be analytical and concise.

{state.documents}
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
    return insights_node      