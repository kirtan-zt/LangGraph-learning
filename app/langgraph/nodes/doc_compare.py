from app.langgraph.state import ResearchState

def make_compare_node(llm):
    async def compare_node(state: ResearchState):
        grouped = {}
        for d in state.documents:
            name = d.metadata["file_name"]
            grouped.setdefault(name, []).append(d.page_content)

        prompt = f"""
Compare the following documents side by side.
Use a markdown table.

Documents:
{grouped}
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
    return compare_node