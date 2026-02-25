from app.langgraph.state import ResearchState


def make_insights_node(llm):
    async def insights_node(state: ResearchState):

        # Skip if irrelevant
        if state.force_no_context or not state.documents:
            state.answer = "No context found."
            state.sources = []
            state.confidence = 0.0
            return state

        # Build structured context
        context = "\n\n".join(
            f"(Page {doc.metadata.get('page_number')})\n{doc.page_content}"
            for doc in state.documents
            if doc.page_content
        )

        # Slightly higher temperature for analytical creativity
        analytical_llm = llm.bind(temperature=0.7)

        prompt = f"""
You are an ANALYTICAL INSIGHT ENGINE.

Your task is to generate HIGH-QUALITY INSIGHTS based ONLY on the provided context.

You MUST:
- Base every insight strictly on the context.
- DO NOT use external knowledge.
- DO NOT fabricate information.
- Clearly distinguish between:
  • Observations (directly from text)
  • Insights (inferred patterns)
  • Recommendations (logical next steps)
- Reference supporting pages when applicable: (Page X)
- Be analytical but concise.
- Use bullet points.
- DO NOT use markdown headings.

If the context does not contain sufficient information for insights,
return EXACTLY:
"No context found."

--------------------------------------------------

<CONTEXT>
{context}
</CONTEXT>

--------------------------------------------------

Return only the insights.
"""

        response = await analytical_llm.ainvoke(prompt)
        insight_text = response.content.strip()

        # Hallucination guard
        if "no context found" in insight_text.lower():
            state.answer = "No context found."
            state.sources = []
            state.confidence = 0.0
            return state

        if not insight_text or len(insight_text) < 10:
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

        state.answer = insight_text

        state.report_md = (
            "# Analytical Insights\n\n"
            f"{insight_text}\n\n"
            "## Sources\n\n"
            + "\n".join(f"- {src}" for src in sources)
        )

        state.sources = sources

        # Insight confidence scoring
        coverage_factor = min(1.0, len(state.documents) / 5)
        citation_bonus = 0.15 if "(page" in insight_text.lower() else 0.0

        state.confidence = round(min(1.0, 0.55 + 0.35 * coverage_factor + citation_bonus), 2)

        return state

    return insights_node