from app.langgraph.state import ResearchState


def make_compare_node(llm):
    async def compare_node(state: ResearchState):

        # Skip if irrelevant or no docs
        if state.force_no_context or not state.documents:
            state.answer = "No context found."
            state.sources = []
            state.confidence = 0.0
            return state

        # Group documents by file_name safely
        grouped = {}
        for d in state.documents:
            name = d.metadata.get("file_name")
            if not name:
                continue
            grouped.setdefault(name, []).append(
                f"(Page {d.metadata.get('page_number')})\n{d.page_content}"
            )

        # Need at least 2 documents for comparison
        if len(grouped) < 2:
            state.answer = "Not enough documents to compare."
            state.sources = []
            state.confidence = 0.0
            return state

        # Build structured comparison context
        formatted_docs = ""
        for name, contents in grouped.items():
            formatted_docs += f"\n\n### DOCUMENT: {name}\n"
            formatted_docs += "\n\n".join(contents)

        # Keep deterministic temperature
        compare_llm = llm.bind(temperature=0.3)

        prompt = f"""
You are a STRICT DOCUMENT COMPARISON ENGINE.

Your task is to compare the documents using ONLY the provided context.

You MUST:
- Use ONLY the provided document content.
- DO NOT use external knowledge.
- DO NOT fabricate missing details.
- If comparison cannot be made from context, return EXACTLY:
  "No context found."
- Produce a MARKDOWN TABLE.
- First column MUST be "Criteria".
- Other columns MUST be document names.
- Each comparison cell should reference supporting page numbers (Page X).
- Keep analysis concise and factual.

--------------------------------------------------

<DOCUMENTS>
{formatted_docs}
</DOCUMENTS>

--------------------------------------------------

Return ONLY the markdown table.
"""

        response = await compare_llm.ainvoke(prompt)
        comparison_text = response.content.strip()

        # Hallucination guard
        if "no context found" in comparison_text.lower():
            state.answer = "No context found."
            state.sources = []
            state.confidence = 0.0
            return state

        if "|" not in comparison_text:
            state.answer = "Comparison could not be generated from context."
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

        state.answer = comparison_text

        state.report_md = (
            "# Document Comparison\n\n"
            f"{comparison_text}\n\n"
            "## Sources\n\n"
            + "\n".join(f"- {src}" for src in sources)
        )

        state.sources = sources

        # Smarter confidence
        doc_count_factor = min(1.0, len(grouped) / 3)
        citation_bonus = 0.15 if "(page" in comparison_text.lower() else 0.0

        state.confidence = round(min(1.0, 0.55 + 0.35 * doc_count_factor + citation_bonus), 2)

        return state

    return compare_node