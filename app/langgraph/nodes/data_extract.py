from app.langgraph.state import ResearchState


def make_extract_node(llm):
    async def extract_node(state: ResearchState):

        # Skip if irrelevant or no documents
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

        # Deterministic extraction
        extract_llm = llm.bind(temperature=0.2)

        prompt = f"""
You are a STRICT DATA EXTRACTION ENGINE.

Your task is to extract factual data ONLY from the provided context.

You MUST:
- Extract ALL explicit numbers, statistics, dates, metrics, percentages.
- Extract structured data that resembles tables.
- Extract named entities tied to measurable values.
- DO NOT summarize.
- DO NOT interpret.
- DO NOT infer.
- DO NOT add commentary.
- DO NOT fabricate missing values.
- Every extracted row MUST reference its page number (Page X).

If no extractable structured data is found,
return EXACTLY:
"No context found."

--------------------------------------------------

<CONTEXT>
{context}
</CONTEXT>

--------------------------------------------------

OUTPUT FORMAT RULES:

- Return ONLY markdown tables.
- First column MUST be "Data Point".
- Second column MUST be "Value".
- Third column MUST be "Page Reference".
- Do NOT include explanations outside tables.
"""

        response = await extract_llm.ainvoke(prompt)
        extraction_text = response.content.strip()

        # Hallucination guard
        if "no context found" in extraction_text.lower():
            state.answer = "No context found."
            state.sources = []
            state.confidence = 0.0
            return state

        # Minimal structural validation
        if "|" not in extraction_text:
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

        state.answer = extraction_text

        state.report_md = (
            "# Extracted Data\n\n"
            f"{extraction_text}\n\n"
            "## Sources\n\n"
            + "\n".join(f"- {src}" for src in sources)
        )

        state.sources = sources

        # Extraction confidence (higher baseline because deterministic)
        coverage_factor = min(1.0, len(state.documents) / 5)
        citation_bonus = 0.15 if "(page" in extraction_text.lower() else 0.0

        state.confidence = round(min(1.0, 0.6 + 0.3 * coverage_factor + citation_bonus), 2)

        return state

    return extract_node