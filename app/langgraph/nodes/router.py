from app.langgraph.state import ResearchState

ALLOWED_TASKS = {"summarize", "compare", "extract", "insights", "qa"}

ROUTER_PROMPT = """
You are a TASK ROUTER for a DOCUMENT-BASED AI system.

Your responsibility is to classify the USER_QUERY into EXACTLY ONE task.

You MUST return ONLY one lowercase label.
You MUST NOT return explanations.
You MUST NOT return punctuation.
You MUST NOT return multiple labels.

If the query is unrelated to the uploaded documents,
return EXACTLY:
"irrelevant"

--------------------------------------------------

<TASK_PRIORITY_ORDER>

1. summarize
2. compare
3. extract
4. insights
5. qa

</TASK_PRIORITY_ORDER>

--------------------------------------------------

<TASK_DEFINITIONS>

"summarize" → Requests for overview, recap, brief explanation, main points.
**Even if the word "summarize" is not used**, but the tone implies condensation,
classify as summarize.

"compare" → Explicit comparison between TWO OR MORE entities.
Keywords like compare, difference, vs, contrast.

"extract" → Request for specific factual data.
Statistics, numbers, dates, names, or structured list extraction.
"extract":
ONLY when the user explicitly requests:
- numbers
- statistics
- metrics
- tabular data
- percentages
- dates
- measurable values

If the request is about concepts, themes, tone, language, or qualitative elements,
DO NOT classify as extract.

"insights" → Analytical reasoning.
Trends, risks, interpretation, implications.

"qa" → Direct factual question answerable from document.
WHO / WHAT / WHEN / WHERE / WHY / HOW.
Default fallback if uncertain.

--------------------------------------------------

IMPORTANT:

If the query is about:
- tone
- sentiment
- language quality
- inappropriate/vulgar/profane content
- conceptual meaning
- themes

Then DO NOT classify as extract.
Use:
- qa (if direct answerable)
- insights (if analytical)
--------------------------------------------------

<CLASSIFICATION_RULES>

- Evaluate INTENT, not keywords alone.
- Apply PRIORITY ORDER if multiple intents appear.
- If summary + extraction → choose summarize.
- If comparison + analysis → choose insight.
- If uncertain → choose qna.
- If clearly unrelated to document context → choose "irrelevant".

--------------------------------------------------

<INPUT>
USER_QUERY:
"{question}"
</INPUT>

<OUTPUT>
Return ONLY one of:
"summarize"
"compare"
"extract"
"insights"
"qna"
"irrelevant"
</OUTPUT>
"""

def make_router_node(llm):
    async def router_node(state: ResearchState):
        """Graph routing logic for LangGraph nodes

        Args:
            state (ResearchState): Pydantic model of Agent State

        Returns:
            Assigned Task value
        """
        response = await llm.ainvoke(
            ROUTER_PROMPT.format(question=state.question)
        )

        task = response.content.strip().lower()

        # Hard sanitize output
        if task not in ALLOWED_TASKS and task != "irrelevant":
            task = "qa"

        # Non-relevant fallback handling
        if task == "irrelevant":
            state.task_type = "qa"
            state.force_no_context = True
            return state

        state.task_type = task
        state.force_no_context = False

        return state

    return router_node