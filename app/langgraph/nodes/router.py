# app/langgraph/nodes/router.py
from app.langgraph.state import ResearchState

ROUTER_PROMPT = """
You are a research task router.

Classify the user request into ONE of the following:
- summarize
- qa
- compare
- extract
- insights

User question:
"{question}"

Return ONLY the task name.
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

        # safety guard
        if task not in {"summarize", "qa", "compare", "extract", "insights"}:
            task = "qa" # default

        state.task_type = task
        return state

    return router_node
