TASK_CLASSIFIER_PROMPT = """
Classify the user request into ONE of the following:
- summarize
- qa
- compare
- extract
- insights

Return ONLY the task name.
"""

async def classify_task(llm, question: str) -> str:
    response = await llm.ainvoke(
        TASK_CLASSIFIER_PROMPT + f"\nQuestion:\n{question}"
    )
    return response.content.strip().lower()
