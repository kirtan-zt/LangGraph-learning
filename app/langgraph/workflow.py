from langgraph.graph import StateGraph, END
from app.langgraph.state import ResearchState
from app.langgraph.nodes.summarization import make_summarize_node
from app.langgraph.nodes.qa import make_qa_node
from app.langgraph.nodes.doc_compare import make_compare_node
from app.langgraph.nodes.data_extract import make_extract_node
from app.langgraph.nodes.insights import make_insights_node
from app.langgraph.nodes.router import make_router_node

def build_research_graph(llm):
    """Graph building logic with conditional edges

    Args:
        llm: Model definition

    Returns:
        Compiled LangGraph workflow
    """
    graph = StateGraph(ResearchState)

    router_node = make_router_node(llm)
    summarize_node=make_summarize_node(llm)
    qa_node=make_qa_node(llm)
    compare_node=make_compare_node(llm)
    extract_node=make_extract_node(llm)
    insights_node=make_insights_node(llm)

    graph.add_node("router", router_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("qa", qa_node)
    graph.add_node("compare", compare_node)
    graph.add_node("extract", extract_node)
    graph.add_node("insights", insights_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda state: state.task_type,
        {
            "summarize": "summarize",
            "qa": "qa",
            "compare": "compare",
            "extract": "extract",
            "insights": "insights",
        },
    )

    for node in ["summarize", "qa", "compare", "extract", "insights"]:
        graph.add_edge(node, END)
    
    return graph.compile(name="content_research_agent")