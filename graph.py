from langgraph.graph import StateGraph, START, END
from state import AnalyticsState
from agents.nodes import profiler_node, cleaner_node, analytics_node, insights_node, dashboard_plan_node

def build_graph():
    workflow = StateGraph(AnalyticsState)
    workflow.add_node("profile", profiler_node)
    workflow.add_node("clean", cleaner_node)
    workflow.add_node("analyze", analytics_node)
    workflow.add_node("insights", insights_node)
    workflow.add_node("dashboard_plan", dashboard_plan_node)

    workflow.add_edge(START, "profile")
    workflow.add_edge("profile", "clean")
    workflow.add_edge("clean", "analyze")
    workflow.add_edge("analyze", "insights")
    workflow.add_edge("insights", "dashboard_plan")
    workflow.add_edge("dashboard_plan", END)
    return workflow.compile()

analytics_graph = build_graph()
