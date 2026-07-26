# graph/workflow.py
# --------------------
# WHY: This file is the heart of the "Agentic AI" part of the project. It
#      wires our 5 separate agents together into ONE LangGraph StateGraph,
#      so they run in a defined order and can make decisions ("conditional
#      edges") about what to do next.
# WHAT: Builds and returns a compiled LangGraph graph:
#
#           START
#             |
#             v
#         Profiler Agent
#             |
#             v
#         Insight Agent
#             |
#             v
#        (decision node)
#          /        \
#         v          v
#     RAG Agent   Chart Agent
#         \          /
#          v        v
#         Report Agent
#             |
#             v
#            END
#
# HOW: LangGraph's StateGraph lets us add_node() for each agent, add_edge()
#      for fixed connections, and add_conditional_edges() for "decision"
#      points where the next step depends on the current state.

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from agents.profiler_agent import run_profiler_agent
from agents.insight_agent import run_insight_agent
from agents.rag_agent import run_rag_agent
from agents.chart_agent import run_chart_agent
from agents.report_agent import run_report_agent


def route_after_insight(state: AgentState) -> str:
    """
    WHY: This is a "conditional edge" function. LangGraph calls it after the
         Insight Agent finishes, and uses its return value to decide which
         node to run next.
    WHAT: Looks at the state to see whether the user asked a question and/or
          requested a chart, and routes accordingly.
    HOW: Simple if/elif logic - beginners can trace exactly why the graph
         goes one way or another.
    """
    if state.get("user_question"):
        # The user asked a question -> go answer it with RAG first.
        return "rag"
    elif state.get("chart_requested"):
        # No question, but a chart was requested -> go straight to charting.
        return "chart"
    else:
        # Neither a question nor a chart -> skip straight to the report.
        return "report"


def route_after_rag(state: AgentState) -> str:
    """
    WHY: After answering the user's question, we still need to check if a
         chart was ALSO requested before moving on to the final report.
    WHAT: Routes to the Chart Agent if a chart was requested, otherwise
          skips straight to the Report Agent.
    HOW: A single if/else check on the chart_requested flag in state.
    """
    if state.get("chart_requested"):
        return "chart"
    else:
        return "report"


def build_workflow():
    """
    WHY: This function assembles all 5 agents into a single runnable
         LangGraph workflow. Keeping this assembly in one function makes the
         overall "shape" of our agentic system easy to see at a glance.
    WHAT: Returns a COMPILED LangGraph graph, ready to call .invoke(state) on.
    HOW: 1) Create a StateGraph using our AgentState schema.
         2) Register each agent function as a "node".
         3) Connect nodes with fixed edges and conditional edges.
         4) Compile the graph into a runnable object.
    """
    # Step 1: Create a new graph that will pass around our AgentState dictionary.
    workflow = StateGraph(AgentState)

    # Step 2: Register every agent as a node. The string is the node's name;
    # the function is what actually runs when LangGraph reaches that node.
    # NOTE: Node names must NOT be identical to a key in AgentState (e.g. plain
    # "report" clashes with state["report"]), so every node is named "<agent>_agent".
    workflow.add_node("profiler_agent", run_profiler_agent)
    workflow.add_node("insight_agent", run_insight_agent)
    workflow.add_node("rag_agent", run_rag_agent)
    workflow.add_node("chart_agent", run_chart_agent)
    workflow.add_node("report_agent", run_report_agent)

    # Step 3a: Set the entry point - the very first node LangGraph runs.
    workflow.set_entry_point("profiler_agent")

    # Step 3b: Fixed edge - Profiler always leads to Insight.
    workflow.add_edge("profiler_agent", "insight_agent")

    # Step 3c: Conditional edge - after Insight, the NEXT node depends on
    # what the user asked for. route_after_insight() decides that.
    workflow.add_conditional_edges(
        "insight_agent",
        route_after_insight,
        {
            "rag": "rag_agent",
            "chart": "chart_agent",
            "report": "report_agent",
        },
    )

    # Step 3d: Another conditional edge - after answering the question (RAG),
    # check if a chart is ALSO needed before moving to the report.
    workflow.add_conditional_edges(
        "rag_agent",
        route_after_rag,
        {
            "chart": "chart_agent",
            "report": "report_agent",
        },
    )

    # Step 3e: Chart always leads to the Report Agent (the final step).
    workflow.add_edge("chart_agent", "report_agent")

    # Step 3f: Report is the last node - it leads to END.
    workflow.add_edge("report_agent", END)

    # Step 4: Compile the graph so it can actually be run with .invoke().
    compiled_graph = workflow.compile()
    return compiled_graph


def run_workflow(df, dataset_name: str = "dataset.csv", user_question: str = None, chart_requested: bool = False,
                  chart_type: str = None, chart_x_axis: str = None, chart_y_axis: str = None) -> AgentState:
    """
    WHY: Streamlit pages need one simple function to call that runs the ENTIRE
         agentic workflow end-to-end, without worrying about LangGraph internals.
    WHAT: Builds the initial state, compiles the graph, runs it, and returns
          the final state (which contains every agent's output).
    HOW: Constructs the AgentState dictionary with the inputs, then calls
         graph.invoke(initial_state).
    """
    graph = build_workflow()

    initial_state: AgentState = {
        "uploaded_dataframe": df,
        "dataset_name": dataset_name,
        "user_question": user_question,
        "chart_requested": chart_requested,
        "chart_type": chart_type,
        "chart_x_axis": chart_x_axis,
        "chart_y_axis": chart_y_axis,
        "messages": [],
    }

    final_state = graph.invoke(initial_state)
    return final_state


def stream_workflow_with_trace(df, dataset_name: str = "dataset.csv", user_question: str = None,
                                chart_requested: bool = False, chart_type: str = None,
                                chart_x_axis: str = None, chart_y_axis: str = None):
    """
    WHY: This is a BONUS teaching feature - it lets the Streamlit UI show
         "which agent is currently running" and how long each one took,
         which makes the abstract idea of a "graph of agents" feel real.
    WHAT: A generator that yields (node_name, elapsed_seconds, state_so_far)
          one step at a time, instead of waiting for the whole graph to finish.
    HOW: LangGraph's graph.stream(..., stream_mode="updates") yields a small
         dictionary after EACH node finishes, in the shape {node_name: state}.
         We just time how long each step takes and yield it upward.
    """
    import time

    graph = build_workflow()

    initial_state: AgentState = {
        "uploaded_dataframe": df,
        "dataset_name": dataset_name,
        "user_question": user_question,
        "chart_requested": chart_requested,
        "chart_type": chart_type,
        "chart_x_axis": chart_x_axis,
        "chart_y_axis": chart_y_axis,
        "messages": [],
    }

    last_step_time = time.time()

    # stream_mode="updates" gives us one dictionary per node: {node_name: new_state}
    for step_output in graph.stream(initial_state, stream_mode="updates"):
        node_name = list(step_output.keys())[0]
        node_state = step_output[node_name]

        now = time.time()
        elapsed_seconds = round(now - last_step_time, 2)
        last_step_time = now

        yield node_name, elapsed_seconds, node_state
