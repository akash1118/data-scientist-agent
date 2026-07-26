# tools/chart_tools.py
# ----------------------
# WHY: The Chart Agent decides WHAT chart to draw (using the LLM), but the
#      actual drawing is just plotting code - no AI needed for that part.
#      Separating "decide" (agent) from "draw" (tool) is a common and
#      beginner-friendly pattern in agentic systems.
# WHAT: Functions that take a DataFrame + chart settings and return a
#       ready-to-display Plotly figure.
# HOW: Uses Plotly Express (px), which creates nice interactive charts with
#      very little code - perfect for a teaching project.

import pandas as pd
import plotly.express as px


# Every chart type we support maps to a short, readable name shown in the UI.
SUPPORTED_CHART_TYPES = ["Bar", "Line", "Pie", "Histogram", "Scatter", "Box Plot", "Heatmap"]


def create_chart(df: pd.DataFrame, chart_type: str, x_axis: str = None, y_axis: str = None):
    """
    WHY: This is the single entry point the rest of the app calls to turn
         "chart_type + column choices" into an actual chart image.
    WHAT: Builds and returns a Plotly figure based on the requested chart_type.
    HOW: A simple if/elif chain - one branch per chart type. Kept intentionally
         simple (no clever abstractions) so beginners can read it top to bottom.
    """
    chart_type = chart_type.lower()

    if chart_type == "bar":
        fig = px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart: {y_axis} by {x_axis}")

    elif chart_type == "line":
        fig = px.line(df, x=x_axis, y=y_axis, title=f"Line Chart: {y_axis} over {x_axis}")

    elif chart_type == "pie":
        # Pie charts need one "names" column (categories) and one "values" column.
        fig = px.pie(df, names=x_axis, values=y_axis, title=f"Pie Chart: {y_axis} share by {x_axis}")

    elif chart_type == "histogram":
        # Histograms only need ONE column - they show the distribution of values.
        fig = px.histogram(df, x=x_axis, title=f"Histogram of {x_axis}")

    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot: {y_axis} vs {x_axis}")

    elif chart_type == "box plot":
        fig = px.box(df, x=x_axis, y=y_axis, title=f"Box Plot: {y_axis} by {x_axis}")

    elif chart_type == "heatmap":
        # A heatmap here shows the correlation between numeric columns.
        numeric_df = df.select_dtypes(include=["number"])
        correlation_matrix = numeric_df.corr().round(2)
        fig = px.imshow(correlation_matrix, text_auto=True, title="Correlation Heatmap")

    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    # A clean, readable layout that works in both light and dark Streamlit themes.
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))
    return fig
