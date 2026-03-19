"""Page 4 - Pipeline Editor: JSON DSL editor + Graph execution with event timeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils import get_collection_name, init_sidebar, run_async

init_sidebar()

st.title("Pipeline Editor")
st.markdown("Design and run pipelines using the **JSON DSL** graph engine.")

# ----- Default DSL template -----
_DEFAULT_COLLECTION = get_collection_name()
_DEFAULT_DSL = json.dumps(
    {
        "components": {
            "begin": {
                "obj": {"component_name": "Begin", "params": {"prologue": "Welcome"}},
                "downstream": ["retrieval"],
            },
            "retrieval": {
                "obj": {
                    "component_name": "Retrieval",
                    "params": {"collection": _DEFAULT_COLLECTION, "top_k": 3},
                },
                "upstream": ["begin"],
                "downstream": ["llm"],
            },
            "llm": {
                "obj": {
                    "component_name": "LLM",
                    "params": {
                        "prompt_template": (
                            "Answer based on the following context:\n\n{retrieval@content}\n\nQuestion: {sys.query}"
                        ),
                    },
                },
                "upstream": ["retrieval"],
                "downstream": ["message"],
            },
            "message": {
                "obj": {"component_name": "Message", "params": {"content": "{llm@content}"}},
                "upstream": ["llm"],
            },
        },
        "path": ["begin"],
    },
    indent=2,
    ensure_ascii=False,
)

# ----- DSL editor -----
dsl_text = st.text_area("Pipeline DSL (JSON)", value=_DEFAULT_DSL, height=400)

# ----- Query input -----
query = st.text_input("Query", placeholder="Enter a query for the pipeline...")

# ----- Run -----
if st.button("Run Pipeline", type="primary"):
    if not query:
        st.warning("Please enter a query.")
        st.stop()

    # Validate JSON
    try:
        dsl_dict = json.loads(dsl_text)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    from fastrag.pipeline.graph import Graph

    st.divider()
    st.subheader("Execution Timeline")

    try:

        async def run_pipeline():
            graph = Graph(dsl_dict)
            events = []
            async for event in graph.run(query):
                events.append(event)
            return events

        events = run_async(run_pipeline())

        for event in events:
            evt_type = event["event"]
            data = event["data"]

            if evt_type == "workflow_started":
                st.markdown(f"**Workflow Started** | query: `{data.get('query', '')}`")

            elif evt_type == "node_started":
                st.markdown(f"  **Node Started**: `{data['node_id']}` ({data['component']})")

            elif evt_type == "node_finished":
                elapsed = data.get("elapsed", 0)
                error = data.get("error")
                outputs = data.get("outputs", {})

                if error:
                    st.error(f"  Node `{data['node_id']}` FAILED ({elapsed}s): {error}")
                else:
                    st.markdown(f"  **Node Finished**: `{data['node_id']}` ({elapsed}s)")
                    if outputs:
                        with st.expander(f"Outputs of {data['node_id']}", expanded=False):
                            # Display outputs, truncating long values
                            for k, v in outputs.items():
                                if k.startswith("_"):
                                    continue
                                v_str = str(v)
                                if len(v_str) > 500:
                                    v_str = v_str[:500] + "..."
                                st.code(f"{k}: {v_str}", language=None)

            elif evt_type == "message":
                st.divider()
                st.subheader("Pipeline Output")
                st.markdown(data.get("content", ""))

            elif evt_type == "workflow_finished":
                st.divider()
                st.success(f"Workflow completed in **{data.get('elapsed', 0)}s**")

    except Exception as e:
        st.error(f"Pipeline execution failed: {e}")

# ----- Available components reference -----
with st.expander("Available Components Reference", expanded=False):
    st.markdown(
        """
| Component | Description | Key Params |
|-----------|-------------|------------|
| **Begin** | Pipeline entry point | `prologue` |
| **Retrieval** | Vector search from Milvus | `collection`, `top_k` |
| **LLM** | Call LLM with template | `prompt_template`, `temperature` |
| **Message** | Output final message | `content` (template) |
| **Categorize** | LLM-based classification | `categories: {name: {description, examples, to}}` |
| **Switch** | Conditional branching | `conditions: [{variable, operator, value, to}]` |

**Variable syntax**: `{sys.query}`, `{component_id@output_key}`, `{env.VAR}`
"""
    )
