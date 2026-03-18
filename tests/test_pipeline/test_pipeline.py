import pytest

from fastrag.pipeline.context import PipelineContext
from fastrag.pipeline.base import ComponentBase, ComponentParam
from fastrag.pipeline.registry import register_component, register_param, get_component_class, list_components
from fastrag.pipeline.graph import Graph


class TestPipelineContext:
    def test_globals(self):
        ctx = PipelineContext(globals={"sys.query": "hello"})
        assert ctx.get_global("sys.query") == "hello"

    def test_set_get_variable(self):
        ctx = PipelineContext()
        ctx.set_variable("sys.query", "hello")
        assert ctx.get_variable("sys.query") == "hello"

    def test_component_output(self):
        ctx = PipelineContext()
        ctx.set_component_output("llm_0", {"content": "response"})
        assert ctx.get_variable("llm_0@content") == "response"

    def test_nested_access(self):
        ctx = PipelineContext()
        ctx.set_component_output("r_0", {"data": {"items": [{"name": "a"}, {"name": "b"}]}})
        assert ctx.get_variable("r_0@data.items.0.name") == "a"
        assert ctx.get_variable("r_0@data.items.1.name") == "b"

    def test_resolve_template(self):
        ctx = PipelineContext(globals={"sys.query": "what is RAG?"})
        ctx.set_component_output("r_0", {"content": "RAG is ..."})
        template = "Question: {sys.query}\nContext: {r_0@content}"
        result = ctx.resolve_template(template)
        assert result == "Question: what is RAG?\nContext: RAG is ..."

    def test_resolve_missing_variable(self):
        ctx = PipelineContext()
        assert ctx.resolve_template("Hello {sys.missing}") == "Hello {sys.missing}"


class TestComponentRegistry:
    def test_list_components(self):
        components = list_components()
        assert "Begin" in components
        assert "LLM" in components
        assert "Message" in components
        assert "Retrieval" in components
        assert "Categorize" in components
        assert "Switch" in components

    def test_get_component_class(self):
        cls = get_component_class("Begin")
        assert issubclass(cls, ComponentBase)

    def test_unknown_component(self):
        with pytest.raises(KeyError, match="Unknown component"):
            get_component_class("NonExistent")


class TestGraphBasic:
    @pytest.mark.asyncio
    async def test_simple_pipeline(self):
        """Test: Begin -> Message pipeline (no LLM, no external services)."""
        dsl = {
            "components": {
                "begin": {
                    "obj": {"component_name": "Begin", "params": {}},
                    "upstream": [],
                    "downstream": ["msg_0"],
                },
                "msg_0": {
                    "obj": {
                        "component_name": "Message",
                        "params": {"content": "You asked: {sys.query}"},
                    },
                    "upstream": ["begin"],
                    "downstream": [],
                },
            },
            "globals": {"sys.query": ""},
            "path": ["begin"],
        }

        graph = Graph(dsl)
        events = []
        async for event in graph.run(query="hello"):
            events.append(event)

        event_types = [e["event"] for e in events]
        assert "workflow_started" in event_types
        assert "workflow_finished" in event_types
        assert "message" in event_types

        msg_event = next(e for e in events if e["event"] == "message")
        assert msg_event["data"]["content"] == "You asked: hello"

    @pytest.mark.asyncio
    async def test_run_to_completion(self):
        dsl = {
            "components": {
                "begin": {
                    "obj": {"component_name": "Begin", "params": {}},
                    "upstream": [],
                    "downstream": ["msg_0"],
                },
                "msg_0": {
                    "obj": {
                        "component_name": "Message",
                        "params": {"content": "Echo: {sys.query}"},
                    },
                    "upstream": ["begin"],
                    "downstream": [],
                },
            },
            "globals": {},
            "path": ["begin"],
        }

        graph = Graph(dsl)
        result = await graph.run_to_completion(query="test")
        assert result["message"] == "Echo: test"

    @pytest.mark.asyncio
    async def test_variable_passing(self):
        """Test that variables pass between components."""
        dsl = {
            "components": {
                "begin": {
                    "obj": {"component_name": "Begin", "params": {"prologue": "Welcome!"}},
                    "upstream": [],
                    "downstream": ["msg_0"],
                },
                "msg_0": {
                    "obj": {
                        "component_name": "Message",
                        "params": {"content": "Prologue: {begin@prologue}, Query: {begin@query}"},
                    },
                    "upstream": ["begin"],
                    "downstream": [],
                },
            },
            "globals": {"sys.query": ""},
            "path": ["begin"],
        }

        graph = Graph(dsl)
        result = await graph.run_to_completion(query="hi there")
        assert "Welcome!" in result["message"]
        assert "hi there" in result["message"]

    @pytest.mark.asyncio
    async def test_json_dsl(self):
        """Test loading from JSON string."""
        import json
        dsl = json.dumps({
            "components": {
                "begin": {
                    "obj": {"component_name": "Begin", "params": {}},
                    "upstream": [],
                    "downstream": ["msg_0"],
                },
                "msg_0": {
                    "obj": {"component_name": "Message", "params": {"content": "OK"}},
                    "upstream": ["begin"],
                    "downstream": [],
                },
            },
            "globals": {},
            "path": ["begin"],
        })

        graph = Graph(dsl)
        result = await graph.run_to_completion()
        assert result["message"] == "OK"
