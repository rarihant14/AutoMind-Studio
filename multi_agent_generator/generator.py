"""
Agent configuration generator that analyzes user requirements.
Now supports Gemini 2.0 Flash and Groq providers.
"""
import os
import json
import streamlit as st
from typing import Dict, Any, Optional, List
from .model_inference import ModelInference, Message


class AgentGenerator:
    """
    Generates agent configurations based on natural language descriptions.
    Uses LiteLLM for provider-agnostic inference.
    """

    def __init__(self, provider: str = "gemini"):
        """
        Initialize the generator with the specified provider.

        Args:
            provider: The LLM provider to use (gemini, groq, openai, watsonx, etc.)
        """
        self.provider = provider.lower()
        self.model: Optional[ModelInference] = None

    def set_provider(self, provider: str):
        """Change the LLM provider dynamically."""
        self.provider = provider.lower()
        self.model = None

    def _initialize_model(self):
        """Initialize the LiteLLM ModelInference if not already done."""
        if self.model is not None:
            return

        default_models = {
            "gemini": "gemini/gemini-2.0-flash",
            "groq": "groq/llama-3.1-70b-versatile",
            "openai": "gpt-4o-mini",
            "watsonx": "watsonx/meta-llama/llama-3-3-70b-instruct",
            "ollama": "ollama/llama3.2:3b"
        }

        model_name = default_models.get(self.provider, self.provider)
        model_name = os.getenv("DEFAULT_MODEL", model_name)

        # Dynamically pick API key if in Streamlit session
        api_key = None
        if st is not None and "session_state" in dir(st):
            if self.provider == "gemini":
                api_key = st.session_state.get("gemini_api_key", "")
            elif self.provider == "groq":
                api_key = st.session_state.get("groq_api_key", "")
            elif self.provider == "openai":
                api_key = st.session_state.get("openai_api_key", "")
            elif self.provider == "watsonx":
                api_key = st.session_state.get("watsonx_api_key", "")

        self.model = ModelInference(
            model=model_name,
            api_key=api_key,
            max_tokens=1000,
            temperature=0.7,
            top_p=0.95,
        )

    def analyze_prompt(self, user_prompt: str, framework: str) -> Dict[str, Any]:
        """Analyze a natural language prompt to generate agent configuration."""
        self._initialize_model()
        system_prompt = self._get_system_prompt_for_framework(framework)

        try:
            messages: List[Message] = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt)
            ]

            response = self.model.generate_text(messages)

            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                if st is not None:
                    st.warning("Could not extract valid JSON. Using default configuration.")
                return self._get_default_config(framework)

        except Exception as e:
            if st is not None:
                st.error(f"Error analyzing prompt: {e}")
            return self._get_default_config(framework)

    # ==================== Framework Prompts ====================

    def _get_system_prompt_for_framework(self, framework: str) -> str:
        """System prompt per framework."""
        if framework == "crewai":
            return """
            You are an expert at creating AI research assistants using CrewAI.
            Suggest agents, roles, tools, and tasks.
            Follow JSON format:
            {
                "process": "sequential" or "hierarchical",
                "agents": [{"name": "", "role": "", "goal": "", "backstory": "", "tools": [], "verbose": true}],
                "tasks": [{"name": "", "description": "", "tools": [], "agent": "", "expected_output": ""}]
            }
            """
        elif framework == "crewai-flow":
            return """
            You are an expert in CrewAI Flow orchestration.
            Suggest agents, roles, and event-driven workflow tasks.
            Follow JSON format like CrewAI but emphasize 'flow' sequence.
            """
        elif framework == "langgraph":
            return """
            You are an expert in LangGraph.
            Suggest agents, nodes, and edges in JSON format:
            {
                "agents": [{"name": "", "role": "", "goal": "", "tools": [], "llm": ""}],
                "nodes": [{"name": "", "description": "", "agent": ""}],
                "edges": [{"source": "", "target": "", "condition": ""}]
            }
            """
        elif framework == "react":
            return """
            You are an expert at creating ReAct (Reasoning + Acting) agents.
            Suggest reasoning steps, actions, and example flows.
            JSON format:
            {
                "agents": [{"name": "", "role": "", "goal": "", "tools": [], "llm": ""}],
                "tools": [{"name": "", "description": "", "parameters": {}}],
                "examples": [{"query": "", "thought": "", "action": "", "observation": "", "final_answer": ""}]
            }
            """
        else:
            return "You are an AI agent framework expert. Return a valid JSON config."

    # ==================== Default Configs ====================

    def _get_default_config(self, framework: str) -> Dict[str, Any]:
        """Default fallback configs for frameworks."""
        if framework in ["crewai", "crewai-flow"]:
            return {
                "process": "sequential",
                "agents": [
                    {
                        "name": "research_specialist",
                        "role": "Research Specialist",
                        "goal": "Collect and analyze data",
                        "backstory": "Expert in research and analysis",
                        "tools": ["search_tool", "data_parser"],
                        "verbose": True,
                    },
                    {
                        "name": "writer_agent",
                        "role": "Content Writer",
                        "goal": "Draft clear content",
                        "backstory": "Professional writer",
                        "tools": ["editor_tool"],
                        "verbose": True,
                    },
                ],
                "tasks": [
                    {
                        "name": "research",
                        "description": "Collect and analyze topic data",
                        "tools": ["search_tool"],
                        "agent": "research_specialist",
                        "expected_output": "Research summary",
                    },
                    {
                        "name": "write",
                        "description": "Draft content from research",
                        "tools": ["editor_tool"],
                        "agent": "writer_agent",
                        "expected_output": "Final document",
                    },
                ],
            }

        elif framework == "langgraph":
            return {
                "agents": [
                    {
                        "name": "assistant",
                        "role": "General Assistant",
                        "goal": "Support tasks",
                        "tools": ["basic_tool"],
                        "llm": "gemini/gemini-2.0-flash"
                    }
                ],
                "nodes": [
                    {"name": "start", "description": "Initial node", "agent": "assistant"}
                ],
                "edges": [{"source": "start", "target": "END", "condition": "complete"}],
            }

        elif framework == "react":
            return {
                "agents": [
                    {
                        "name": "reasoner",
                        "role": "Reasoning Agent",
                        "goal": "Perform reasoning and actions",
                        "tools": ["search_tool"],
                        "llm": "groq/llama-3.1-70b-versatile",
                    }
                ],
                "tools": [
                    {
                        "name": "search_tool",
                        "description": "Searches information online",
                        "parameters": {"query": "search input"},
                    }
                ],
                "examples": [
                    {
                        "query": "Find AI research papers",
                        "thought": "Search latest AI publications",
                        "action": "search_tool",
                        "observation": "Found 5 papers",
                        "final_answer": "Summarized papers found.",
                    }
                ],
            }
        else:
            return {}

