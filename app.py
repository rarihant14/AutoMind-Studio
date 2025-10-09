"""
Streamlit UI for Multi-Agent Generator (Gemini, Groq, OpenAI, WatsonX).
"""
import os
import time
import streamlit as st
import json
from dotenv import load_dotenv

from multi_agent_generator.generator import AgentGenerator
from multi_agent_generator.frameworks.crewai_generator import create_crewai_code
from multi_agent_generator.frameworks.langgraph_generator import create_langgraph_code
from multi_agent_generator.frameworks.react_generator import create_react_code
from multi_agent_generator.frameworks.crewai_flow_generator import create_crewai_flow_code

# Load environment variables
load_dotenv()

def create_code_block(config, framework):
    """Generate code for the selected framework."""
    if framework == "crewai":
        return create_crewai_code(config)
    elif framework == "crewai-flow":
        return create_crewai_flow_code(config)
    elif framework == "langgraph":
        return create_langgraph_code(config)
   #  elif framework == "react":
   #     return create_react_code(config)
    else:
        return "# Invalid framework"

def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(page_title="Multi-Framework Agent Generator", page_icon="🚀", layout="wide")
    
    st.title("Multi-Framework Agent Generator")
    st.write("Generate agent code for different frameworks using Gemini, Groq, OpenAI, or WatsonX!")

    # === Session State Initialization ===
    if 'model_provider' not in st.session_state:
        st.session_state.model_provider = 'gemini'

    for key in ['openai_api_key', 'watsonx_api_key', 'watsonx_project_id', 'gemini_api_key', 'groq_api_key']:
        if key not in st.session_state:
            st.session_state[key] = ''

    # === Sidebar: Provider Selection ===
    st.sidebar.title("🤖 LLM Provider Settings")
    model_provider = st.sidebar.radio(
        "Choose LLM Provider:",
        ["Gemini", "Groq", "OpenAI", "WatsonX"],
        index=["gemini", "groq", "openai", "watsonx"].index(st.session_state.model_provider),
        key="provider_radio"
    )
    st.session_state.model_provider = model_provider.lower()

    # === Provider Badge ===
    badges = {
        "gemini": "![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)",
        "groq": "![Groq](https://img.shields.io/badge/Groq-FF4B00?style=for-the-badge&logo=groq&logoColor=white)",
        "openai": "![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)",
        "watsonx": "![IBM](https://img.shields.io/badge/IBM%20WatsonX-052FAD?style=for-the-badge&logo=ibm&logoColor=white)"
    }
    st.sidebar.markdown(badges[st.session_state.model_provider])

    # === Sidebar: API Key Management ===
    with st.sidebar.expander("🔑 API Credentials", expanded=False):
        provider = st.session_state.model_provider

        if provider == "gemini":
            key_env = os.getenv("GEMINI_API_KEY", "")
            if key_env:
                st.success("Gemini API Key found in environment.")
                st.session_state.gemini_api_key = key_env
            else:
                api_key = st.text_input("Enter Gemini API Key:", value=st.session_state.gemini_api_key, type="password")
                if api_key:
                    st.session_state.gemini_api_key = api_key
                    st.success("Gemini API Key saved for this session.")

        elif provider == "groq":
            key_env = os.getenv("GROQ_API_KEY", "")
            if key_env:
                st.success("Groq API Key found in environment.")
                st.session_state.groq_api_key = key_env
            else:
                api_key = st.text_input("Enter Groq API Key:", value=st.session_state.groq_api_key, type="password")
                if api_key:
                    st.session_state.groq_api_key = api_key
                    st.success("Groq API Key saved for this session.")

        elif provider == "openai":
            key_env = os.getenv("OPENAI_API_KEY", "")
            if key_env:
                st.success("OpenAI API Key found in environment.")
                st.session_state.openai_api_key = key_env
            else:
                api_key = st.text_input("Enter OpenAI API Key:", value=st.session_state.openai_api_key, type="password")
                if api_key:
                    st.session_state.openai_api_key = api_key
                    st.success("OpenAI API Key saved for this session.")

        else:  # watsonx
            watsonx_key_env = os.getenv("WATSONX_API_KEY", "")
            watsonx_project_env = os.getenv("WATSONX_PROJECT_ID", "")
            if watsonx_key_env and watsonx_project_env:
                st.success("WatsonX credentials found in environment.")
                st.session_state.watsonx_api_key = watsonx_key_env
                st.session_state.watsonx_project_id = watsonx_project_env
            else:
                col1, col2 = st.columns(2)
                with col1:
                    api_key = st.text_input("WatsonX API Key:", value=st.session_state.watsonx_api_key, type="password")
                    if api_key:
                        st.session_state.watsonx_api_key = api_key
                with col2:
                    project_id = st.text_input("WatsonX Project ID:", value=st.session_state.watsonx_project_id)
                    if project_id:
                        st.session_state.watsonx_project_id = project_id
                if st.session_state.watsonx_api_key and st.session_state.watsonx_project_id:
                    st.success("WatsonX credentials saved for this session.")

    # === Sidebar: Model Info ===
    with st.sidebar.expander("ℹ️ Model Information", expanded=False):
        model_info = {
            "gemini": ("Gemini 2.0 Flash", "Google's multimodal model optimized for speed and efficiency."),
            "groq": ("LLaMA 3.1 70B Versatile (via Groq)", "Ultra-fast inference with low latency hardware."),
            "openai": ("GPT-4o-mini", "OpenAI’s compact model for reasoning and code generation."),
            "watsonx": ("Llama-3-70B-Instruct (via WatsonX)", "Enterprise-grade model with IBM governance.")
        }
        name, desc = model_info[st.session_state.model_provider]
        st.write(f"**Model:** {name}")
        st.write(desc)

    # === Sidebar: Frameworks ===
    st.sidebar.title("🔄 Framework Selection")
    framework = st.sidebar.radio(
        "Choose a framework:",
        ["crewai", "crewai-flow", "langgraph",],  # "react"],
        format_func=lambda x: {
            "crewai": "CrewAI",
            "crewai-flow": "CrewAI Flow",
            "langgraph": "LangGraph",
            #"react": "ReAct Framework"
        }[x],
    )

    st.sidebar.markdown({
        "crewai": "**CrewAI** lets specialized agents collaborate sequentially or hierarchically.",
        "crewai-flow": "**CrewAI Flow** enables event-driven orchestration and stateful workflows.",
        "langgraph": "**LangGraph** builds directed graphs of LLM-powered nodes and edges.",
        "react": "**ReAct** blends reasoning traces with tool-based actions."
    }[framework])

    # === Example Prompts ===
    st.sidebar.title("📚 Example Prompts")
    example_prompts = {
        "Research Assistant": "I need a research assistant that summarizes papers and answers questions",
        "Content Creation": "I need a team to create viral social media content and manage our brand presence",
        "Data Analysis": "I need a team to analyze customer data and create visualizations",
        "Technical Writing": "I need a team to create technical documentation and API guides"
    }
    selected_example = st.sidebar.selectbox("Choose an example:", list(example_prompts.keys()))

    # === Main Input Area ===
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🎯 Define Your Requirements")
        user_prompt = st.text_area("Describe what you need:", value=example_prompts[selected_example], height=100)

        if st.button(f"🚀 Generate using {model_provider} & {framework.upper()}"):
            missing = False
            prov = st.session_state.model_provider
            if prov == "gemini" and not st.session_state.gemini_api_key:
                st.error("Please enter your Gemini API Key.")
                missing = True
            elif prov == "groq" and not st.session_state.groq_api_key:
                st.error("Please enter your Groq API Key.")
                missing = True
            elif prov == "openai" and not st.session_state.openai_api_key:
                st.error("Please enter your OpenAI API Key.")
                missing = True
            elif prov == "watsonx" and (not st.session_state.watsonx_api_key or not st.session_state.watsonx_project_id):
                st.error("Please enter your WatsonX credentials.")
                missing = True

            if not missing:
                with st.spinner(f"Generating your {framework} code using {model_provider}..."):
                    generator = AgentGenerator(provider=prov)
                    config = generator.analyze_prompt(user_prompt, framework)
                    code = create_code_block(config, framework)
                    st.session_state.config = config
                    st.session_state.code = code
                    st.session_state.framework = framework
                    time.sleep(0.5)
                    st.success(f"✨ {framework.upper()} code generated successfully with {model_provider}!")
                    st.info(f"Generated using {model_info[prov][0]}")

    # === Display Results ===
    if 'config' in st.session_state:
        st.subheader("💻 Generated Code")
        st.code(st.session_state.code, language="python")
        st.download_button("💾 Download Code", st.session_state.code,
                           file_name=f"{st.session_state.framework}_agent.py", mime="text/plain")
        st.subheader("🔧 Configuration (JSON)")
        st.json(st.session_state.config)

if __name__ == "__main__":
    main()
