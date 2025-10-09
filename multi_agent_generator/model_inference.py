"""
Model inference utilities using LiteLLM for multiple providers.
Supports Gemini 2.0 Flash and Groq.
"""
import os
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
from dotenv import load_dotenv
from litellm import completion

# Optional import for Streamlit session (if running inside app)
try:
    import streamlit as st
except ImportError:
    st = None

# Load environment variables
load_dotenv()


class Message(BaseModel):
    role: str
    content: str


class ModelInference:
    """
    Unified LiteLLM-based model inference class.
    Supports Gemini, Groq, OpenAI, WatsonX, Ollama, Anthropic, etc. via LiteLLM.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **default_params
    ):
        self.model = model
        self.api_base = api_base or os.getenv("API_BASE")
        self.default_params = default_params

        model_lower = model.lower()

        # 1️⃣ Check if API key was provided directly (from UI)
        self.api_key = api_key

        # 2️⃣ Otherwise, try Streamlit session state
        if not self.api_key and st is not None and "session_state" in dir(st):
            if "gemini" in model_lower:
                self.api_key = st.session_state.get("gemini_api_key", "")
            elif "groq" in model_lower:
                self.api_key = st.session_state.get("groq_api_key", "")
            elif "openai" in model_lower:
                self.api_key = st.session_state.get("openai_api_key", "")
            elif "watsonx" in model_lower:
                self.api_key = st.session_state.get("watsonx_api_key", "")
            else:
                self.api_key = st.session_state.get("api_key", "")

        # 3️⃣ Lastly, fallback to environment variable if none provided
        if not self.api_key:
            if "gemini" in model_lower:
                self.api_key = os.getenv("GEMINI_API_KEY")
            elif "groq" in model_lower:
                self.api_key = os.getenv("GROQ_API_KEY")
            elif "openai" in model_lower:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif "watsonx" in model_lower:
                self.api_key = os.getenv("WATSONX_API_KEY")
            else:
                self.api_key = os.getenv("API_KEY")

        
        if not self.api_key and st is not None:
            st.warning(f"No API key found for model '{model}'. Please provide one in the sidebar.")
        elif not self.api_key:
            print(f"⚠️ Warning: No API key found for model '{model}'.")

    def generate_text(self, messages: List[Union[Dict, Message]], **override_params) -> str:
        """Generate text using LiteLLM."""
        try:
            msg_list = [m.dict() if isinstance(m, Message) else m for m in messages]
            response = completion(
                model=self.model,
                messages=msg_list,
                api_key=self.api_key,
                api_base=self.api_base,
                **{**self.default_params, **override_params}
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}")
