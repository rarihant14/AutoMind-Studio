## multi-agent-generator/__init__.py
__version__ = "0.3.1"

from .model_inference import (
    ModelInference,
    Message
)

from .frameworks import (
    create_crewai_code,
    create_crewai_flow_code,
    create_langgraph_code,
    create_react_code
)