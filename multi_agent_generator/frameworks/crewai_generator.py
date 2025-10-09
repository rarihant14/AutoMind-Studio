from crewai import Agent as CrewAgent, Task as CrewTask, Crew, Process
from typing import List, Dict, Any
from pydantic import BaseModel, Field


def _sanitize_var_name(name: str) -> str:
    """Convert agent/task name to a valid Python variable name."""
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("'", "")
        .replace('"', "")
    )


def create_crewai_code(config: Dict[str, Any]) -> str:
    """Generate CrewAI code based on config dictionary."""
    process_type = config.get("process", "sequential").lower()

    # --- Imports ---
    code = "from crewai import Agent, Task, Crew, Process\n"
    if process_type == "sequential":
        code += "from crewai.flow.flow import Flow, listen, start\n"
    code += "from typing import Dict, Any\n"
    code += "from pydantic import BaseModel, Field\n\n"

   
    if process_type == "sequential":
        code += "# Define state model for sequential workflow\n"
        code += "class AgentState(BaseModel):\n"
        code += "    query: str = Field(default=\"\")\n"
        code += "    results: Dict[str, Any] = Field(default_factory=dict)\n"
        code += "    current_step: str = Field(default=\"\")\n\n"

    # --- Agent Definitions ---
    agent_name_to_var = {}
    for i, agent in enumerate(config["agents"]):
        var_name = f"agent_{_sanitize_var_name(agent['name'])}"
        agent_name_to_var[agent["name"]] = var_name

        code += f"# Agent: {agent['name']}\n"
        code += f"{var_name} = Agent(\n"
        code += f"    role={agent['role']!r},\n"
        code += f"    goal={agent['goal']!r},\n"
        code += f"    backstory={agent['backstory']!r},\n"
        code += f"    verbose={agent.get('verbose', True)},\n"
        code += f"    allow_delegation={agent.get('allow_delegation', False)},\n"
        code += f"    tools={agent.get('tools', [])}\n"
        code += ")\n\n"

    # --- Task Definitions ---
    for task in config["tasks"]:
        task_var = f"task_{_sanitize_var_name(task['name'])}"
        code += f"# Task: {task['name']}\n"
        code += f"{task_var} = Task(\n"
        code += f"    description={task['description']!r},\n"

        # assign correct agent
        agent_name = task.get("agent")
        if agent_name in agent_name_to_var:
            agent_var = agent_name_to_var[agent_name]
        else:
            # fallback assignment
            fallback = config["agents"][0]["name"]
            agent_var = agent_name_to_var[fallback]
            code += f"    # Auto-assigned to agent: {fallback}\n"

        code += f"    agent={agent_var},\n"
        code += f"    expected_output={task['expected_output']!r}\n"
        code += ")\n\n"

    # --- Crew Configuration ---
    code += "# Crew Configuration\n"
    code += "crew = Crew(\n"
    code += f"    agents=[{', '.join(agent_name_to_var.values())}],\n"
    code += f"    tasks=[{', '.join(f'task_{_sanitize_var_name(t['name'])}' for t in config['tasks'])}],\n"

    if process_type == "hierarchical":
        code += "    process=Process.hierarchical,\n"
        code += f"    manager_agent={list(agent_name_to_var.values())[0]},\n"
    else:
        code += "    process=Process.sequential,\n"

    code += "    verbose=True\n"
    code += ")\n\n"

    # --- Run Function ---
    code += "# Run the workflow\n"
    code += "def run_workflow(query: str):\n"
    code += "    result = crew.kickoff(inputs={\"query\": query})\n"
    code += "    return result\n\n"

    # --- Example ---
    code += "if __name__ == '__main__':\n"
    code += "    result = run_workflow('Your query here')\n"
    code += "    print(result)\n"

    return code
