"""
Generator for CrewAI Flow code.
Safely handles missing 'agents' or 'tasks' in config.
"""

from typing import Dict, Any


def create_crewai_flow_code(config: Dict[str, Any]) -> str:
    """
    Generate CrewAI Flow code from a configuration safely.
    
    Args:
        config: Dictionary containing agents, tasks, and workflow configuration.
    
    Returns:
        Generated Python code as a string.
    """
    # Ensure essential keys exist to prevent KeyErrors
    config.setdefault("agents", [])
    config.setdefault("tasks", [])

    # Basic imports
    code = "from crewai import Agent, Task, Crew\n"
    code += "from crewai.flow.flow import Flow, listen, start\n"
    code += "from typing import Dict, List, Any\n"
    code += "from pydantic import BaseModel, Field\n\n"

    # Flow state model
    code += "# Define flow state\n"
    code += "class AgentState(BaseModel):\n"
    code += "    query: str = Field(default=\"\")\n"
    code += "    results: Dict[str, Any] = Field(default_factory=dict)\n"
    code += "    current_step: str = Field(default=\"\")\n\n"

    # --- Agent Definitions ---
    if config["agents"]:
        for agent in config["agents"]:
            code += f"# Agent: {agent.get('name', 'agent')}\n"
            code += f"agent_{agent.get('name', 'agent')} = Agent(\n"
            code += f"    role={repr(agent.get('role', 'AI Agent'))},\n"
            code += f"    goal={repr(agent.get('goal', 'Assist with task'))},\n"
            code += f"    backstory={repr(agent.get('backstory', 'No backstory provided'))},\n"
            code += f"    verbose={agent.get('verbose', True)},\n"
            code += f"    allow_delegation={agent.get('allow_delegation', False)},\n"
            code += f"    tools={agent.get('tools', [])}\n"
            code += ")\n\n"
    else:
        code += "# No agents provided in configuration\n"
        code += "agent_default = Agent(role='Default Agent', goal='Execute fallback tasks', backstory='Auto-generated agent', verbose=True, allow_delegation=False, tools=[])\n\n"

    # --- Task Definitions ---
    if config["tasks"]:
        for task in config["tasks"]:
            agent_name = task.get("agent", (config["agents"][0]["name"] if config["agents"] else "default"))
            code += f"# Task: {task.get('name', 'task')}\n"
            code += f"task_{task.get('name', 'task')} = Task(\n"
            code += f"    description={repr(task.get('description', 'Perform task'))},\n"
            code += f"    agent=agent_{agent_name},\n"
            code += f"    expected_output={repr(task.get('expected_output', 'Task result'))}\n"
            code += ")\n\n"
    else:
        code += "# No tasks provided in configuration\n"
        code += "task_default = Task(description='Default placeholder task', agent=agent_default, expected_output='Default output')\n\n"

    # --- Crew Configuration ---
    code += "# Crew Configuration\n"
    agent_list = [f"agent_{a['name']}" for a in config["agents"]] if config["agents"] else ["agent_default"]
    task_list = [f"task_{t['name']}" for t in config["tasks"]] if config["tasks"] else ["task_default"]

    code += "crew = Crew(\n"
    code += f"    agents=[{', '.join(agent_list)}],\n"
    code += f"    tasks=[{', '.join(task_list)}],\n"
    code += "    verbose=True\n"
    code += ")\n\n"

    # --- Flow Definition ---
    code += "# Define CrewAI Flow\n"
    code += "class WorkflowFlow(Flow[AgentState]):\n"

    # Initial step
    code += "    @start()\n"
    code += "    def initial_input(self):\n"
    code += "        \"\"\"Start the workflow.\"\"\"\n"
    code += "        print('Starting workflow...')\n"
    first_task = config["tasks"][0]["name"] if config["tasks"] else "completed"
    code += f"        self.state.current_step = '{first_task}'\n"
    code += "        return self.state\n\n"

    # --- Task Execution Steps ---
    tasks = config.get("tasks", [])
    previous_step = "initial_input"

    if tasks:
        for i, task in enumerate(tasks):
            task_name = task["name"].replace("-", "_")
            code += f"    @listen('{previous_step}')\n"
            code += f"    def execute_{task_name}(self, state):\n"
            code += f"        \"\"\"Execute the {task['name']} task.\"\"\"\n"
            code += f"        print(f'Executing task: {task['name']}')\n"
            code += f"        result = crew.kickoff(tasks=[task_{task['name']}], inputs={{'query': self.state.query, 'previous_results': self.state.results}})\n"
            code += f"        self.state.results['{task['name']}'] = result\n"
            if i < len(tasks) - 1:
                next_task = tasks[i + 1]["name"]
                code += f"        self.state.current_step = '{next_task}'\n"
            else:
                code += "        self.state.current_step = 'completed'\n"
            code += "        return self.state\n\n"
            previous_step = f"execute_{task_name}"
    else:
        code += "    @listen('initial_input')\n"
        code += "    def execute_default(self, state):\n"
        code += "        print('No specific tasks defined; running default flow.')\n"
        code += "        self.state.results['default'] = 'Executed default workflow'\n"
        code += "        self.state.current_step = 'completed'\n"
        code += "        return self.state\n\n"

    # --- Aggregation Step ---
    code += f"    @listen('{previous_step}')\n"
    code += "    def aggregate_results(self, state):\n"
    code += "        print('Workflow completed, aggregating results...')\n"
    code += "        combined_result = ''\n"
    code += "        for task_name, result in state.results.items():\n"
    code += "            combined_result += f'\\n=== {task_name} ===\\n{result}'\n"
    code += "        return combined_result\n\n"

    # --- Run Function ---
    code += "# Run the flow\n"
    code += "def run_workflow(query: str):\n"
    code += "    flow = WorkflowFlow()\n"
    code += "    flow.state.query = query\n"
    code += "    result = flow.kickoff()\n"
    code += "    return result\n\n"

    # --- Visualization ---
    code += "# Visualize the flow\n"
    code += "def visualize_flow():\n"
    code += "    flow = WorkflowFlow()\n"
    code += "    flow.plot('workflow_flow')\n"
    code += "    print('Flow visualization saved to workflow_flow.html')\n\n"

    # --- Example Run ---
    code += "if __name__ == '__main__':\n"
    code += "    result = run_workflow('Your query here')\n"
    code += "    print(result)\n"

    return code
