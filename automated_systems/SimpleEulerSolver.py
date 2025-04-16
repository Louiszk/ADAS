# SimpleEulerSolver System Configuration
# Total nodes: 2
# Total tools: 1

from langgraph.graph import StateGraph
from langchain_core.tools import tool
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls


def build_system():
    # Define state attributes for the system
    class AgentState(TypedDict):
        messages: List[Any]
        solution: str

    # Initialize graph with state
    graph = StateGraph(AgentState)

    # Tool definitions
    # ===== Tool Definitions =====
    tools = {}
    # Tool: python_executor
    # Description: Executes Python code and returns the output. Use this tool to run your solution to the problem.
    def python_executor(code: str) -> str:
       """
       Executes Python code and returns the output. Use this tool to run your solution to the problem.
    
       Args:
           code: The Python code to execute.
    
       Returns:
           The output of the code execution, or an error message if execution fails.
       """
       try:
           # Redirect stdout to capture the output
           import io
           import sys
           old_stdout = sys.stdout
           sys.stdout = io.StringIO()
    
           # Execute the code
           exec(code)
    
           # Get the output
           output = sys.stdout.getvalue()
    
           # Restore stdout
           sys.stdout = old_stdout
    
           return output.strip()
       except Exception as e:
           return f"Error: {str(e)}"
    

    tools["python_executor"] = tool(runnable=python_executor, name_or_callable="python_executor")

    # Register tools with LargeLanguageModel class
    LargeLanguageModel.register_available_tools(tools)
    # ===== Node Definitions =====
    # Node: solver_agent
    # Description: An agent that solves Project Euler problems using the python_executor tool.
    def solver_agent(state):
       llm = LargeLanguageModel(temperature=0.1)
       system_prompt = """You are an expert Python programmer and mathematician. 
       You are tasked with solving Project Euler problems. 
       You will receive the problem description as input. 
       You must write Python code to solve the problem and use the python_executor tool to execute the code. 
       Once you have the solution, extract the final answer and return it.
       Always start by outlining your approach to the problem.
       Then write the python code.
       Finally, verify the solution.
       The code MUST print the final answer to standard output.
       """
       messages = state.get("messages", [])
       full_messages = [SystemMessage(content=system_prompt)] + messages
       llm.bind_tools(["python_executor"])
       response = llm.invoke(full_messages)
       tool_messages, tool_results = execute_tool_calls(response)
    
       solution = None
       if "python_executor" in tool_results and tool_results["python_executor"]:
           output = tool_results["python_executor"][0]
           try:
               solution = str(int(output.strip()))  # Extract integer value
           except ValueError:
               solution = "Error: Could not extract numerical solution"
    
       new_state = {"messages": messages + [response] + tool_messages, "solution": solution}
       return new_state
    

    graph.add_node("solver_agent", solver_agent)

    # Node: extract_solution
    # Description: Extracts the final solution from the messages and sets it in the state.
    def extract_solution(state):
       messages = state.get("messages", [])
       solution = "No solution found"
       for message in reversed(messages):
           if isinstance(message, ToolMessage) and message.name == "python_executor":
               try:
                   solution = str(int(message.content.strip()))
                   break  # Found the solution, so exit the loop
               except ValueError:
                   solution = "Error: Could not extract numerical solution"
                   break
       new_state = {"messages": messages, "solution": solution}
       return new_state
    

    graph.add_node("extract_solution", extract_solution)

    # ===== Standard Edges =====
    graph.add_edge("solver_agent", "extract_solution")

    # ===== Entry/Exit Configuration =====
    graph.set_entry_point("solver_agent")
    graph.set_finish_point("extract_solution")

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, tools
