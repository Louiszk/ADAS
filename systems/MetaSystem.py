# MetaSystem System Configuration
# Total nodes: 2
# Total tools: 10

# Already installed packages
# langchain-core 0.3.45
# langgraph 0.3.5

from agentic_system.large_language_model import LargeLanguageModel
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, trim_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
import os
from agentic_system.utils import get_filtered_packages, clean_messages, get_metrics
from agentic_system.virtual_agentic_system import VirtualAgenticSystem
from agentic_system.materialize import materialize_system
target_agentic_system = None
import dill as pickle
import traceback
import time
import re
import io
import contextlib
import sys
import subprocess
from systems.MetaSystem_system_prompts import *

# ===== Agentic System =====
def build_system():
    # Define state attributes for the system
    class AgentState(TypedDict):
        messages: List[Any]
        design_completed: bool

    # Initialize graph with state
    graph = StateGraph(AgentState)

    # Tool definitions
    # ===== Tool Definitions =====
    tools = {}
    # Tool: PipInstall
    # Description: Securely installs a Python package using pip
    def pip_install(package_name: str) -> str:
        """
            Securely installs a Python package using pip.
                package_name: Name of the package to install e.g. "langgraph==0.3.5"
        """
    
        exclude_packages = [
            "datasets", "docker", "grpcio-status", "langchain-google-genai", "langchain-openai", "wheel",
            "llm-sandbox", "pip", "dill", "podman", "python-dotenv", "setuptools"
            ]
        # Validate package name to prevent command injection
        valid_pattern = r'^[a-zA-Z0-9._-]+(\s*[=<>!]=\s*[0-9a-zA-Z.]+)?$'
    
        if not re.match(valid_pattern, package_name):
            return f"!!Error: Invalid package name format. Package name '{package_name}' contains invalid characters."
        if any((ep in package_name for ep in exclude_packages + ["langgraph", "langchain-core"])):
            return f"{package_name} is already installed."
    
        try:
            process = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                shell=False
            )
    
            if process.returncode == 0:
                # get_filtered_packages will execute pip list --not-required
                target_agentic_system.packages = get_filtered_packages(exclude_packages) + ["langchain-core 0.3.45"]
                return f"Successfully installed {package_name}"
            else:
                return f"!!Error installing {package_name}:\n{process.stdout}"
    
        except Exception as e:
            return f"!!Error installing {package_name}: {str(e)}"
    

    tools["PipInstall"] = tool(runnable=pip_install, name_or_callable="PipInstall")

    # Tool: SetImports
    # Description: Sets the list of necessary import statements for the target system, replacing existing custom imports.
    def set_imports(import_statements: List[str]) -> str:
        """
            Sets the list of import statements for the target system. This replaces any existing imports.
                import_statements: A list of strings, where each string is a complete import statement (e.g., ['import os', 'from typing import List']).
        """
    
        try:
            # Basic validation for each statement
            for stmt in import_statements:
                if not isinstance(stmt, str) or not (stmt.startswith("import ") or stmt.startswith("from ")):
                    return f"!!Error: Invalid import statement format: '{stmt}'. Must start with 'import' or 'from'."
    
            # Always keep the mandatory base imports
            base_imports = [
                "from agentic_system.large_language_model import LargeLanguageModel",
                "from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict",
                "from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage",
                "from langgraph.graph import StateGraph, START, END",
                "from langchain_core.tools import tool",
                "import os"
            ]
            # Use a set to avoid duplicates and preserve order for non-base imports
            final_imports = base_imports + sorted(list(set(stmt.strip() for stmt in import_statements if stmt.strip() not in base_imports)))
    
            target_agentic_system.imports = final_imports
            return f"Import statements set successfully for target system. Total imports: {len(target_agentic_system.imports)}."
        except Exception as e:
            return f"!!Error setting imports: {repr(e)}"
    

    tools["SetImports"] = tool(runnable=set_imports, name_or_callable="SetImports")

    # Tool: SetStateAttributes
    # Description: Sets state attributes with type annotations for the target system
    def set_state_attributes(attributes: Dict[str, str]) -> str:
        """
            Defines state attributes accessible throughout the system. Only defines the type annotations, not the values.
                attributes: A python dictionary mapping attribute names to string type annotations. 
                {"messages": "List[Any]"} is the default and will be set automatically.
        """
        try:
            target_agentic_system.set_state_attributes(attributes)
            return f"State attributes set successfully: {attributes}"
        except Exception as e:
            return f"!!Error setting state attributes: {repr(e)}"
    

    tools["SetStateAttributes"] = tool(runnable=set_state_attributes, name_or_callable="SetStateAttributes")

    # Tool: UpsertComponent
    # Description: Creates or updates a component
    def upsert_component(component_type: str, name: str, function_code: str, description: Optional[str] = None) -> str:
        """
            Creates or updates a component in the target system.
                component_type: Type of the component ('node', 'tool', or 'router')
                name: Name of the component
                function_code: Python code defining the component's function
                description: Description of the component (required for new components)
        """
        try:
            if component_type.lower() not in ["node", "tool", "router"]:
                return f"!!Error: Invalid component type '{component_type}'. Must be 'node', 'tool', or 'router'."
    
            if not function_code:
                return f"!!Error: You must provide the function implementation below the decorator, not as argument."
    
            # Get the function implementation from the code
            func = target_agentic_system.get_function(function_code)
            if isinstance(func, str) and func.startswith("!!Error"):
                return func  # Return the error
    
            # Check if component exists
            component_exists = False
            if component_type.lower() == "node":
                component_exists = name in target_agentic_system.nodes
            elif component_type.lower() == "tool":
                component_exists = name in target_agentic_system.tools
            elif component_type.lower() == "router":
                component_exists = name in target_agentic_system.conditional_edges
    
            # If new component but no description provided
            if not component_exists and not description:
                return f"!!Error: Description required when creating a new {component_type}"
    
            # Use existing description if updating without new description
            if component_exists and not description:
                if component_type.lower() == "node" and name in target_agentic_system.nodes:
                    description = target_agentic_system.nodes[name].get("description", "")
                elif component_type.lower() == "tool" and name in target_agentic_system.tools:
                    description = target_agentic_system.tools[name].get("description", "")
    
            # Create or update the component
            action = "updated" if component_exists else "created"
    
            if component_type.lower() == "node":
                target_agentic_system.create_node(name, description, func, function_code)
                return f"Node '{name}' {action} successfully"
    
            elif component_type.lower() == "tool":
                target_agentic_system.create_tool(name, description, func, function_code)
                return f"Tool '{name}' {action} successfully"
    
            elif component_type.lower() == "router":
                target_agentic_system.create_conditional_edge(
                    source=name,
                    condition=func,
                    condition_code=function_code
                )
    
                return f"Conditional edge from '{name}' {action} successfully"
    
        except Exception as e:
            return f"!!Error with {component_type}: {repr(e)}"
    

    tools["UpsertComponent"] = tool(runnable=upsert_component, name_or_callable="UpsertComponent")

    # Tool: DeleteComponent
    # Description: Deletes a component from the target system
    def delete_component(component_type: str, name: str) -> str:
        """
            Deletes a component from the target system.
                component_type: Type of component to delete ('node', 'tool', or 'router')
                name: Name of the component or source node for routers
        """
        try:
            if component_type.lower() not in ["node", "tool", "router"]:
                return f"!!Error: Invalid component type '{component_type}'. Must be 'node', 'tool', or 'router'."
    
            if component_type.lower() == "node":
                result = target_agentic_system.delete_node(name)
                return f"Node '{name}' deleted successfully" if result else f"Failed to delete node '{name}'"
    
            elif component_type.lower() == "tool":
                result = target_agentic_system.delete_tool(name)
                return f"Tool '{name}' deleted successfully" if result else f"Failed to delete tool '{name}'"
    
    
            elif component_type.lower() == "router":
                result = target_agentic_system.delete_conditional_edge(name)
                return f"Router for node '{name}' deleted successfully" if result else f"No router found for node '{name}'"
    
        except Exception as e:
            return f"!!Error deleting {component_type}: {repr(e)}"
    

    tools["DeleteComponent"] = tool(runnable=delete_component, name_or_callable="DeleteComponent")

    # Tool: AddEdge
    # Description: Adds an edge between nodes in the target system
    def add_edge(source: str, target: str) -> str:
        """
            Adds an edge between nodes in the target system.
                source: Name of the source node
                target: Name of the target node
        """
        try:
            target_agentic_system.create_edge(source, target)
            return f"Edge from '{source}' to '{target}' added successfully"
        except Exception as e:
            return f"!!Error adding edge: {repr(e)}"
    

    tools["AddEdge"] = tool(runnable=add_edge, name_or_callable="AddEdge")

    # Tool: SystemPrompt
    # Description: Adds or updates system prompts
    def system_prompt(system_prompt_code: str) -> str:
        """
            Adds a system prompt to the system_prompts file. Can be either a constant or a function that returns the system prompt.
            If a constant or function with the same name already exists in the file, it will be replaced.
                system_prompt: A system prompt that can be used to invoke large language models.
        """
        try:
            target_agentic_system.add_system_prompt(system_prompt_code)
            return f"System prompts file updated successfully"
        except Exception as e:
            return f"!!Error updating system prompts file: {repr(e)}"
    

    tools["SystemPrompt"] = tool(runnable=system_prompt, name_or_callable="SystemPrompt")

    # Tool: TestSystem
    # Description: Tests the target system with a given state
    def test_system(state: Dict[str, Any]) -> str:
        """
            Executes the current system with a test input state to validate functionality.
                state: A python dictionary with state attributes e.g. {"messages": ["Test Input"], "attr2": [3, 5]}
        """
        final_state = {}
        raw_outputs = []
        error_message = ""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        start_time = time.time()
    
        try:
            # Validate graph structure before execution
            validation_errors = target_agentic_system.validate_graph()
            if validation_errors:
                return "!!Error: Validation failed. The TargetSystem has structural flaws::\n" + "\n".join(validation_errors)
    
            source_code, system_prompts = materialize_system(target_agentic_system, output_dir=None)
            namespace = {}
    
            if system_prompts:
                exec(system_prompts, namespace, namespace)
    
            escaped_name_for_module = target_agentic_system.system_name.replace("/", "").replace("\\", "").replace(":", "")
            prompt_import_line_pattern = f"from automated_systems.{escaped_name_for_module}_system_prompts import *"
    
            patched_source_code = source_code.replace(prompt_import_line_pattern, "")
    
            # Capture stdout and stderr during execution
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(patched_source_code, namespace, namespace)
    
                if 'build_system' not in namespace:
                    raise Exception("Could not find build_system function in generated code")
    
                target_workflow, _ = namespace['build_system']()
    
                for output in target_workflow.stream(state, config={"recursion_limit": 20}):
                    raw_outputs.append(output.copy())  # Store raw output for metrics
                    if "messages" in output:
                        output["messages"] = clean_messages(output) # Only get content and tool_calls from messages
                    final_state = output # Only get the final state that contains all the messages
    
        except Exception as e:
            error_message = f"\n\n !!Error while testing the system:\n{traceback.format_exc(chain=False)}"
    
        # Calculate execution time and metrics
        end_time = time.time()
        duration = end_time - start_time
        metrics = get_metrics(raw_outputs, duration)
    
        # Capture console output
        captured_output = ""
        stdout = stdout_capture.getvalue() or ""
        stderr = stderr_capture.getvalue() or ""
        captured_output = f"\n\n<STDOUT+STDERR>\n{stdout}\n{stderr}\n</STDOUT+STDERR>"
    
        # Format metrics for display
        metrics_str = "\n\n<Metrics>\n"
        metrics_str += f"Iterations: {metrics['total_iterations']}\n"
        metrics_str += f"Duration: {metrics['duration_seconds']} seconds\n"
        metrics_str += f"LLM Calls: {metrics['llm_calls']}\n"
        metrics_str += f"Tokens: {metrics['token_usage']['total_tokens']} "
        metrics_str += f"(Input: {metrics['token_usage']['input_tokens']}, "
        metrics_str += f"Output: {metrics['token_usage']['output_tokens']})\n"
        metrics_str += "</Metrics>"
    
        result = str(final_state)
    
        # Construct the final result with metrics
        test_result = f"Test completed.\n<FinalState>\n{result}\n</FinalState>"
        test_result += metrics_str
        test_result += captured_output
    
        reminder = "\n\nAnalyze these results of the TargetSystem, and plan and act accordingly."
        reminder += "\nIf everything works properly with different test cases, or if you have reached the iteration limit, end the design."
        reminder += "\nOtherwise, identify the ROOT CAUSES of the problems and resolve them."
    
        return test_result + error_message + reminder
    

    tools["TestSystem"] = tool(runnable=test_system, name_or_callable="TestSystem")

    # Tool: DeleteEdge
    # Description: Deletes an edge between nodes in the target system
    def delete_edge(source: str, target: str) -> str:
        """
            Deletes an edge between nodes.
                source: Name of the source node
                target: Name of the target node
        """
        try:
            result = target_agentic_system.delete_edge(source, target)
            return f"Edge from '{source}' to '{target}' deleted successfully" if result else f"No such edge from '{source}' to '{target}'"
        except Exception as e:
            return f"!!Error deleting edge: {repr(e)}"
    

    tools["DeleteEdge"] = tool(runnable=delete_edge, name_or_callable="DeleteEdge")

    # Tool: EndDesign
    # Description: Finalizes the system design process
    def end_design() -> str:
        """
            Finalizes the system design process.
        """
        try:  
            code_dir = "sandbox/workspace/automated_systems"
            materialize_system(target_agentic_system, output_dir=code_dir)
            print(f"System code materialized to {code_dir}")
    
            return "Ending the design process..."
        except Exception as e:
            error_msg = f"!!Error finalizing system: {repr(e)}"
            print(error_msg)
            return error_msg
    

    tools["EndDesign"] = tool(runnable=end_design, name_or_callable="EndDesign")

    # ===== Node Definitions =====
    # Node: MetaAgent
    # Description: Meta Agent
    def meta_agent_function(state: Dict[str, Any]) -> Dict[str, Any]:  
        llm = LargeLanguageModel(temperature=0.2, wrapper="google", model_name="gemini-2.0-flash")
        llm.bind_tools(list(tools.values()), function_call_type="decorator")
    
        context_length = 16
        messages = state.get("messages", [])
        # Filter out empty messages to avoid 'GenerateContentRequest.contents: contents is not specified'
        messages = [msg for msg in messages if hasattr(msg, 'content') and msg.content]
        iteration = len([msg for msg in messages if isinstance(msg, AIMessage)])
        initial_messages, current_messages = messages[:1], messages[1:]
        try:
            trimmed_messages = trim_messages(
                current_messages,
                max_tokens=context_length,
                strategy="last",
                token_counter=len, # correctly counts messages
                allow_partial=False
            )
        except Exception as e:
            print(f"Error during message trimming: {e}")
    
        code, prompt_code = materialize_system(target_agentic_system, output_dir=None)
        code_message = f"**You are now in Iteration {iteration}**\n**Here is the Current Code:**\n" + code
        code_message += ("\n\n**System Prompts File:**\n" + prompt_code) if prompt_code else ""
    
        full_messages = [SystemMessage(content=meta_agent)] + initial_messages + trimmed_messages + [HumanMessage(content=code_message)]
        response = llm.invoke(full_messages)
    
        if not hasattr(response, 'content') or not response.content:
            response.content = "I will execute the necessary decorators."
        response.content = f"[Iteration {iteration}]\n\n" + response.content
    
        # This will parse the decorators from the response and correctly execute the functions
        human_message, tool_results = llm.execute_tool_calls(response.content, function_call_type="decorator")
    
        if not human_message:
            human_message = HumanMessage(content="\n".join([
                "In this previous response, you executed no decorators.",
                "Do not repeat yourself indefinitely.",
                "Remember to always structure your output like this:",
                "## Current System Analysis\n## Reasoning\n## Actions",
                "Use this syntax to execute decorators:\n```\n@@decorator_name()\n```"
                ]))
        if iteration == 50:
            human_message.content += "\n\nYou have reached 50 of 60 iterations. Try to finish during the next iterations, run a successful test and end the design."
    
        updated_messages = messages + [response]
        human_message.content = f"[Iteration {iteration}]\n\n" + human_message.content
        updated_messages.append(human_message)
    
        # Ending the design if the last test ran without errors
        design_completed = False
        if tool_results and 'EndDesign' in tool_results and "Ending the design process" in str(tool_results['EndDesign']):
            test_passed_recently = False
            search_start_index = max(0, len(messages) - 4)
            for msg in reversed(updated_messages[search_start_index:]):
                if isinstance(msg, HumanMessage) and hasattr(msg, 'content'):
                    if "Test completed." in msg.content:
                        if not "!!Error" in msg.content:
                            test_passed_recently = True
                        break
    
            if test_passed_recently or iteration >= 58:
                design_completed = True
            else:
                if human_message and "Ending the design process..." in human_message.content:
                    human_message.content = human_message.content.replace(
                        "Ending the design process...",
                        "Error: Cannot finalize the design. Please run successful tests using @@test_system first."
                    )
    
        new_state = {"messages": updated_messages, "design_completed": design_completed}
        return new_state
    

    graph.add_node("MetaAgent", meta_agent_function)

    # Node: EndDesign
    # Description: Terminal node for workflow completion
    def end_design_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return state
    

    graph.add_node("EndDesign", end_design_node)

    # ===== Standard Edges =====
    graph.add_edge(START, "MetaAgent")

    graph.add_edge("EndDesign", END)

    # ===== Conditional Edges =====
    # Conditional Router from: MetaAgent
    def design_completed_router(state: Dict[str, Any]) -> str:
        """Routes to EndDesign if design is completed, otherwise back to MetaAgent."""
        if state.get("design_completed", False):
            return "EndDesign"
        return "MetaAgent"
    

    graph.add_conditional_edges("MetaAgent", design_completed_router)

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, tools
