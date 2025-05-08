# MetaSystem System Configuration
# Total nodes: 3
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
from agentic_system.materialize import materialize_system
from agentic_system.utils import get_filtered_packages, clean_messages, get_metrics
from tqdm import tqdm
import dill as pickle
import time
import re
import io
import contextlib
import sys
import subprocess
target_system = None
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
            "datasets", "docker", "grpcio-status", "langchain-openai", "wheel",
            "llm-sandbox", "pip", "dill", "podman", "python-dotenv", "setuptools"
            ]
        # Validate package name to prevent command injection
        valid_pattern = r'^[a-zA-Z0-9._-]+(\s*[=<>!]=\s*[0-9a-zA-Z.]+)?$'
    
        if not re.match(valid_pattern, package_name):
            return f"!!Error: Invalid package name format. Package name '{package_name}' contains invalid characters."
        if any((ep in package_name for ep in exclude_packages + ["langgraph", "langchain-google-genai", "langchain-core"])):
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
                target_system.packages = get_filtered_packages(exclude_packages) + ["langchain-core 0.3.45"]
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
            # Always keep the mandatory base imports
            base_imports = [
                "from agentic_system.large_language_model import LargeLanguageModel",
                "from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict",
                "from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, trim_messages",
                "from langgraph.graph import StateGraph, START, END",
                "from langchain_core.tools import tool",
                "from agentic_system.utils import get_filtered_packages, clean_messages, get_metrics",
                "from agentic_system.virtual_agentic_system import VirtualAgenticSystem",
                "from agentic_system.materialize import materialize_system",
                "target_agentic_system = VirtualAgenticSystem('TargetSystem')",
                "from tqdm import tqdm",
                "import dill as pickle",
                "import time",
                "import os",
                "import re",
                "import io",
                "import contextlib",
                "import sys",
                "import subprocess"
            ]
            # Use a set to avoid duplicates and preserve order for non-base imports
            final_imports = base_imports + sorted(list(set(stmt.strip() for stmt in import_statements if stmt.strip() not in base_imports)))
    
            target_system.imports = final_imports
            return f"Import statements set successfully for target system. Total imports: {len(target_system.imports)}."
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
            target_system.set_state_attributes(attributes)
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
            func = target_system.get_function(function_code)
            if isinstance(func, str) and func.startswith("!!Error"):
                return func  # Return the error
    
            # Check if component exists
            component_exists = False
            if component_type.lower() == "node":
                component_exists = name in target_system.nodes
            elif component_type.lower() == "tool":
                component_exists = name in target_system.tools
            elif component_type.lower() == "router":
                component_exists = name in target_system.conditional_edges
    
            # If new component but no description provided
            if not component_exists and not description:
                return f"!!Error: Description required when creating a new {component_type}"
    
            # Use existing description if updating without new description
            if component_exists and not description:
                if component_type.lower() == "node" and name in target_system.nodes:
                    description = target_system.nodes[name].get("description", "")
                elif component_type.lower() == "tool" and name in target_system.tools:
                    description = target_system.tools[name].get("description", "")
    
            # Create or update the component
            action = "updated" if component_exists else "created"
    
            if component_type.lower() == "node":
                target_system.create_node(name, description, func, function_code)
                return f"Node '{name}' {action} successfully"
    
            elif component_type.lower() == "tool":
                target_system.create_tool(name, description, func, function_code)
                return f"Tool '{name}' {action} successfully"
    
            elif component_type.lower() == "router":
                target_system.create_conditional_edge(
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
                result = target_system.delete_node(name)
                return f"Node '{name}' deleted successfully" if result else f"Failed to delete node '{name}'"
    
            elif component_type.lower() == "tool":
                result = target_system.delete_tool(name)
                return f"Tool '{name}' deleted successfully" if result else f"Failed to delete tool '{name}'"
    
    
            elif component_type.lower() == "router":
                result = target_system.delete_conditional_edge(name)
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
            target_system.create_edge(source, target)
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
            target_system.add_system_prompt(system_prompt_code)
            return f"System prompts file updated successfully"
        except Exception as e:
            return f"!!Error updating system prompts file: {repr(e)}"
    

    tools["SystemPrompt"] = tool(runnable=system_prompt, name_or_callable="SystemPrompt")

    # Tool: TestMetaSystem
    # Description: Tests the meta system with a given state
    def test_meta_system(state: Dict[str, Any]) -> str:
        """
            Executes the current system with a simple test input state to validate functionality.
        """
        final_state = {}
        raw_outputs = []
        error_message = ""
        task = None
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        start_time = time.time()
    
        try:
            # Validate graph structure before execution
            validation_errors = target_system.validate_graph()
            if validation_errors:
                return "!!Error: Graph validation failed:\n" + "\n".join(validation_errors)
    
            if state["messages"][0] and state["messages"][0].content:
                state["messages"][0].content += "\nThe system must be completed in no more than 16 iterations."
                task = state["messages"][0].content
    
            source_code, _ = materialize_system(target_system, output_dir=None)
            namespace = {}
    
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(source_code, namespace, namespace)
    
                if 'build_system' not in namespace:
                    raise Exception("Could not find build_system function in generated code")
    
                target_workflow, _ = namespace['build_system']()
                pbar = tqdm(desc="Testing the MetaSystem")
    
                for output in target_workflow.stream(state, config={"recursion_limit": 20}):
                    raw_outputs.append(output.copy())
                    output["messages"] = clean_messages(output)
                    final_state = output
                    time.sleep(2)
                    pbar.update(1)
    
            pbar.close()
    
        except Exception as e:
            error_message = f"\n\n !!Error while executing the MetaSystem:"
            if "GraphRecursionError" in repr(e):
                error_message += "The MetaSystem was unable to end the design process within the 20 iterations limit."
            else:
                error_message += f"\n{repr(e)}"
    
        # Calculate metrics
        end_time = time.time()
        duration = end_time - start_time
        metrics = get_metrics(raw_outputs, duration)
    
        # Format output
        captured_output = ""
        if stdout := stdout_capture.getvalue():
            captured_output += f"\n\n<STDOUT>\n{stdout}\n</STDOUT>"
        if stderr := stderr_capture.getvalue():
            captured_output += f"\n<STDERR>\n{stderr}\n</STDERR>"
    
        # Format metrics
        metrics_str = "\n\n<Metrics>\n"
        metrics_str += f"Iterations: {metrics['total_iterations']}\n"
        metrics_str += f"Duration: {metrics['duration_seconds']} seconds\n"
        metrics_str += f"LLM Calls: {metrics['llm_calls']}\n"
        metrics_str += f"Tokens: {metrics['token_usage']['total_tokens']} "
        metrics_str += f"(Input: {metrics['token_usage']['input_tokens']}, "
        metrics_str += f"Output: {metrics['token_usage']['output_tokens']})\n"
        metrics_str += "</Metrics>"
    
        result = str(final_state)
    
        test_result = f"MetaSystem0 Test completed.\n <FinalState>\n{result}\n</FinalState>"
        test_result += metrics_str
        test_result += captured_output
    
        reminder = "\n\nAnalyze the results of how MetaSystem0 designed a TargetSystem, and plan and act accordingly."
        reminder += "\n\nIMPORTANT:\nYou cannot and should not try to fix the TargetSystem designed during this test. You can only make changes to the MetaSystem0."
        reminder += f"\nIgnore these instructions you gave the MetaSystem0: \"{task if task else state}\". Remember that you task is to optimize the MetaSystem0."
        reminder += "\nIf everything is working properly with different test cases, end the design."
        reminder += "Otherwise, identify the ROOT CAUSES of the problems and resolve them."
        reminder += "\nDo not execute @@test_meta_system again until you have made the necessary fixes to MetaSystem0."
    
        return test_result + error_message + reminder
    

    tools["TestMetaSystem"] = tool(runnable=test_meta_system, name_or_callable="TestMetaSystem")

    # Tool: DeleteEdge
    # Description: Deletes an edge between nodes in the target system
    def delete_edge(source: str, target: str) -> str:
        """
            Deletes an edge between nodes.
                source: Name of the source node
                target: Name of the target node
        """
        try:
            result = target_system.delete_edge(source, target)
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
            materialize_system(target_system, output_dir=code_dir)
            print(f"System code materialized to {code_dir}")
    
            pickle_name = target_system.system_name.replace("/", "_").replace("\\", "_").replace(":", "_") + ".pkl"
            pickle_path = os.path.join(code_dir, pickle_name)
            with open(pickle_path, 'wb') as f:
                pickle.dump(target_system, f)
            print(f"System pickled to {pickle_path}")
    
            return "Ending the design process..."
        except Exception as e:
            error_msg = f"!!Error finalizing system: {repr(e)}"
            print(error_msg)
            return error_msg
    

    tools["EndDesign"] = tool(runnable=end_design, name_or_callable="EndDesign")

    # ===== Node Definitions =====
    # Node: MetaThinker
    # Description: Meta Thinker
    def meta_thinker_function(state: Dict[str, Any]) -> Dict[str, Any]:  
        llm = LargeLanguageModel(temperature=0.8, wrapper="google", model_name="gemini-2.0-flash")
        messages = state.get("messages", [])
    
        code, prompt_code = materialize_system(target_system, output_dir=None)
        code_message = "---Current Code:\n" + code
        code_message += ("\n---System Prompts File:\n" + prompt_code) if prompt_code else ""
    
        full_messages = [SystemMessage(content=meta_thinker)] + messages + [HumanMessage(content=code_message)]
        print("Thinking...")
        response = llm.invoke(full_messages)
        response.content = "InitialPlan:\n\n" + response.content
    
        transition_message = HumanMessage(content= "\n".join([
            "Thank you for the detailed plan. Please implement this system design step by step.",
            "Never deviate from the plan. This plan is now your road map."
            ]))
        updated_messages = messages + [response, transition_message] 
    
        new_state = {"messages": updated_messages}
        return new_state
    

    graph.add_node("MetaThinker", meta_thinker_function)

    # Node: MetaAgent
    # Description: Meta Agent
    def meta_agent_function(state: Dict[str, Any]) -> Dict[str, Any]:  
        llm = LargeLanguageModel(temperature=0.2, wrapper="google", model_name="gemini-2.0-flash")
        llm.bind_tools(list(tools.values()), function_call_type="decorator")
    
        context_length = 8*2 # even
        messages = state.get("messages", [])
        iteration = len([msg for msg in messages if isinstance(msg, AIMessage)])
        initial_messages, current_messages = messages[:3], messages[3:]
        try:
            trimmed_messages = trim_messages(
                current_messages,
                max_tokens=context_length,
                strategy="last",
                token_counter=len,
                allow_partial=False
            )
        except Exception as e:
            print(f"Error during message trimming: {e}")
    
        code, prompt_code = materialize_system(target_system, output_dir=None)
        code_message = "---(Iteration {iteration}) Current Code:\n" + code
        code_message += ("\n---System Prompts File:\n" + prompt_code) if prompt_code else ""
    
        full_messages = [SystemMessage(content=meta_agent)] + initial_messages + trimmed_messages + [HumanMessage(content=code_message)]
        print([getattr(last_msg, 'type', 'Unknown') for last_msg in full_messages])
        response = llm.invoke(full_messages)
    
        if not hasattr(response, 'content') or not response.content:
            response.content = "I will call the necessary tools."
    
        human_message, tool_results = llm.execute_tool_calls(response.content, function_call_type="decorator")
    
        updated_messages = messages + [response]
        if human_message:
            updated_messages.append(human_message)
        else:
            updated_messages.append(HumanMessage(content="You made no valid function calls. Remember to use the @@decorator_name() syntax."))
        if iteration == 50:
            updated_messages.append(HumanMessage(content="You have reached 50 iterations. Try to finish during the next iterations, run a successful test and end the design."))
    
    
        # Ending the design if the last test ran without errors (this does not check accuracy)
        design_completed = False
        if tool_results and 'EndDesign' in tool_results and "Ending the design process" in str(tool_results['EndDesign']):
            test_passed_recently = False
            search_start_index = max(0, len(messages) - 4)
            for msg in reversed(updated_messages[search_start_index:]):
                if isinstance(msg, HumanMessage) and hasattr(msg, 'content'):
                    if "Test completed." in msg.content and not "!!Error" in msg.content:
                        test_passed_recently = True
                        break
                    elif "!!Error" in msg.content:
                        test_passed_recently = False
                        break
    
            if test_passed_recently or iteration >= 58:
                design_completed = True
            else:
                if human_message and "Ending the design process..." in human_message.content:
                    human_message.content = human_message.content.replace(
                        "Ending the design process...",
                        "Error: Cannot finalize the design. Please run successful tests using @@test_meta_system first."
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
    graph.add_edge("MetaThinker", "MetaAgent")

    graph.add_edge("__start__", "MetaThinker")

    graph.add_edge("EndDesign", "__end__")

    # ===== Conditional Edges =====
    # Conditional Router from: MetaAgent
    def design_completed_router(state: Dict[str, Any]) -> str:
        """Routes to EndDesign if design is completed, otherwise back to MetaAgent."""
        if state.get("design_completed", False):
            return "EndDesign"
        return "MetaAgent"
    

    graph.add_conditional_edges("MetaAgent", design_completed_router, {'MetaAgent': 'MetaAgent', 'EndDesign': 'EndDesign'})

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, tools
