# MetaSystem System Configuration
# Total nodes: 3
# Total tools: 11

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
from agentic_system.utils import get_filtered_packages, clean_messages
from tqdm import tqdm
import dill as pickle
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

    # Tool: AddComponent
    # Description: Adds a component (node, tool, or router) to the target system
    def add_component(component_type: str, name: str, description: str, function_code: str = None) -> str:
        """
            Creates a component in the target system.
                component_type: Type of component to create ('node', 'tool', or 'router')
                name: Name of the component
                description: Description of the component
                function_code: Python code defining the component's function (required for all component types)
        """
        try:
            if component_type.lower() not in ["node", "tool", "router"]:
                return f"!!Error: Invalid component type '{component_type}'. Must be 'node', 'tool', or 'router'."
    
            if not function_code:
                return f"!!Error: function_code is required for all component types"
    
            # Get the function implementation from the code
            func = target_system.get_function(function_code)
            if isinstance(func, str) and func.startswith("!!Error"):
                return func  # Return the error
    
            if component_type.lower() == "node":
                target_system.create_node(name, description, func, function_code)
                return f"Node '{name}' created successfully"
    
            elif component_type.lower() == "tool":
                target_system.create_tool(name, description, func, function_code)
                return f"Tool '{name}' created successfully"
    
            elif component_type.lower() == "router":
                target_system.create_conditional_edge(
                    source=name,
                    condition=func,
                    condition_code=function_code
                )
    
                return f"Conditional edge from '{name}' added successfully"
    
        except Exception as e:
            return f"!!Error creating {component_type}: {repr(e)}"
    

    tools["AddComponent"] = tool(runnable=add_component, name_or_callable="AddComponent")

    # Tool: EditComponent
    # Description: Edits a component's implementation
    def edit_component(component_type: str, name: str, new_function_code: str, new_description: Optional[str] = None) -> str:
        """
            Modifies an existing component's implementation.
                component_type: Type of component to edit ('node', 'tool', or 'router')
                name: Name of the component to edit
                new_function_code: New Python code for the component's function
                new_description: Optional new description for the component
        """
        try:
            if component_type.lower() not in ["node", "tool", "router"]:
                return f"!!Error: Invalid component type '{component_type}'. Must be 'node', 'tool', or 'router'."
    
            new_function = target_system.get_function(new_function_code)
            if isinstance(new_function, str) and new_function.startswith("Error"):
                return new_function  # Return the error
    
            if component_type.lower() == "node":
                if name not in target_system.nodes:
                    return f"!!Error: Node '{name}' not found"
    
                target_system.create_node(name, new_description, new_function, new_function_code)
                return f"Node '{name}' updated successfully"
    
            elif component_type.lower() == "tool":
                if name not in target_system.tools:
                    return f"!!Error: Tool '{name}' not found"
    
                target_system.create_tool(name, new_description, new_function, new_function_code)
                return f"Tool '{name}' updated successfully"
    
            elif component_type.lower() == "router":
                if name not in target_system.conditional_edges:
                    return f"!!Error: Router for node '{name}' not found"
    
                target_system.create_conditional_edge(
                    source=name,
                    condition=new_function,
                    condition_code=new_function_code
                )
    
                return f"Router for node '{name}' updated successfully"
    
        except Exception as e:
            return f"!!Error editing {component_type}: {repr(e)}"
    

    tools["EditComponent"] = tool(runnable=edit_component, name_or_callable="EditComponent")

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

    # Tool: TestSystem
    # Description: Tests the target system with a given state
    def test_system(state: Dict[str, Any]) -> str:
        """
            Executes the current system with a test input state to validate functionality.
                state: A python dictionary with state attributes e.g. {"messages": ["Test Input"], "attr2": [3, 5]}
        """
        all_outputs = []
        error_message = ""
        stdout_capture = io.StringIO()
    
        try:
            source_code, _ = materialize_system(target_system, output_dir=None)
            namespace = {}
    
            # Capture stdout during execution
            with contextlib.redirect_stdout(stdout_capture):
                exec(source_code, namespace, namespace)
    
                if 'build_system' not in namespace:
                    raise Exception("Could not find build_system function in generated code")
    
                target_workflow, _ = namespace['build_system']()
                pbar = tqdm(desc="Testing the System")
    
                for output in target_workflow.stream(state, config={"recursion_limit": 20}):
                    output["messages"] = clean_messages(output)
                    all_outputs.append(output)
                    pbar.update(1)
    
            pbar.close()
    
        except Exception as e:
            error_message = f"\n\n !!Error while testing the system:\n{repr(e)}"
    
        # Always capture stdout after try block
        captured_output = stdout_capture.getvalue()
    
        result = "\n".join([f"State {i}: " + str(out) for i, out in enumerate(all_outputs)]) if all_outputs else {}
    
        # Add captured stdout to the result
        test_result = f"Test completed.\n <SystemStates>\n{result}\n</SystemStates>"
        std_out = ""
        if captured_output:
            std_out = f"\n\n<Stdout>\n{captured_output}\n</Stdout>"
            test_result += std_out
        if error_message:
            return error_message + std_out
        else:
            return test_result
    

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
    
        transition_message = HumanMessage(content= "\n".join([
            "Thank you for the detailed plan. Please implement this system design step by step.",
            "Start by setting up the state attributes, imports and installing the necessary packages."
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
        response = llm.invoke(full_messages)
    
        if not hasattr(response, 'content') or not response.content:
            response.content = "I will call the necessary tools."
    
        human_message, tool_results = llm.execute_tool_calls(response.content, function_call_type="decorator")
    
        updated_messages = messages + [response]
        if human_message:
            updated_messages.append(human_message)
        else:
            updated_messages.append(HumanMessage(content="You made no valid function calls. Remember to use the @@decorator_name() syntax."))
    
    
        # Ending the design if the last test ran without errors (this does not check accuracy)
        design_completed = False
        if tool_results and 'EndDesign' in tool_results and "Ending the design process" in str(tool_results['EndDesign']):
            test_passed_recently = False
            search_start_index = max(0, len(messages) - 4)
            for msg in reversed(updated_messages[search_start_index:]):
                if isinstance(msg, HumanMessage) and hasattr(msg, 'content'):
                    if "Test completed." in msg.content and not "Error while testing the system" in msg.content:
                        test_passed_recently = True
                        break
                    elif "Error while testing the system" in msg.content:
                        test_passed_recently = False
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
