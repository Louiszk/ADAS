import re
import os
import sys
import dill as pickle
import inspect
from tqdm import tqdm
import subprocess
import io
import time
import traceback
import contextlib
from langgraph.graph import START, END
from typing import Dict, List, Any, Optional, Union
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, trim_messages
from agentic_system.large_language_model import LargeLanguageModel
from agentic_system.virtual_agentic_system import VirtualAgenticSystem
from agentic_system.materialize import materialize_system
from agentic_system.utils import get_filtered_packages, clean_messages, get_metrics
from systems.system_prompts import *

def create_meta_system():
    print(f"\n----- Creating Meta System -----\n")
    
    # Create a virtual agentic system
    meta_system = VirtualAgenticSystem("MetaSystem")
    target_system = None
    
    with open("systems/system_prompts.py", "r") as spf:
        system_prompts_content = spf.read()
    meta_system.add_system_prompt(system_prompts_content)
    meta_system.set_state_attributes({"design_completed": "bool"})
    
    # Add imports
    imports = [
        "from agentic_system.materialize import materialize_system",
        "from agentic_system.utils import get_filtered_packages, clean_messages, get_metrics",
        "from tqdm import tqdm",
        "import dill as pickle",
        "import traceback",
        "import time",
        "import re",
        "import io",
        "import contextlib",
        "import sys",
        "import subprocess",
        "target_system = None"
    ]
    
    for import_stmt in imports:
        meta_system.add_imports(import_stmt)
    
    # Create all tool functions
    
    # PipInstall tool
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
                target_system.packages = get_filtered_packages(exclude_packages) + ["langchain-core 0.3.59"]
                return f"Successfully installed {package_name}"
            else:
                return f"!!Error installing {package_name}:\n{process.stdout}"
    
        except Exception as e:
            return f"!!Error installing {package_name}: {str(e)}"
    
    meta_system.create_tool(
        "PipInstall",
        "Securely installs a Python package using pip",
        pip_install
    )

    # SetImports tool
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
                "import dill as pickle",
                "import traceback",
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
    
    meta_system.create_tool(
        "SetImports",
        "Sets the list of necessary import statements for the target system, replacing existing custom imports.",
        set_imports
    )

    # SetStateAttributes tool
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
    
    meta_system.create_tool(
        "SetStateAttributes",
        "Sets state attributes with type annotations for the target system",
        set_state_attributes
    )

    def upsert_component(component_type: str, name: str, function_code: str, description: Optional[str] = None) -> str:
        """
            Creates or updates a component in the target system.
                component_type: Type of the component ('node', 'tool', or 'router')
                name: Name of the component
                function_code: Python code defining the component's function
                description: Description of the component (required for new components)
        """
        try:
            if "greeting" in name.lower():
                return "!!Error: DO NOT MODIFY THE 'Greeting System'. Remember that your task is to optimize the MetaSystem0."

            if component_type.lower() not in ["node", "tool", "router"]:
                return f"!!Error: Invalid component type '{component_type}'. Must be 'node', 'tool', or 'router'."
            
            if not function_code:
                return "!!Error: You must provide the function implementation below the decorator, not as argument."
                
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
                if name in ["MetaAgent", "MetaThinker"]:
                    if "LargeLanguageModel(" not in function_code:
                        return f"!!Error: {name} must use a LargeLanguageModel."
                    if name == "MetaAgent" and "meta_agent" not in function_code:
                        return f"!!Error: {name} must include the meta_agent system prompt."
                    if name == "MetaThinker" and "meta_thinker" not in function_code:
                        return f"!!Error: {name} must include the meta_thinker system prompt."
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
        
    meta_system.create_tool(
        "UpsertComponent",
        "Creates or updates a component",
        upsert_component
    )

    # DeleteComponent tool
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
                critical_nodes = ["MetaThinker", "MetaAgent", "EndDesign"]
                if component_type.lower() == "node" and name in critical_nodes:
                    return f"!!Error: Deleting the critical node '{name}' is not allowed. MetaSystem0 requires this node to function."
                result = target_system.delete_node(name)
                return f"Node '{name}' deleted successfully" if result else f"Failed to delete node '{name}'"
                
            elif component_type.lower() == "tool":
                metasystem0_core_tool_names = ["PipInstall", "SetImports", "UpsertComponent", "DeleteComponent", "AddEdge", "DeleteEdge", "TestSystem", "EndDesign"]
                if component_type.lower() == "tool" and name in metasystem0_core_tool_names:
                    return f"!!Error: Deleting the core tool '{name}' from MetaSystem0 is not allowed. MetaSystem0 uses this tool to design other systems."
                result = target_system.delete_tool(name)
                return f"Tool '{name}' deleted successfully" if result else f"Failed to delete tool '{name}'"
                    
            elif component_type.lower() == "router":
                result = target_system.delete_conditional_edge(name)
                return f"Router for node '{name}' deleted successfully" if result else f"No router found for node '{name}'"
                
        except Exception as e:
            return f"!!Error deleting {component_type}: {repr(e)}"
    
    meta_system.create_tool(
        "DeleteComponent",
        "Deletes a component from the target system",
        delete_component
    )

    # AddEdge tool
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
    
    meta_system.create_tool(
        "AddEdge",
        "Adds an edge between nodes in the target system",
        add_edge
    )

    # SystemPrompt tool
    def system_prompt(system_prompt_code: str) -> str:
        """
            Adds a system prompt to the system_prompts file. Can be either a constant or a function that returns the system prompt.
            If a constant or function with the same name already exists in the file, it will be replaced.
                system_prompt: A system prompt that can be used to invoke large language models.
        """
        try:
            target_system.add_system_prompt(system_prompt_code, meta_meta_system = True)
            return f"System prompts file updated successfully"
        except Exception as e:
            return f"!!Error updating system prompts file: {repr(e)}"
    
    meta_system.create_tool(
        "SystemPrompt",
        "Adds or updates system prompts",
        system_prompt
    )

    # TestSystem tool
    def test_meta_system(state=None) -> str:
        """
            Executes the current system with a fixed test state to validate functionality.
        """
        state = {"messages": [HumanMessage(
            "Design a simple system that greets the user. It should include a 'GreetingNode' using an LLM."
            "\nThe system must be completed in no more than 16 iterations."
            )]}
        final_state = {}
        raw_outputs = []
        error_message = ""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        start_time = time.time()
    
        try:
            # Validate graph structure before execution
            validation_errors = target_system.validate_graph()
            if validation_errors:
                return "!!Error: Validation failed. The MetaSystem0 has structural flaws:\n" + "\n".join(validation_errors)
                
            source_code, system_prompts = materialize_system(target_system, output_dir=None)
            namespace = {}

            if system_prompts:
                exec(system_prompts, namespace, namespace)

            escaped_name_for_module = target_system.system_name.replace("/", "").replace("\\", "").replace(":", "")
            prompt_import_line_pattern = f"from automated_systems.{escaped_name_for_module}_system_prompts import *"
            
            patched_source_code = source_code.replace(prompt_import_line_pattern, "")
            
            # Capture stdout and stderr during execution
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(patched_source_code, namespace, namespace)
                
                if 'build_system' not in namespace:
                    raise Exception("Could not find build_system function in generated code")
                    
                target_workflow, _ = namespace['build_system']()
                pbar = tqdm(desc="Testing the MetaSystem")
                
                for output in target_workflow.stream(state, config={"recursion_limit": 20}):
                    raw_outputs.append(output.copy())
                    output["messages"] = clean_messages(output)
                    final_state = output
                    pbar.update(1)
        
            pbar.close()
            
        except Exception as e:
            error_message = f"\n\n !!Error while executing the MetaSystem:"
            if "GraphRecursionError" in repr(e):
                error_message += "The MetaSystem was unable to end the design process within the 20 iterations limit."
            else:
                error_message += f"\n{traceback.format_exc(chain=False)}"
                
        # Calculate metrics
        end_time = time.time()
        duration = end_time - start_time
        metrics = get_metrics(raw_outputs, duration)
        
        # Format output
        captured_output = ""
        stdout = stdout_capture.getvalue() or ""
        stderr = stderr_capture.getvalue() or ""
        captured_output = f"\n\n<STDOUT+STDERR>\n{stdout}\n{stderr}\n</STDOUT+STDERR>"
        
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
    
        test_result = f"MetaSystem0 Test completed.\n<FinalState>\n{result}\n</FinalState>"
        test_result += metrics_str
        test_result += captured_output
        
        return test_result + error_message + test_reminder
    
    meta_system.create_tool(
        "TestMetaSystem",
        "Tests the meta system with a given state",
        test_meta_system
    )

    # DeleteEdge tool
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
    
    meta_system.create_tool(
        "DeleteEdge",
        "Deletes an edge between nodes in the target system",
        delete_edge
    )

    # EndDesign tool
    def end_design() -> str:
        """
            Finalizes the system design process.
        """
        try:  
            code_dir = "sandbox/workspace/automated_systems"
            materialize_system(target_system, output_dir=code_dir)
            print(f"System code materialized to {code_dir}")
    
            return "Ending the design process..."
        except Exception as e:
            error_msg = f"!!Error finalizing system: {repr(e)}"
            print(error_msg)
            return error_msg
    
    meta_system.create_tool(
        "EndDesign",
        "Finalizes the system design process",
        end_design
    )

    # Create node functions
    
    # MetaThinker node
    def meta_thinker_function(state: Dict[str, Any]) -> Dict[str, Any]:  
        llm = LargeLanguageModel(temperature=0.8, wrapper="google", model_name="gemini-2.5-pro-preview-05-06")
        messages = state.get("messages", [])

        code, prompt_code = materialize_system(target_system, output_dir=None)
        code_message = f"--- Current Code of MetaSystem0 ---\n```\n{code}\n```"
        code_message += f"\n\n--- Current System Prompts File of MetaSystem0 ---\n```\n{prompt_code}\n```" if prompt_code else ""
        
        # Read solution archive if it exists
        solution_archive_content = ""
        archive_path = "/sandbox/workspace/systems/solution_archive.txt"
        if os.path.exists(archive_path):
            try:
                with open(archive_path, 'r') as f:
                    archive_content = f.read().strip()
                    if archive_content:
                        solution_archive_content = "\n\n--- Previous Solution Concepts ---\n"
                        solution_archive_content += "You have previously generated the following solution concepts in earlier attempts.\n"
                        solution_archive_content += "You can get inspired by the concepts, but try to come up with NOVEL ideas or alternatives this time, while still addressing the core optimization goals.\n"
                        solution_archive_content += archive_content
                        print(f"Loaded solution archive with {len(archive_content)} characters")
            except Exception as e:
                print(f"Warning: Could not read solution archive: {e}")
        
        # Combine all content for the prompt
        full_code_message = code_message + solution_archive_content
        
        full_messages = [SystemMessage(content=meta_thinker)] + messages + [HumanMessage(content=full_code_message)]
        print("Thinking...")
        response = llm.invoke(full_messages)
        response.content = "[Iteration 0]\n\n# Roadmap\n" + response.content

        # Extract and append Solution Concepts to archive
        try:
            import re
            pattern = r'## Solution Concepts\s*\n(.*?)(?=\n##|\Z)'
            match = re.search(pattern, response.content, re.DOTALL)
            
            if match:
                concepts = match.group(1).strip()
                if concepts:
                    with open(archive_path, 'a') as f:
                        system_name = target_system.system_name if target_system else "X"
                        attempt = system_name.replace("MetaSystem", "")
                        f.write('\n-----------------------------------------\n')
                        f.write(f'Solutions from Attempt {attempt}:\n')
                        f.write(concepts)
                        f.write('\n-----------------------------------------\n')
                    print(f"Appended new solution concepts to archive")
                else:
                    print("Solution Concepts section was empty")
            else:
                print("Could not find Solution Concepts section in response")
        except Exception as e:
            print(f"Warning: Could not extract/save solution concepts: {e}")

        transition_message = HumanMessage(content= "\n".join([
            "Thank you for the detailed plan. Please implement this system design step by step.",
            "Try not to deviate from the plan. This plan is now your roadmap."
            ]))
        updated_messages = messages + [response, transition_message] 

        new_state = {"messages": updated_messages}
        return new_state
    
    meta_system.create_node(
        "MetaThinker", 
        "Meta Thinker",
        meta_thinker_function
    )

    tools = {}
    # MetaAgent node
    def meta_agent_function(state: Dict[str, Any]) -> Dict[str, Any]:  
        llm = LargeLanguageModel(temperature=0.2, wrapper="google", model_name="gemini-2.5-flash-preview-05-20")
        llm.bind_tools(list(tools.values()), function_call_type="decorator")

        context_length = 16
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
        code_message = f"**You are now in Iteration {iteration}**\n--- Current Code of MetaSystem0 ---\n```\n{code}\n```"
        code_message += f"\n\n--- Current System Prompts File of MetaSystem0 ---\n```\n{prompt_code}\n```" if prompt_code else ""
    
        full_messages = [SystemMessage(content=meta_agent)] + initial_messages + trimmed_messages + [HumanMessage(content=code_message)]
        print([getattr(last_msg, 'type', 'Unknown') for last_msg in full_messages])
        response = llm.invoke(full_messages)
        response_content = response.content

        if not hasattr(response, 'content') or not response_content:
            response_content = "I will execute the necessary decorators."
        iteration_info = f"[Iteration {iteration}]"
        response.content = f"{iteration_info}\n\n" + response_content if iteration_info not in response_content else response_content
        
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
                        if not "!!Error" in str(msg.content).split("/STDOUT")[-1]:
                            test_passed_recently = True
                        break

            if (test_passed_recently and iteration > 8) or iteration >= 58:
                design_completed = True
            else:
                if iteration <=8:
                    replace_text = f"Error: Cannot finalize the design yet. You have only used {iteration}/60 iterations. Please try to optimize further."
                else: 
                    replace_text = "Error: Cannot finalize the design. Please run successful tests using @@test_meta_system first."
                if human_message and "Ending the design process..." in human_message.content:
                    human_message.content = human_message.content.replace("Ending the design process...", replace_text)
    
        new_state = {"messages": updated_messages, "design_completed": design_completed}
        return new_state
    
    meta_system.create_node(
        "MetaAgent", 
        "Meta Agent",
        meta_agent_function
    )

    # EndDesign node
    def end_design_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return state
    
    meta_system.create_node(
        "EndDesign", 
        "Terminal node for workflow completion",
        end_design_node
    )

    # Create edges
    meta_system.create_edge("MetaThinker", "MetaAgent")
    
    # Add conditional edge
    def design_completed_router(state: Dict[str, Any]) -> str:
        """Routes to EndDesign if design is completed, otherwise back to MetaAgent."""
        if state.get("design_completed", False):
            return "EndDesign"
        return "MetaAgent"
    
    meta_system.create_conditional_edge(
        source="MetaAgent", 
        condition=design_completed_router,
        condition_code=inspect.getsource(design_completed_router),
        path_map={'MetaAgent': 'MetaAgent', 'EndDesign': 'EndDesign'}
    )
    
    # Set entry and exit points
    meta_system.create_edge(START, "MetaThinker")
    meta_system.create_edge("EndDesign", END)
    
    # Materialize the system
    materialize_system(meta_system)
    print("----- Materialized Meta System -----")

if __name__ == "__main__":
    create_meta_system()
