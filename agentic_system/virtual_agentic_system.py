import re
from langgraph.graph import START, END

class VirtualAgenticSystem:
    """
    A virtual representation of an agentic system with nodes, tools, and edges.
    This class provides a way to define the structure of an agentic system
    without actually compiling it.
    """
    
    def __init__(self, system_name='Default'):
        self.system_name = system_name
        
        self.nodes = {}  # node_name -> description
        self.node_functions = {}  # node_name -> function implementation
        self.tools = {}  # tool_name -> description
        self.tool_functions = {}  # tool_name -> function implementation
        
        self.edges = []  # list of (source, target) tuples
        self.conditional_edges = {}  # source_node -> {condition: func, path_map: map}

        self.imports = [
            "from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict",
            "from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage",
            "from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls",
            "from langgraph.graph import StateGraph, START, END",
            "from langchain_core.tools import tool",
            "import os"
            ]
        
        self.state_attributes = {"messages": "List[Any]"}
        
        # For helper file functionality
        self.helper_code = ""
        
    def set_state_attributes(self, attrs):
        self.state_attributes = {"messages": "List[Any]"}
        for name, type_annotation in attrs.items():
            self.state_attributes[name] = type_annotation
        return True
        
    def add_imports(self, import_statement):
        if import_statement not in self.imports:
            self.imports.append(import_statement)

    def create_node(self, name, description, func, source_code=None):
        if func.__doc__ is None or func.__doc__.strip() == "":
            func.__doc__ = description
        if source_code:
            func._source_code = source_code
            
        self.nodes[name] = description
        self.node_functions[name] = func

        return True

    def create_tool(self, name, description, func, source_code=None):
        """Create a tool function that can be used by nodes."""
        if func.__doc__ is None or func.__doc__.strip() == "":
            raise ValueError("Tool function must contain a docstring.")
        if source_code:
            func._source_code = source_code
            
        self.tools[name] = description
        self.tool_functions[name] = func

        return True

    def create_edge(self, source, target):
        """Create a standard edge between nodes."""
        if source in ["START", "__start__"]:
            source = START
        if target in ["END", "__end__"]:
            target = END
            
        if source not in self.nodes and source != START:
            raise ValueError(f"Invalid source node: '{source}' does not exist")
            
        if target not in self.nodes and target != END:
            raise ValueError(f"Invalid target node: '{target}' does not exist")
        
        if any(edge_source == source for edge_source, _ in self.edges):
            raise ValueError(f"Source node '{source}' already has an outgoing edge. Parallel processing is disabled.")
        
        self.edges.append((source, target))
        return True

    def create_conditional_edge(self, source, condition, condition_code=None, path_map=None):
        """Create a conditional edge with a router function."""
        if source in ["START" "__start__", START, "END", "__end__", END]:
            raise ValueError(f"Invalid source node: Routers from endpoints are not allowed.")
        if source not in self.nodes:
            raise ValueError(f"Invalid source node: '{source}' does not exist")
        
        if condition_code:
            condition._source_code = condition_code
        
        edge_info = {"condition": condition}
        
        # if path_map is None:
        #     path_map = self._auto_path_map(condition_code)
        
        # for target in path_map.values():
        #     if target in ["END", "__end__"]:
        #         target = END
        #     if target not in self.nodes and target != END:
        #         raise ValueError(f"Invalid target node in path_map: '{target}' does not exist")
        
        if path_map:
            edge_info["path_map"] = path_map.copy()
        
        self.conditional_edges[source] = edge_info
        
        return True

    def delete_node(self, name):
        """Delete a node and all associated edges."""
        if name in ["START" "__start__", START, "END", "__end__", END]:
            raise ValueError(f"Deletion of endpoints is not allowed")
        if name not in self.nodes:
            return False
        
        del self.nodes[name]
        if name in self.node_functions:
            del self.node_functions[name]
        
        if name in self.conditional_edges:
            del self.conditional_edges[name]
        
        self.edges = [(s, t) for s, t in self.edges if s != name and t != name]
        
        return True
    
    def delete_tool(self, name):
        """Delete a tool."""
        if name not in self.tools:
            return False
        
        del self.tools[name]
        if name in self.tool_functions:
            del self.tool_functions[name]
        
        return True

    def delete_edge(self, source, target):
        """Delete a standard edge."""
        if source in ["START", "__start__"]:
            source = START
        if target in ["END", "__end__"]:
            target = END
            
        edge = (source, target)
        if edge in self.edges:
            self.edges.remove(edge)
            return True
        
        return False

    def delete_conditional_edge(self, source):
        """Delete a conditional edge."""
        if source in self.conditional_edges:
            del self.conditional_edges[source]
            return True
        
        return False
    
    def get_function(self, function_code):
        match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', function_code)
        if not match:
            return "Error: Could not identify function name in the provided code"
        
        function_name = match.group(1)
        completed_function_code = "\n".join(self.imports) + "\n" + function_code    
        local_vars = {}
        exec(completed_function_code, {"__builtins__": __builtins__}, local_vars)
        
        if function_name in local_vars and callable(local_vars[function_name]):
            new_function = local_vars[function_name]
            return new_function
        else:
            return f"Error: Function '{function_name}' not found after execution"
    
    def add_helper_code(self, new_code):
        """
        Adds helper code to the system. If a function or constant with the same name 
        already exists, it will be replaced.
        """
        # Get function names in new code
        function_pattern = re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        functions_in_new = set(function_pattern.findall(new_code))
        
        constant_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=', re.MULTILINE)
        constants_in_new = set(constant_pattern.findall(new_code))
        
        # If there's no existing code, just add the new code
        if not self.helper_code:
            self.helper_code = new_code
            return True
        
        # Process existing code line by line
        existing_lines = self.helper_code.splitlines()
        result_lines = []
        
        # Skip state for removing function bodies
        skip_function = None
        indentation_level = 0
        
        for line in existing_lines:
            # Check if this line starts a function definition
            func_match = function_pattern.match(line)
            if func_match:
                func_name = func_match.group(1)
                if func_name in functions_in_new:
                    # Skip this function since it will be replaced
                    skip_function = func_name
                    indentation_level = len(line) - len(line.lstrip())
                    continue
            
            # Handle skipping functions
            if skip_function:
                if line.strip() and len(line) - len(line.lstrip()) <= indentation_level:
                    # End of function reached
                    skip_function = None
                else:
                    # Still inside function body, skip
                    continue
            
            # Check if this line declares a constant
            const_match = constant_pattern.match(line)
            if const_match:
                const_name = const_match.group(1)
                if const_name in constants_in_new:
                    # Skip this constant since it will be replaced
                    continue
            
            # If we get here, keep the line
            result_lines.append(line)
        
        # Add the new code
        result_lines.append("")  # Add a blank line for separation
        result_lines.append(new_code)
        
        # Update helper code
        self.helper_code = "\n".join(result_lines)
        
        return True

    def _auto_path_map(self, function_code):
        string_pattern = r"['\"]([^'\"]*)['\"]"
        potential_nodes = set(re.findall(string_pattern, function_code))
    
        auto_path_map = {}
        for node_name in potential_nodes:
            if node_name in self.nodes:
                auto_path_map[node_name] = node_name
        if "END" in function_code:
            auto_path_map["END"] = END
    
        return auto_path_map