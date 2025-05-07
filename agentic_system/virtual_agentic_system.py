import re
import ast
import textwrap
import collections
import inspect
from langgraph.graph import START, END

class ReturnValueExtractor(ast.NodeVisitor):
    """AST visitor that extracts string literals and END constants from return statements."""
    def __init__(self):
        self.returned_values = set()

    def visit_Return(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            self.returned_values.add(node.value.value)
        elif isinstance(node.value, ast.Name) and node.value.id == "END":
            self.returned_values.add(END)

def _extract_top_level_names(ast_module: ast.Module) -> set[str]:
    names = set()
    for node in ast_module.body:
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names

class RemoveDefinitionsTransformer(ast.NodeTransformer):
    def __init__(self, names_to_remove: set[str]):
        self.names_to_remove = names_to_remove
        super().__init__()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        if node.name in self.names_to_remove:
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self.names_to_remove:
                return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if isinstance(node.target, ast.Name) and node.target.id in self.names_to_remove:
            return None
        return self.generic_visit(node)

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
        
        self.packages = ["langchain-core 0.3.45", "langgraph 0.3.5"]
        self.imports = [
            "from agentic_system.large_language_model import LargeLanguageModel",
            "from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict",
            "from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, trim_messages",
            "from langgraph.graph import StateGraph, START, END",
            "from langchain_core.tools import tool",
            "import os"
            ]
        
        self.state_attributes = {"messages": "List[Any]"}
        
        self.system_prompt_code = ""
        
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
        else:
            func._source_code = inspect.getsource(func)
            
        self.nodes[name] = description
        self.node_functions[name] = func

        return True

    def create_tool(self, name, description, func, source_code=None):
        """Create a tool function that can be used by nodes."""
        if func.__doc__ is None or func.__doc__.strip() == "":
            raise ValueError("Tool function must contain a docstring.")
        if source_code:
            func._source_code = source_code
        else:
            func._source_code = inspect.getsource(func)
            
        self.tools[name] = description
        self.tool_functions[name] = func

        return True
    
    def _infer_path_map(self, function_code: str) -> dict:
        """Infer possible return values from conditional edge function using AST parsing."""
        try:
            dedented_code = textwrap.dedent(function_code)
            tree = ast.parse(dedented_code)
            
            # Get function node if within a module
            func_node = tree
            if isinstance(tree, ast.Module) and tree.body and isinstance(tree.body[0], ast.FunctionDef):
                func_node = tree.body[0]

            # Extract return values
            extractor = ReturnValueExtractor()
            extractor.visit(func_node)
            
            # Build path map
            path_map = {}
            for val in extractor.returned_values:
                if val == END:
                    path_map["END"] = END
                elif isinstance(val, str):
                    path_map[val] = val
            
            return path_map
        except SyntaxError:
            return {}
        except Exception:
            return {}

    def create_edge(self, source, target):
        """Create a standard edge between nodes."""
        if source in ["START", "__start__"]:
            source = START
        if target in ["END", "__end__"]:
            target = END
            
        # Validate source and target nodes
        if source != START and source not in self.nodes:
            raise ValueError(f"Invalid source node: '{source}' does not exist")
            
        if target != END and target not in self.nodes:
            raise ValueError(f"Invalid target node: '{target}' does not exist")
        
        # Check for existing standard edge from source
        if any(edge_source == source for edge_source, _ in self.edges):
            raise ValueError(f"Source node '{source}' already has an outgoing standard edge.")
        
        # Check for conflicting conditional edge
        if source in self.conditional_edges:
            raise ValueError(f"Source node '{source}' is already a conditional edge source.")

        self.edges.append((source, target))
        return True

    def create_conditional_edge(self, source, condition, condition_code=None, path_map=None):
        """Create a conditional edge with a router function."""
        if source in ["START", "__start__", START, "END", "__end__", END]:
            raise ValueError(f"Invalid source node: Routers from endpoints are not allowed.")
        if source not in self.nodes:
            raise ValueError(f"Invalid source node: '{source}' does not exist")
        
        # Check for conflicting standard edge
        if any(edge_source == source for edge_source, _ in self.edges):
            raise ValueError(f"Source node '{source}' already has a standard outgoing edge.")

        # Get or set condition code
        if condition_code:
            condition._source_code = condition_code
        else:
            condition._source_code = inspect.getsource(condition)
        
        edge_info = {"condition": condition}
        
        # Get path map
        inferred_path_map = self._infer_path_map(condition._source_code)
        final_path_map = path_map if path_map is not None else inferred_path_map

        # Save path map if available
        if final_path_map:
            edge_info["path_map"] = final_path_map.copy()
        
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
            return "!!Error: Could not identify function name in the provided code"
        
        function_name = match.group(1)
        completed_function_code = "\n".join(self.imports) + "\n" + function_code    
        local_vars = {}
        exec(completed_function_code, {"__builtins__": __builtins__}, local_vars)
        
        if function_name in local_vars and callable(local_vars[function_name]):
            new_function = local_vars[function_name]
            return new_function
        else:
            return f"!!Error: Function '{function_name}' not found after execution"

    def add_system_prompt(self, new_code: str) -> bool:
        """
        Adds or updates functions and constants using AST parsing.
        If a function or constant with the same name already exists at the top level,
        it will be replaced. Preserves other existing code.
        """
        dedented_new_code = textwrap.dedent(new_code).strip()

        if re.search(r"def\s+build_system\s*\(", dedented_new_code):
            raise ValueError("system_prompt code cannot contain 'build_system' function definition.")

        if not dedented_new_code:
            return True

        try:
            new_ast = ast.parse(dedented_new_code)
            names_to_replace = _extract_top_level_names(new_ast)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in new system_prompt code: {e}") from e

        dedented_existing_code = textwrap.dedent(self.system_prompt_code).strip()

        if not dedented_existing_code:
            self.system_prompt_code = ast.unparse(new_ast)
            return True

        try:
            existing_ast = ast.parse(dedented_existing_code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in existing system_prompt code, cannot merge: {e}") from e

        transformer = RemoveDefinitionsTransformer(names_to_replace)
        modified_existing_ast = transformer.visit(existing_ast)

        if not isinstance(modified_existing_ast, ast.Module):
             raise TypeError("AST transformation did not return a Module node.")

        final_body = modified_existing_ast.body + new_ast.body
        final_ast = ast.Module(body=final_body, type_ignores=[])

        try:
            self.system_prompt_code = ast.unparse(final_ast)
        except Exception as e:
            raise RuntimeError(f"Failed to unparse combined system_prompt code AST: {e}") from e

        return True

    def validate_graph(self):
        """Validate graph structure and return list of errors."""
        errors = []

        # Get all defined nodes
        all_defined_nodes = set(self.nodes.keys())
        
        # Check edges for invalid connections
        for s, t in self.edges:
            if s != START and s not in all_defined_nodes:
                errors.append(f"Edge source '{s}' in ('{s}', '{t}') is not a defined node and not START.")
            if t != END and t not in all_defined_nodes:
                errors.append(f"Edge target '{t}' in ('{s}', '{t}') is not a defined node and not END.")
            if t == START: 
                errors.append(f"Edge ('{s}', '{t}') targets START, which is invalid.")
            if s == END: 
                errors.append(f"Edge ('{s}', '{t}') originates from END, which is invalid.")

        # Check conditional edges for invalid configurations
        for s, edge_info in self.conditional_edges.items():
            if s not in all_defined_nodes:
                errors.append(f"Conditional edge source '{s}' is not a defined node.")
            
            path_map = edge_info.get("path_map", {})
            if not path_map:
                errors.append(f"Conditional edge source '{s}' has no paths defined in its path_map.")
            
            for path_key, path_target in path_map.items():
                if path_target != END and path_target not in all_defined_nodes:
                    errors.append(f"Conditional edge target '{path_target}' (for key '{path_key}' from '{s}') is not a defined node.")

        # Check reachability from START
        reachable_nodes = set()
        q = collections.deque()
        
        # Verify START has outgoing edges
        start_has_outgoing = any(s == START for s, _ in self.edges)
        if not start_has_outgoing and all_defined_nodes:
            errors.append("No entry point: START has no outgoing edges.")
        
        # Perform BFS from START
        if start_has_outgoing or not all_defined_nodes:
            q.append(START)
            reachable_nodes.add(START)

            while q:
                curr = q.popleft()
                # Follow standard edges
                for s_edge, t_edge in self.edges:
                    if s_edge == curr and t_edge not in reachable_nodes:
                        reachable_nodes.add(t_edge)
                        if t_edge != END: 
                            q.append(t_edge)
                
                # Follow conditional edges
                if curr in self.conditional_edges:
                    path_map = self.conditional_edges[curr].get("path_map", {})
                    for target_node in path_map.values():
                        if target_node not in reachable_nodes:
                            reachable_nodes.add(target_node)
                            if target_node != END: 
                                q.append(target_node)
            
            # Check for unreachable nodes
            for node_name in all_defined_nodes:
                if node_name not in reachable_nodes:
                    errors.append(f"Node '{node_name}' is unreachable from START.")
            
            # Check if END is reachable
            if all_defined_nodes and END not in reachable_nodes:
                errors.append("END node is unreachable from START.")

        # Check if all nodes can reach END
        if END in reachable_nodes:
            can_reach_end = {END}
            q_rev = collections.deque([END])
            
            while q_rev:
                curr_target = q_rev.popleft()
                
                # Find nodes that can reach current target via standard edges
                for s_edge, t_edge in self.edges:
                    if (t_edge == curr_target and s_edge in reachable_nodes 
                            and s_edge not in can_reach_end):
                        can_reach_end.add(s_edge)
                        if s_edge != START: 
                            q_rev.append(s_edge)
                
                # Find nodes that can reach current target via conditional edges
                for cond_s, edge_info in self.conditional_edges.items():
                    if (cond_s in reachable_nodes and cond_s not in can_reach_end):
                        path_map = edge_info.get("path_map", {})
                        if any(path_val == curr_target for path_val in path_map.values()):
                            can_reach_end.add(cond_s)
                            q_rev.append(cond_s)
            
            # Find dead-end nodes
            for node_name in all_defined_nodes:
                if node_name in reachable_nodes and node_name not in can_reach_end:
                    errors.append(f"Node '{node_name}' is reachable from START but cannot reach END (forms a dead-end path).")

        # Check for explicit dead ends (nodes with no outgoing edges)
        for node_name in all_defined_nodes:
            if node_name in reachable_nodes and node_name != END:
                has_outgoing = (any(s_edge == node_name for s_edge, _ in self.edges) or 
                               node_name in self.conditional_edges)
                               
                if not has_outgoing:
                    errors.append(f"Node '{node_name}' has no outgoing edges.")

        return sorted(list(set(errors)))