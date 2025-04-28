import os
from langgraph.graph import START, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from agentic_system.utils import extract_parenthesized_content, find_code_blocks
from dotenv import load_dotenv
import re

def parse_arguments(args_str):
    args = {}
    
    if args_str:
        try:
            exec_str = f"def parsing_function(**kwargs): return kwargs\nargs = parsing_function({args_str})"
            locals = {"HumanMessage" : HumanMessage, "START": START, "END": END}
            exec(exec_str, {}, locals)
            args = locals.get('args', {})
        except Exception as e:
            print(f"Error parsing arguments: {e}")
    
    return args

def parse_decorator_tool_calls(text):
    """Parse decorator-style tool calls from text."""
    tool_calls = []
    
    def camelfy(snake_case):
        parts = snake_case.split("_")
        return "".join([part.capitalize() for part in parts])
    
    # Code-related tools that need special handling
    code_related_tools = {
        'add_component': 'function_code',
        'edit_component': 'new_function_code',
        'system_prompt': 'system_prompt_code'
    }
    
    # Extract code blocks
    code_blocks = find_code_blocks(text)
    if not code_blocks:
        print("No code blocks found!")
    
    for block in code_blocks:
        lines = block.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('@@'):
                call_match = re.match(r'@@([a-zA-Z_][a-zA-Z0-9_]*)', line)
                if call_match:
                    decorator_name = call_match.group(1)
                    start_pos = call_match.end()
                    
                    open_paren_pos = line.find('(', start_pos)
                    
                    if open_paren_pos != -1:
                        # Extract arguments
                        args_str, end_line_idx = extract_parenthesized_content(lines, i, open_paren_pos)
                        args = parse_arguments(args_str)

                        # Map decorator name to tool name if possible
                        tool_name = camelfy(decorator_name)
                        
                        if decorator_name in code_related_tools:
                            code_start = end_line_idx + 1
                            code_end = len(lines)
                            
                            # Find the next decorator
                            for k in range(code_start, len(lines)):
                                if lines[k].strip().startswith('@@'):
                                    code_end = k
                                    break
                            
                            content = '\n'.join(lines[code_start:code_end])
                            
                            param_name = code_related_tools[decorator_name]
                            args[param_name] = content
                            
                            i = code_end - 1
                        else:
                            i = end_line_idx
                        
                        tool_calls.append({
                            'name': tool_name,
                            'args': args
                        })
            
            i += 1
    
    return tool_calls

def execute_decorator_tool_calls(response, available_tools):
    """Execute decorator-style tool calls found in the text."""
    tool_calls = parse_decorator_tool_calls(response)
    if not tool_calls:
        return [], {}
        
    tool_messages = []
    tool_results = {}
    
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        if tool_name in available_tools:
            try:
                result = available_tools[tool_name].invoke(tool_args)
                
                tool_messages.append(str(result) if result else f"Tool {tool_name} executed successfully.")
                tool_results[tool_name] = result
            except Exception as e:
                error_message = f"Error executing tool {tool_name}: {repr(e)}."
                tool_messages.append(error_message)
                tool_results[tool_name] = error_message
                break
        else:
            tool_messages.append(f"Tool {tool_name} not found")
            break

    human_message = HumanMessage(content = "\n\n".join(tool_messages)) if tool_messages else None
                
    return human_message, tool_results

def execute_tool_calls(response, available_tools):
    """Execute any tool calls in the llm response."""
    if not hasattr(response, "tool_calls") or not response.tool_calls:
        return [], {}
        
    tool_messages = []
    tool_results = {}
    for tool_call in response.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']
        
        if tool_name in available_tools:
            try:
                result = available_tools[tool_name].invoke(tool_args)
                tool_messages.append(ToolMessage(
                    content=str(result) if result else f"Tool {tool_name} executed successfully.",
                    tool_call_id=tool_id,
                    name=tool_name
                ))
                tool_results[tool_name] = result
            except Exception as e:
                error_message = f"Error executing tool {tool_name}: {repr(e)}"
                tool_messages.append(ToolMessage(
                    content=error_message,
                    tool_call_id=tool_id,
                    name=tool_name
                ))
                tool_results[tool_name] = error_message
                
    return tool_messages, tool_results

load_dotenv()
def get_model(wrapper, model_name, temperature):
    api_keys = {
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "blablador": "HELMHOLTZ_API_KEY",
        "scads": "SCADS_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY"
    }
    
    if wrapper not in api_keys:
        raise ValueError(f"Invalid wrapper: '{wrapper}'. Supported: {', '.join(api_keys.keys())}")
    
    key_name = api_keys[wrapper]
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"Missing environment variable: {key_name} required for {wrapper}")
    
    try:
        model_wrapper = {
            "google": ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key
                ),
            "openai": ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=api_key
                ),
            "blablador": ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=api_key,
                    cache=False,
                    max_retries=2,
                    base_url="https://api.helmholtz-blablador.fz-juelich.de/v1/"
                ),
            "scads": ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=api_key,
                    cache=False,
                    max_retries=2,
                    base_url="https://llm.scads.ai/v1"
                ),
            "perplexity": ChatOpenAI(
                    model="sonar",
                    temperature=temperature,
                    api_key=api_key,
                    base_url="https://api.perplexity.ai"
            )
        }
        return model_wrapper[wrapper]
    except Exception as e:
        raise RuntimeError(f"Failed to initialize {wrapper} model: {str(e)}") from e

class LargeLanguageModel:

    def __init__(self, temperature=0.2, wrapper="openai", model_name="gpt-4.1-nano"):
        self.model = get_model(wrapper, model_name, temperature)
        self.wrapper = wrapper
        self.available_tools = {}

    def bind_tools(self, tools, function_call_type="normal", parallel_tool_calls=True):
        if tools:
            tool_objects = [tool for tool in tools if tool.name]
            if len(tools) > len(tool_objects):
                raise ValueError("All values in the list must be tool objects")
            self.available_tools.update({tool.name: tool for tool in tool_objects})
            
            if function_call_type == "normal":
                if self.wrapper=="google":
                    self.model = self.model.bind_tools(tool_objects)
                else:
                    self.model = self.model.bind_tools(tool_objects, parallel_tool_calls=parallel_tool_calls)
        return self.model

    def invoke(self, input):
        return self.model.invoke(input)
    
    def execute_tool_calls(self, response, function_call_type="normal"):
        if function_call_type == "normal":
            return execute_tool_calls(response, self.available_tools)
        elif function_call_type == "decorator":
            return execute_decorator_tool_calls(response, self.available_tools)
        else:
            raise ValueError(f"function_call_type {function_call_type} is not available")
