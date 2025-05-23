import os
from langgraph.graph import START, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage
from agentic_system.utils import extract_parenthesized_content, find_code_blocks
from dotenv import load_dotenv
import re

def parse_arguments(args_str):
    pos_args, kw_args = (), {}
    
    if args_str:
        eval_context = {
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
            "ToolMessage": ToolMessage,
            "START": START, 
            "END": END
        }
        
        results_from_exec = {} 
        exec_str = f"def _parsing_temp_func(*args, **kwargs): return args, kwargs\npos_args, kw_args = _parsing_temp_func({args_str})"
        
        try:
            exec(exec_str, eval_context, results_from_exec)
            
            pos_args = results_from_exec.get('pos_args', ())
            kw_args = results_from_exec.get('kw_args', {})
        except NameError as ne:
            error_msg = (f"Invalid argument: a name '{ne.name}' was used in the tool arguments "
                         f"'{args_str}' but it is not defined. Do not use '{ne.name}'.")
            raise ValueError(error_msg) from ne # Propagate as ValueError
        except SyntaxError as se:
            error_msg = (f"Syntax error in tool arguments: '{args_str}'. Please ensure the "
                         f"arguments are correctly formatted. Details: {se}")
            raise ValueError(error_msg) from se
        except Exception as e:
            # Catch other parsing errors
            raise ValueError(f"Failed to parse tool arguments '{args_str}': {e}") from e
            
    return pos_args, kw_args

def parse_decorator_tool_calls(block_content):
    """Parse decorator-style tool calls from text."""
    tool_call = None
    
    def camelfy(snake_case):
        parts = snake_case.split("_")
        return "".join([part.capitalize() for part in parts])
    
    # Code-related tools that need special handling
    code_related_tools = {
        'upsert_component': 'function_code',
        'system_prompt': 'system_prompt_code'
    }
    

    lines = block_content.split('\n')
    occurrences = sum([1 if line.strip().startswith('@@') else 0 for line in lines])
    
    # First check if we should process this block at all
    first_decorator = None
    for line in lines:
        if line.strip().startswith('@@'):
            call_match = re.match(r'@@([a-zA-Z_][a-zA-Z0-9_]*)', line.strip())
            if call_match:
                first_decorator = call_match.group(1)
                break
    
    # Skip blocks with multiple @@ that aren't system_prompt
    if occurrences > 1 and first_decorator != 'system_prompt':
        raise ValueError(f"Found block with multiple decorators (found {occurrences})")
    
    # Process just the first decorator in the block
    processed_decorator = False
    i = 0
    
    while i < len(lines) and not processed_decorator:
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
                    pos_args, kw_args = parse_arguments(args_str)

                    # Map decorator name to tool name if possible
                    tool_name = camelfy(decorator_name)
                    
                    if decorator_name in code_related_tools:
                        # For code-related decorators, include the entire block after the decorator
                        code_start = end_line_idx + 1
                        code_end = len(lines)
                        
                        content = '\n'.join(lines[code_start:code_end])
                        
                        param_name = code_related_tools[decorator_name]
                        kw_args[param_name] = content
                    
                    tool_call = {
                        'name': tool_name,
                        'decorator_name': decorator_name,
                        'pos_args': pos_args,
                        'kw_args': kw_args
                    }
                    
                    processed_decorator = True
            
        i += 1
    
    return tool_call

def execute_decorator_tool_calls(response_content, available_tools):
    """Execute decorator-style tool calls found in the text."""
    # Extract code blocks
    code_blocks = find_code_blocks(response_content)
    if not code_blocks:
        print("No markdown blocks found in the agent's response! Unable to parse decorators.")

        
    tool_messages = []
    tool_results = {}
    
    def add_skipped_calls_message(current_index):
        remaining_calls = len(code_blocks) - current_index - 1
        if remaining_calls > 0:
            tool_messages.append(f"Note: {remaining_calls} remaining decorator call(s) in this response skipped. You can make new decorator calls in your next response.")
    
    for i, code_block in enumerate(code_blocks):
        try:
            tool_call = parse_decorator_tool_calls(code_block)
        except Exception as e:
            parsing_error_message = f"Error parsing code block {i}: {repr(e)}"
            tool_messages.append(parsing_error_message)
            
            add_skipped_calls_message(i)
            break

        if tool_call is None:
            tool_messages.append(f"Found no decorators in code block {i}")
            continue

        tool_name = tool_call['name']
        decorator_name = tool_call['decorator_name']
        pos_args = tool_call.get('pos_args', ())
        kw_args = tool_call.get('kw_args', {})
        
        if tool_name in available_tools:
            try:
                tool = available_tools[tool_name]
                
                if hasattr(tool, 'func') and callable(tool.func):
                    result = tool.func(*pos_args, **kw_args)
                else:
                    result = tool.invoke(kw_args)
                
                tool_messages.append(str(result) if result else f"Decorator @@{decorator_name} executed successfully.")
                tool_results[tool_name] = result
                
                if isinstance(result, str) and "!!Error" in result:
                    add_skipped_calls_message(i)
                    break
            except Exception as e:
                error_message = f"Error executing decorator @@{decorator_name}: {repr(e)}"
                tool_messages.append(error_message)
                tool_results[tool_name] = error_message
                
                add_skipped_calls_message(i)
                break
        else:
            tool_messages.append(f"Decorator @@{decorator_name} not found.")
            if decorator_name == "test_system":
                tool_messages.append(f"Remember that you only have access to the @@test_meta_system decorator.")
            add_skipped_calls_message(i)
            break

    # tool_messages.append("Please continue according to the InitialPlan.")
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

        if wrapper == "google":
            if "gemini-2.5-flash" in model_name:
                model_wrapper["google"] = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key,
                    thinking_budget=0  # non-thinking mode
                )
            else:
                model_wrapper["google"] = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key
                )
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
        return self

    def invoke(self, input):
        return self.model.invoke(input)
    
    def execute_tool_calls(self, response, function_call_type="normal"):
        if function_call_type == "normal":
            return execute_tool_calls(response, self.available_tools)
        elif function_call_type == "decorator":
            return execute_decorator_tool_calls(response, self.available_tools)
        else:
            raise ValueError(f"function_call_type {function_call_type} is not available")