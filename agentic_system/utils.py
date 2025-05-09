import subprocess

def get_filtered_packages(exclude_packages=None):
    if exclude_packages is None:
        exclude_packages = []
    
    result = subprocess.run(['pip', 'list', '--not-required'], 
                          capture_output=True, text=True)
    
    packages = []
    for line in result.stdout.strip().split('\n')[2:]:  # Skip header lines
        if line.strip():
            parts = line.split()
            if len(parts) >= 2:
                package_name = parts[0]
                version = parts[1]
                
                if package_name not in exclude_packages:
                    packages.append(f"{package_name} {version}")
    print(packages)
    return packages

def clean_messages(output):
    cleaned_messages = []
    if "messages" in output:
        for message in output["messages"]:
            cleaned_message_str = getattr(message, 'type', 'Unknown') + ": "
            if hasattr(message, 'content') and message.content:
                cleaned_message_str += str(message.content)
            if hasattr(message, 'tool_calls') and message.tool_calls:
                cleaned_message_str += str(message.tool_calls)
            if not (hasattr(message, 'content') and message.content) and \
               not (hasattr(message, 'tool_calls') and message.tool_calls):
                cleaned_message_str += str(message)
            cleaned_messages.append(cleaned_message_str)
    else:
        print('Warning: No "messages" key found.')
    return cleaned_messages


def extract_parenthesized_content(lines, start_line_idx, start_pos):
    """Extract content inside matching parentheses, handling multi-line cases and preserving structure."""
    content = []
    current_line = ""
    paren_level = 1
    in_string = False
    string_delim = None
    escape_next = False
    line_idx = start_line_idx
    pos = start_pos + 1
    
    while line_idx < len(lines):
        curr_line = lines[line_idx]
        current_line = ""  # Reset for each line
        
        while pos < len(curr_line):
            char = curr_line[pos]
            
            if char == '#' and not in_string:
                break  # Skip rest of line
            
            # Handle string literals
            if (char == '"' or char == "'") and not escape_next:
                if not in_string:
                    in_string = True
                    string_delim = char
                elif char == string_delim:
                    in_string = False
                    string_delim = None
            
            # Handle escape characters in strings
            if char == '\\' and in_string and not escape_next:
                escape_next = True
            else:
                escape_next = False
            
            # Count parentheses (only when not in a string)
            if not in_string:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                    
                    # If we've found the matching closing parenthesis
                    if paren_level == 0:
                        content.append(current_line)
                        return "\n".join(content), line_idx
            
            current_line += char
            pos += 1
        
        content.append(current_line)
        
        # Move to the next line
        line_idx += 1
        if line_idx < len(lines):
            pos = 0
        else:
            break
    
    return "\n".join(content), line_idx - 1

def find_code_blocks(text):
    """
    Finds code blocks delimited by triple backticks (```) with proper handling of
    nested triple quotes and triple backticks within strings.
    """
    lines = text.splitlines()
    code_blocks = []
    current_block = []
    in_code_block = False
    in_triple_string = False
    triple_string_type = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_backtick_line = stripped.startswith('```')
        
        # Process backtick delimiters only when not in a triple-quoted string
        if is_backtick_line and not in_triple_string:
            if not in_code_block:
                # Start a new code block
                in_code_block = True
                current_block = []
            else:
                # End current code block
                in_code_block = False
                if current_block:
                    code_blocks.append('\n'.join(current_block))
                current_block = []
            continue
                
        if in_code_block:
            current_block.append(line)
            
            # Track triple-quoted string state within the line
            j = 0
            while j < len(line):
                if j + 2 < len(line):
                    next_chars = line[j:j+3]
                    is_triple_quote = next_chars == '"""' or next_chars == "'''"
                    # Check it's not escaped
                    is_escaped = j > 0 and line[j-1] == '\\'
                    
                    if is_triple_quote and not is_escaped:
                        if not in_triple_string:
                            # Start triple-quoted string
                            in_triple_string = True
                            triple_string_type = next_chars
                        elif triple_string_type == next_chars:
                            # End matching triple-quoted string
                            in_triple_string = False
                            triple_string_type = None
                        
                        j += 3
                        continue
                j += 1
        
    return code_blocks

def get_metrics(raw_stream_outputs, duration):
    total_iterations = len(raw_stream_outputs)
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    llm_calls = 0
    processed_message_ids = set()
    
    for step_output_dict in raw_stream_outputs:
        if "messages" in step_output_dict:
            messages_in_node_output = step_output_dict.get("messages")
            for msg in messages_in_node_output:
                msg_id = getattr(msg, 'id', None)
                if msg_id and msg_id in processed_message_ids:
                    continue
                if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                    usage = msg.usage_metadata
                    input_tokens += usage.get('input_tokens', 0)
                    output_tokens += usage.get('output_tokens', 0)
                    total_tokens += usage.get('total_tokens', 0)
                    llm_calls += 1
                    if msg_id:
                        processed_message_ids.add(msg_id)
        else:
            print('Warning: No "messages" key found.')
                            
    return {
        "total_iterations": total_iterations,
        "duration_seconds": round(duration, 3),
        "llm_calls": llm_calls,
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
    }
