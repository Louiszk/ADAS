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
            cleaned_message = getattr(message, 'type', 'Unknown') + ": "
            if hasattr(message, 'content') and message.content:
                cleaned_message += message.content
            if hasattr(message, 'tool_calls') and message.tool_calls:
                cleaned_message += str(message.tool_calls)
            if not hasattr(message, 'content') and not hasattr(message, 'tool_calls'):
                cleaned_message += str(message)
            cleaned_messages.append(cleaned_message)
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

def find_code_blocks(text: str) -> list[str]:
    """
    Finds code blocks delimited by triple backticks (```) potentially on their own lines.
    It ignores triple backticks when they appear inside multi-line triple-quoted strings.
    """
    lines = text.splitlines()
    code_blocks = []
    current_block_lines = []
    in_code_block = False
    in_triple_string = False
    triple_string_delimiter = None # Will store '"""' or "'''"

    for i, line in enumerate(lines):
        is_delimiter_line = line.strip().startswith('```')
        effective_line_content = line
        start_search_pos = 0

        while True:
            next_triple_double = effective_line_content.find('"""', start_search_pos)
            next_triple_single = effective_line_content.find("'''", start_search_pos)

            # Find the earliest triple quote delimiter
            if next_triple_double != -1 and (next_triple_single == -1 or next_triple_double < next_triple_single):
                delim_pos = next_triple_double
                current_delim = '"""'
            elif next_triple_single != -1:
                delim_pos = next_triple_single
                current_delim = "'''"
            else:
                break

            if in_triple_string:
                if current_delim == triple_string_delimiter:
                    in_triple_string = False
                    triple_string_delimiter = None
            else:
                in_triple_string = True
                triple_string_delimiter = current_delim

            # Continue searching
            start_search_pos = delim_pos + 3

        if is_delimiter_line and not in_triple_string:
            if not in_code_block:
                # Start of a new code block
                in_code_block = True
                current_block_lines = []
            else:
                # End of the current code block
                in_code_block = False
                if current_block_lines:
                    code_blocks.append("\n".join(current_block_lines))
                # Reset for potential next block
                current_block_lines = []
        elif in_code_block:
            current_block_lines.append(line)

    return code_blocks