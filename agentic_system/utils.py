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