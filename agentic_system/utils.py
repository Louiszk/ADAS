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