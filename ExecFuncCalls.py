import json
import os
from typing import List

from dotenv import load_dotenv
from litellm import completion
from litellm.exceptions import RateLimitError

# Load environment variables from .env so litellm sees API keys.
load_dotenv()

def list_files() -> List[str]:
    """List files in the current directory."""
    return os.listdir(".")

def read_file(file_name: str) -> str:
    """Read a file's contents."""
    try:
        with open(file_name, "r") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: {file_name} not found."
    except Exception as e:
        return f"Error: {str(e)}"


tool_functions = {
    "list_files": list_files,
    "read_file": read_file
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Returns a list of files in the directory.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the content of a specified file in the directory.",
            "parameters": {
                "type": "object",
                "properties": {"file_name": {"type": "string"}},
                "required": ["file_name"]
            }
        }
    }
]

# Our rules are simplified since we don't have to worry about getting a specific output format
agent_rules = [{
    "role": "system",
    "content": """
You are an AI agent that can perform tasks by using available tools. 

If a user asks about files, documents, or content, first list the files before reading them.
"""
}]

user_task = input("What would you like me to do? ")

memory = [{"role": "user", "content": user_task}]

messages = agent_rules + memory
print("Messages to LLM:")
for msg in messages:
    print(msg)

try:
    response = completion(
        model="openai/gpt-4o",
        messages=messages,
        tools=tools,
        max_tokens=1024
    )
except RateLimitError:
    print("OpenAI quota exceeded, falling back to Gemini model.")
    response = completion(
        model="gemini/gemini-2.0-flash",
        messages=messages,
        tools=tools,
        max_tokens=1024
    )

print("Response from LLM:")
print(response)

message = response.choices[0].message

if message.tool_calls:
    # Extract the first tool call from the response when the model decides to use one.
    tool = message.tool_calls[0]
    tool_name = tool.function.name
    tool_args = json.loads(tool.function.arguments)
    result = tool_functions[tool_name](**tool_args)

    print(f"Tool Name: {tool_name}")
    print(f"Tool Arguments: {tool_args}")
    print(f"Result: {result}")
else:
    print("Assistant Response:")
    print(message.content)