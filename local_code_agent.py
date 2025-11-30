import os
import sys
import json
import requests
import argparse
import subprocess
import ast
import difflib
import shutil
import base64
import time
from typing import List, Dict, Any, Optional, Tuple

# --- Configuration & Constants ---
DEFAULT_API_BASE = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_MODEL_ID = "gemma3-4b-it-abliterated-i1"
DEFAULT_WEBUI = False
DEFAULT_STATIC_DIR = r"D:\ai"

MAX_CONTEXT_CHARS = 160000 
MAX_FILE_READ_CHARS = 100000 
DANGEROUS_COMMANDS = ["rm", "sudo", "ip", "mv", "chmod", "chown", "dd", "fdisk", "mkfs", ":(){ :|:& };:"]
IGNORE_DIRS = {".git", "__pycache__", "node_modules", "venv", ".env", "dist", "build", ".idea", ".vscode", "target"}

# Define files that strictly cannot be modified or deleted by the agent
PROTECTED_FILES = {os.path.basename(__file__), "codeai.py", "local_code_agent.py"}

# Globals for paths (set in main)
PATHS = {
    "memory": ".agent_memory.json",
    "prompts": "saved_prompts.json",
    "mcp": "mcp_servers.json",
    "static": "."
}

SYSTEM_PROMPT = """You are a capable coding assistant and agent, similar to Claude Code. 
You have access to the local file system, can execute terminal commands, and have persistent memory.
You may also have access to external tools provided by MCP servers.

Your goal is to help the user write code, debug issues, and manage projects.

You have the following tools available:
1. read_file(path): Reads the content of a file.
2. write_file(path, content): Overwrites or creates a file with the given content.
3. replace_in_file(path, old_text, new_text): Replaces a specific string in a file with new text. Use this for small edits.
4. list_files(path): Lists files in a directory.
5. search_code(directory, term): Searches for a string pattern across files in a directory.
6. run_command(command): Executes a shell command.
7. lint_file(path): Checks the file for syntax errors (Python).
8. clone_repo(url): Clones a public GitHub repository to the current directory.
9. remember_info(category, info): Saves important information to memory.
10. retrieve_info(category): Retrieves information from memory.
11. sequential_thinking(thought, step, total_steps): Plan complex tasks.
(Additional MCP tools may be available)

GUIDELINES:
- **Sequential Thinking**: ALWAYS use `sequential_thinking` first for complex tasks.
- **Diffs**: When using `replace_in_file`, ensure `old_text` matches EXACTLY, including whitespace.
- **Safety**: Be careful with destructive commands.
"""

# --- ANSI Colors for Terminal ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    MAGENTA = '\033[35m'
    YELLOW = '\033[33m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# --- Logic & Managers ---

class SafetyManager:
    def __init__(self, safe_mode: bool = False):
        self.safe_mode = safe_mode

    def check_command(self, command: str) -> bool:
        is_dangerous = any(cmd in command.split() for cmd in DANGEROUS_COMMANDS) or "sudo" in command
        if self.safe_mode or is_dangerous:
            print(f"\n{Colors.WARNING}⚠️  SECURITY ALERT: {command}{Colors.ENDC}")
            try:
                response = input(f"{Colors.WARNING}Allow execution? [y/N]: {Colors.ENDC}")
                return response.lower() == 'y'
            except EOFError:
                return False
        return True

class ContextManager:
    def __init__(self, max_chars: int = MAX_CONTEXT_CHARS):
        self.max_chars = max_chars

    def prune_history(self, history: List[Dict]) -> List[Dict]:
        current_chars = sum(len(json.dumps(m)) for m in history)
        if current_chars <= self.max_chars:
            return history
        
        if len(history) <= 7: return history
        
        system_prompt = history[0]
        recent = history[-6:]
        prunable = history[1:-6]
        
        while prunable and (len(json.dumps([system_prompt] + prunable + recent)) > self.max_chars):
            prunable.pop(0)
            
        return [system_prompt] + prunable + recent

# --- MCP Client Implementation (Basic Stdio) ---

class SimpleMCPClient:
    def __init__(self, name, command, env=None):
        self.name = name
        self.command = command
        self.process = None
        self.request_id = 0
        self.tools = []
        self.env = env or os.environ.copy()

    def start(self):
        try:
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
                bufsize=1 # Line buffered
            )
            # Basic Initialization
            self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "local-code-agent", "version": "1.0"}
            })
            # Wait for init response (naive blocking read)
            init_resp = self._read_response()
            
            self._send_notification("notifications/initialized", {})
            
            # Fetch Tools
            self.refresh_tools()
            return True, "Connected"
        except Exception as e:
            return False, str(e)

    def refresh_tools(self):
        resp = self._send_request("tools/list", {})
        response = self._read_response(resp)
        if response and "result" in response and "tools" in response["result"]:
            self.tools = response["result"]["tools"]
            # Tag tools with source
            for t in self.tools:
                t["source_mcp"] = self.name
        return self.tools

    def call_tool(self, tool_name, arguments):
        resp_id = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        response = self._read_response(resp_id)
        if response and "result" in response:
            return response["result"]
        if response and "error" in response:
            return f"MCP Error: {response['error']['message']}"
        return "Error: No response from tool."

    def _send_request(self, method, params):
        self.request_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        self._write(req)
        return self.request_id

    def _send_notification(self, method, params):
        req = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        self._write(req)

    def _write(self, data):
        if self.process and self.process.stdin:
            try:
                json.dump(data, self.process.stdin)
                self.process.stdin.write('\n')
                self.process.stdin.flush()
            except BrokenPipeError:
                print(f"{Colors.FAIL}MCP Server '{self.name}' pipe broken.{Colors.ENDC}")

    def _read_response(self, wait_for_id=None):
        # Naive blocking reader that filters for the response ID
        if not self.process or not self.process.stdout: return None
        start_time = time.time()
        while time.time() - start_time < 10: # 10s Timeout
            line = self.process.stdout.readline()
            if not line: break
            try:
                msg = json.loads(line)
                if "id" in msg and msg["id"] == wait_for_id:
                    return msg
            except: continue
        return None

    def close(self):
        if self.process:
            self.process.terminate()

class MCPManager:
    def __init__(self):
        self.clients = {} # name -> SimpleMCPClient
        self.config_path = PATHS["mcp"]

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def save_config(self, config):
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def stop_all(self):
        for name, client in self.clients.items():
            client.close()
        self.clients = {}

    def start_all(self):
        config = self.load_config()
        for name, cmd in config.items():
            if name not in self.clients:
                client = SimpleMCPClient(name, cmd)
                ok, msg = client.start()
                if ok:
                    print(f"{Colors.GREEN}MCP '{name}' started.{Colors.ENDC}")
                    self.clients[name] = client
                else:
                    print(f"{Colors.FAIL}MCP '{name}' failed: {msg}{Colors.ENDC}")

    def restart_all(self):
        self.stop_all()
        self.start_all()
        return self.load_config()

    def get_all_tools(self):
        all_tools = []
        for client in self.clients.values():
            # Convert MCP tool format to OpenAI format
            for tool in client.tools:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})
                    }
                }
                all_tools.append(openai_tool)
        return all_tools

# --- Prompt Management & Character Import ---

def get_saved_prompts():
    prompts = {"Default": SYSTEM_PROMPT}
    if os.path.exists(PATHS["prompts"]):
        try:
            with open(PATHS["prompts"], 'r') as f:
                prompts.update(json.load(f))
        except: pass
    return prompts

def save_custom_prompt(name, content):
    prompts = {}
    if os.path.exists(PATHS["prompts"]):
        try:
            with open(PATHS["prompts"], 'r') as f:
                prompts = json.load(f)
        except: pass
    
    prompts[name] = content
    try:
        with open(PATHS["prompts"], 'w') as f:
            json.dump(prompts, f, indent=2)
        return True, "Saved."
    except Exception as e:
        return False, str(e)

def delete_custom_prompt(name):
    if name == "Default": return False, "Cannot delete 'Default'."
    prompts = {}
    if os.path.exists(PATHS["prompts"]):
        try:
            with open(PATHS["prompts"], 'r') as f: prompts = json.load(f)
        except: pass
    
    if name in prompts:
        del prompts[name]
        try:
            with open(PATHS["prompts"], 'w') as f: json.dump(prompts, f, indent=2)
            return True, f"Deleted '{name}'."
        except Exception as e: return False, str(e)
    return False, "Not found."

def parse_character_card(file_path):
    try:
        data = None
        if file_path.lower().endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        elif file_path.lower().endswith('.png'):
            try:
                from PIL import Image
                img = Image.open(file_path)
                img.load()
                if 'chara' in img.info:
                    decoded = base64.b64decode(img.info['chara']).decode('utf-8')
                    data = json.loads(decoded)
                elif 'Description' in img.info:
                    data = {k: v for k, v in img.info.items()}
            except: return "Error: PIL required for PNG cards."

        if not data: return "Error: No data found."

        char = data.get('data', data) if 'data' in data else data
        name = char.get('name', 'Unknown')
        prompt = f"You are {name}.\n\n"
        if char.get('description'): prompt += f"Description:\n{char['description']}\n\n"
        if char.get('personality'): prompt += f"Personality:\n{char['personality']}\n\n"
        if char.get('scenario'): prompt += f"Scenario:\n{char['scenario']}\n\n"
        if char.get('first_mes'): prompt += f"First Message:\n{char['first_mes']}\n\n"
        prompt += "Roleplay Instructions:\nStay in character. React realistically."
        return prompt
    except Exception as e: return f"Import failed: {str(e)}"

# --- Tool Implementations ---

def backup_file(path: str) -> str:
    if os.path.exists(path):
        backup_path = path + ".old"
        try:
            shutil.copy2(path, backup_path)
            return backup_path
        except: pass
    return None

def read_file(path: str) -> str:
    try:
        if not os.path.exists(path): return f"Error: File '{path}' does not exist."
        if os.path.getsize(path) > MAX_FILE_READ_CHARS: return "Error: File too large."
        with open(path, 'r', encoding='utf-8') as f: return f.read()
    except Exception as e: return f"Error: {e}"

def write_file(path: str, content: str) -> str:
    try:
        if os.path.basename(path) in PROTECTED_FILES: return "Error: Access denied."
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        
        diff_text = ""
        if os.path.exists(abs_path):
            backup_file(abs_path)
            with open(abs_path, 'r', encoding='utf-8') as f: old_content = f.read()
            diff = difflib.unified_diff(old_content.splitlines(), content.splitlines(), fromfile='Original', tofile='New', lineterm='')
            diff_text = "\n".join(diff) or "No changes."
            diff_text = f"\n\nDiff:\n```diff\n{diff_text}\n```"
        else: diff_text = "\n\n(New file created)"

        with open(abs_path, 'w', encoding='utf-8') as f: f.write(content)
        return f"Successfully wrote to '{path}'.{diff_text}"
    except Exception as e: return f"Error: {e}"

def replace_in_file(path: str, old_text: str, new_text: str) -> str:
    try:
        if not os.path.exists(path): return "Error: File not found."
        if os.path.basename(path) in PROTECTED_FILES: return "Error: Access denied."
        backup_file(path)
        with open(path, 'r', encoding='utf-8') as f: content = f.read()
        if old_text not in content: return "Error: 'old_text' not found."
        new_content = content.replace(old_text, new_text)
        
        diff = difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), fromfile='Original', tofile='New', lineterm='')
        diff_text = "\n".join(diff)
        
        with open(path, 'w', encoding='utf-8') as f: f.write(new_content)
        return f"Replaced text in '{path}'.\n\nDiff:\n```diff\n{diff_text}\n```"
    except Exception as e: return f"Error: {e}"

def list_files(path: str = ".") -> str:
    try:
        if not os.path.exists(path): return "Error: Path not found."
        files = os.listdir(path)
        return "\n".join([f"{f}/" if os.path.isdir(os.path.join(path, f)) else f for f in files if f not in IGNORE_DIRS])
    except Exception as e: return f"Error: {e}"

def search_code(directory: str, term: str) -> str:
    results = []
    try:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                try:
                    fp = os.path.join(root, file)
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f):
                            if term in line: 
                                results.append(f"{fp}:{i+1}: {line.strip()[:100]}")
                                if len(results) >= 50: return "\n".join(results) + "\n... (more)"
                except: continue
        return "\n".join(results) if results else "No results."
    except Exception as e: return f"Error: {e}"

def run_command(command: str, safety_manager: SafetyManager = None, cwd: str = ".") -> str:
    if any(p in command for p in PROTECTED_FILES) and any(d in command for d in DANGEROUS_COMMANDS + [">"]):
        return "Error: Command blocked (protected file)."
    if safety_manager and not safety_manager.check_command(command): return "Error: Denied."
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120, cwd=cwd)
        return (res.stdout + ("\n[STDERR]\n" + res.stderr if res.stderr else "")).strip() or "[Success]"
    except Exception as e: return f"Error: {e}"

def clone_repo(url: str, cwd: str = ".") -> str:
    return run_command(f"git clone {url}", cwd=cwd)

def lint_file(path: str) -> str:
    if not path.endswith(".py"): return "Linting only for Python."
    try:
        with open(path, "r", encoding='utf-8') as f: ast.parse(f.read())
        return "Passed."
    except Exception as e: return f"Error: {e}"

def remember_info(category: str, info: str) -> str:
    try:
        data = {}
        if os.path.exists(PATHS["memory"]):
            with open(PATHS["memory"], 'r') as f: data = json.load(f)
        data[category] = info
        with open(PATHS["memory"], 'w') as f: json.dump(data, f, indent=2)
        return f"Saved to '{category}'."
    except Exception as e: return f"Error: {e}"

def forget_info(category: str) -> str:
    try:
        if not os.path.exists(PATHS["memory"]): return "No memory file."
        with open(PATHS["memory"], 'r') as f: data = json.load(f)
        if category in data:
            del data[category]
            with open(PATHS["memory"], 'w') as f: json.dump(data, f, indent=2)
            return f"Deleted memory '{category}'."
        return f"Category '{category}' not found."
    except Exception as e: return f"Error: {e}"

def retrieve_info(category: str = None) -> str:
    try:
        if not os.path.exists(PATHS["memory"]): return "No memory."
        with open(PATHS["memory"], 'r') as f: data = json.load(f)
        return data.get(category, "Not found") if category else json.dumps(data, indent=2)
    except Exception as e: return f"Error: {e}"

# --- API Interaction ---

BASE_TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "replace_in_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "search_code", "parameters": {"type": "object", "properties": {"directory": {"type": "string"}, "term": {"type": "string"}}, "required": ["directory", "term"]}}},
    {"type": "function", "function": {"name": "run_command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "clone_repo", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "lint_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "remember_info", "parameters": {"type": "object", "properties": {"category": {"type": "string"}, "info": {"type": "string"}}, "required": ["category", "info"]}}},
    {"type": "function", "function": {"name": "retrieve_info", "parameters": {"type": "object", "properties": {"category": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "sequential_thinking", "parameters": {"type": "object", "properties": {"thought": {"type": "string"}, "step": {"type": "integer"}, "total_steps": {"type": "integer"}}, "required": ["thought", "step", "total_steps"]}}}
]

class AgentClient:
    def __init__(self, base_url, api_key, model=None, safety_manager=None, mcp_manager=None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.model = model or DEFAULT_MODEL_ID
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.safety = safety_manager or SafetyManager()
        self.context_mgr = ContextManager()
        self.mcp_mgr = mcp_manager
        self.cwd = "." 

    def update_system_prompt(self, new_prompt):
        self.history[0]["content"] = new_prompt
        return "System prompt updated."

    def fetch_models(self):
        try:
            resp = requests.get(f"{self.base_url}/models", headers=self.headers)
            if resp.status_code == 200:
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
        except: pass
        return ["local-model", "gemma3-4b-it-abliterated-i1"]

    def chat_step_1_send(self, user_input: str = None) -> Tuple[str, Any]:
        if user_input:
            context_msg = f" [Current Working Directory: ./{self.cwd}] {user_input}"
            self.history.append({"role": "user", "content": context_msg})
        
        self.history = self.context_mgr.prune_history(self.history)
        
        # Merge Base tools with MCP tools
        current_tools = BASE_TOOLS_SCHEMA[:]
        if self.mcp_mgr:
            current_tools.extend(self.mcp_mgr.get_all_tools())

        payload = {
            "model": self.model,
            "messages": self.history,
            "tools": current_tools,
            "tool_choice": "auto",
            "temperature": 0.0
        }

        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload)
            response.raise_for_status()
            choice = response.json()["choices"][0]["message"]
            
            if choice.get("tool_calls"):
                self.history.append(choice)
                return None, choice["tool_calls"]
            else:
                content = choice.get("content", "")
                self.history.append({"role": "assistant", "content": content})
                return content, None
        except Exception as e:
            return f"API Error: {e}", None

    def chat_step_2_execute(self, tool_calls, approved=False) -> str:
        def resolve(path):
            if os.path.isabs(path): return path
            return os.path.join(self.cwd, path)

        results = []
        for tool in tool_calls:
            fname = tool["function"]["name"]
            try:
                args = json.loads(tool["function"]["arguments"])
                res = None
                
                # Built-in Tools
                if fname == "sequential_thinking": res = f"Thought: {args.get('thought')}"
                elif fname == "read_file": res = read_file(resolve(args["path"]))
                elif fname == "write_file": res = write_file(resolve(args["path"]), args["content"])
                elif fname == "replace_in_file": res = replace_in_file(resolve(args["path"]), args["old_text"], args["new_text"])
                elif fname == "list_files": res = list_files(resolve(args.get("path", ".")))
                elif fname == "search_code": res = search_code(resolve(args["directory"]), args["term"])
                elif fname == "run_command": 
                    sm = None if approved else self.safety
                    res = run_command(args["command"], sm, cwd=self.cwd)
                elif fname == "clone_repo": res = clone_repo(args["url"], cwd=self.cwd)
                elif fname == "lint_file": res = lint_file(resolve(args["path"]))
                elif fname == "remember_info": res = remember_info(args["category"], args["info"])
                elif fname == "retrieve_info": res = retrieve_info(args.get("category"))
                
                # Check MCP Tools
                elif self.mcp_mgr:
                    # Find which client owns this tool
                    for client_name, client in self.mcp_mgr.clients.items():
                        for t in client.tools:
                            if t["name"] == fname:
                                res = client.call_tool(fname, args)
                                break
                        if res is not None: break
                
                if res is None: res = f"Unknown tool: {fname}"

            except Exception as e: res = f"Tool Error: {e}"

            results.append({"role": "tool", "tool_call_id": tool["id"], "name": fname, "content": str(res)})
            
        self.history.extend(results)
        return "Tools executed."

# --- Web UI Logic ---

def launch_webui(client, mcp_mgr):
    try: import gradio as gr
    except: return print("Error: Install gradio.")

    custom_css = """
    /* Main Backgrounds - Strictly Black */
    :root, body, .gradio-container { 
        background-color: #000000 !important; 
        --body-background-fill: #000000 !important;
        --background-fill-primary: #000000 !important;
        --background-fill-secondary: #000000 !important;
        color: #00FFFF !important; 
    }
    
    /* Aggressive override for any white backgrounds */
    .bg-white, .bg-gray-50, .bg-gray-100, .bg-gray-200 { 
        background-color: #000000 !important; 
        color: #00FFFF !important; 
    }

    /* Text */
    h1, h2, h3, h4, h5, h6, span, p, label { color: #00FFFF !important; }
    h1, h2, h3, h4, h5, h6 { text-shadow: 6px 5px 5px magenta !important; }
    a { color: #FFFF00 !important; text-decoration: none; }

    /* Inputs: Dark Gray Backgrounds for contrast */
    textarea, input, .form, .block.svelte-12cmxck, .wrap { 
        background-color: #111111 !important; 
        color: #00FFFF !important; 
        border: 1px solid #00FFFF !important; 
    }

    /* Dropdowns - Gray Background for Visibility */
    .dropdown, select, option, .options { 
        background-color: #333333 !important; 
        color: #00FFFF !important; 
        border: 1px solid #00FFFF !important; 
    }
    /* Specific fix for Gradio dropdown options container */
    ul.options, .options li {
        background-color: #333333 !important;
        color: #00FFFF !important;
    }

    /* Buttons */
    button, .primary, .secondary, .stop { 
        background-color: #1a1a1a !important; 
        color: #00FFFF !important; 
        border: 1px solid #00FFFF !important; 
    }
    button:hover { background-color: #333333 !important; }

    /* Chat Messages */
    .message.user { background-color: #002222 !important; border: 1px solid #00FFFF; }
    .message.bot { background-color: #111111 !important; border: 1px solid #00aaaa; }
    
    /* Code Blocks */
    pre, code { background-color: #1a1a1a !important; border: 1px solid #333; color: #00FF00 !important; }
    
    /* Collapsible Details */
    details { border: 1px solid #333; padding: 0.5em; border-radius: 4px; background-color: #111; }
    summary { cursor: pointer; font-weight: bold; color: #ff00ff; }
    """

    with gr.Blocks(title="Local Code Agent", css=custom_css) as demo:
        state_pending_tools = gr.State(None)
        state_pwd = gr.State("")

        with gr.Row():
            # Left: Explorer
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("## 📂 Explorer")
                # Nav Toolbar
                with gr.Row(variant="panel"):
                    btn_root = gr.Button("🏠", size="sm")
                    btn_up = gr.Button("⬆️", size="sm")
                    refresh_btn = gr.Button("🔄", size="sm")
                
                file_view = gr.Markdown()
                
                # Selection & Action Group
                with gr.Group():
                    with gr.Row():
                        btn_open = gr.Button("Open", size="sm")
                        btn_edit = gr.Button("Edit", size="sm")
                        btn_delete = gr.Button("Del", variant="stop", size="sm")
                    item_selector = gr.Dropdown(label="Select Item", choices=[], interactive=True)
                
                # Creation / Clone Tabs
                with gr.Accordion("Create / Clone", open=False):
                    with gr.Tabs():
                        with gr.Tab("New"):
                            new_name_box = gr.Textbox(show_label=False, placeholder="Name...")
                            with gr.Row():
                                btn_new_file = gr.Button("File", size="sm")
                                btn_new_dir = gr.Button("Folder", size="sm")
                        with gr.Tab("Clone"):
                            repo_url = gr.Textbox(show_label=False, placeholder="https://github.com/...")
                            clone_btn = gr.Button("Clone", size="sm")

            # Center: Chat/Editor
            with gr.Column(scale=3):
                gr.Markdown("# Local Code Agent")
                with gr.Tabs():
                    with gr.Tab("Chat"):
                        chatbot = gr.Chatbot(height=450)
                        with gr.Group(visible=False) as approval_group:
                            gr.Markdown("### ⚠️ Approve Tool Execution?")
                            tool_details = gr.Code(language="json")
                            with gr.Row():
                                btn_approve = gr.Button("✅ Approve")
                                btn_deny = gr.Button("❌ Deny")
                        msg_input = gr.Textbox(placeholder="Input...", label="User Input")
                    
                    with gr.Tab("Editor"):
                        # Changing Ed path to Dropdown for easier file selection
                        ed_path = gr.Dropdown(label="File", allow_custom_value=True)
                        with gr.Row():
                            ed_load = gr.Button("Load")
                            ed_save = gr.Button("Save")
                        ed_content = gr.Code(language="python", lines=25)
                        ed_status = gr.Markdown()

            # Right: Settings
            with gr.Column(scale=1):
                gr.Markdown("## Settings")
                avail = client.fetch_models()
                model_dd = gr.Dropdown(choices=avail, value=client.model, label="Model", allow_custom_value=True)
                
                with gr.Accordion("System Prompt", open=False):
                    saved = get_saved_prompts()
                    keys = list(saved.keys())
                    cust_keys = [k for k in keys if k != "Default"]
                    
                    prompt_sel = gr.Dropdown(choices=keys, value="Default", label="Load")
                    sys_box = gr.Textbox(value=client.history[0]["content"], lines=5)
                    sys_upd = gr.Button("Update")
                    
                    with gr.Tab("Save"):
                        s_name = gr.Textbox(label="Name")
                        s_btn = gr.Button("Save")
                    with gr.Tab("Delete"):
                        d_sel = gr.Dropdown(choices=cust_keys, label="Prompt")
                        d_btn = gr.Button("Delete", variant="stop")
                    with gr.Tab("Import"):
                        imp_file = gr.File(label="Card")
                        imp_btn = gr.Button("Import")
                    sys_stat = gr.Markdown()

                with gr.Accordion("MCP Servers", open=False):
                    gr.Markdown("Manage MCP servers via `mcp_servers.json` in the static directory.")
                    mcp_list = gr.JSON(value=mcp_mgr.load_config(), label="Configuration")
                    
                    with gr.Row():
                        mcp_upload = gr.File(label="Upload mcp_servers.json", file_types=[".json"])
                        mcp_reload = gr.Button("Reload Config & Restart")
                    
                    mcp_stat = gr.Markdown()

                with gr.Accordion("Memory", open=False):
                    mem_view = gr.JSON(label="Current Memory")
                    mem_ref = gr.Button("Refresh View", size="sm")
                    
                    gr.Markdown("### Edit Memory")
                    mem_key_box = gr.Textbox(label="Key")
                    mem_val_box = gr.Textbox(label="Value", lines=2)
                    with gr.Row():
                        mem_add_btn = gr.Button("Save")
                        mem_del_btn = gr.Button("Delete Key", variant="stop")
                    mem_stat = gr.Markdown()

        # --- UI Logic ---
        # Explorer Helpers
        # Hardened path resolution to prevent directory traversal
        def resolve_path(rel, target):
            SCRIPT_ROOT = os.path.abspath(".")
            
            # If input is absolute path, reject if outside root
            if os.path.isabs(target):
                normalized = os.path.abspath(target)
                if not normalized.startswith(SCRIPT_ROOT):
                    return SCRIPT_ROOT, "."
                return normalized, os.path.relpath(normalized, SCRIPT_ROOT)
            
            # If relative, join and normalize
            abs_p = os.path.abspath(os.path.join(SCRIPT_ROOT, rel, target))
            
            # Security check
            if not abs_p.startswith(SCRIPT_ROOT):
                return SCRIPT_ROOT, "."
            
            return abs_p, os.path.relpath(abs_p, SCRIPT_ROOT)

        def update_exp(pwd):
            client.cwd = pwd if pwd else "."
            path_val = (pwd + "/") if pwd else ""
            abs_p = os.path.abspath(os.path.join(".", pwd))
            # Ensure ed_path choices are cleared if error
            if not os.path.exists(abs_p): 
                return "Error", gr.update(choices=[]), gr.update(choices=[])
            
            items = []
            files_only = []
            for f in os.listdir(abs_p):
                if f not in IGNORE_DIRS:
                    full = os.path.join(abs_p, f)
                    is_d = os.path.isdir(full)
                    items.append((f + "/", "dir") if is_d else (f, "file"))
                    if not is_d:
                        files_only.append(f)
            
            items.sort(key=lambda x: (x[1]!="dir", x[0]))
            files_only.sort()
            
            md = f"### 📂 ./{pwd}\n" + ("" if items else "*Empty*")
            for n, t in items: md += f" {'📁' if t=='dir' else '📄'} {n}\n"
            
            # Return: Markdown, Item Selector Update, Editor Dropdown Update
            return md, gr.update(choices=[x[0].rstrip('/') for x in items], value=None), gr.update(choices=files_only, value=None)

        # Explorer Events
        btn_root.click(lambda: "", None, state_pwd).then(update_exp, state_pwd, [file_view, item_selector, ed_path])
        btn_up.click(lambda p: os.path.dirname(p) if p and p!="." else "", state_pwd, state_pwd).then(update_exp, state_pwd, [file_view, item_selector, ed_path])
        refresh_btn.click(update_exp, state_pwd, [file_view, item_selector, ed_path])
        
        def on_open(pwd, sel):
            if not sel: return pwd
            ap, rp = resolve_path(pwd, sel)
            return rp if os.path.isdir(ap) else pwd
        btn_open.click(on_open, [state_pwd, item_selector], state_pwd).then(update_exp, state_pwd, [file_view, item_selector, ed_path])

        def on_del(pwd, sel):
            if not sel: return pwd
            ap, _ = resolve_path(pwd, sel)
            if os.path.exists(ap): 
                if os.path.basename(ap) in PROTECTED_FILES: return pwd
                if os.path.isdir(ap): shutil.rmtree(ap)
                else: os.remove(ap)
            return pwd
        btn_delete.click(on_del, [state_pwd, item_selector], state_pwd).then(update_exp, state_pwd, [file_view, item_selector, ed_path])

        def on_create(pwd, n, is_d):
            if not n: return pwd
            ap, _ = resolve_path(pwd, n)
            if is_d: os.makedirs(ap, exist_ok=True)
            else: open(ap, 'w').close()
            return pwd
        btn_new_file.click(lambda p,n: on_create(p,n,False), [state_pwd, new_name_box], state_pwd).then(update_exp, state_pwd, [file_view, item_selector, ed_path])
        btn_new_dir.click(lambda p,n: on_create(p,n,True), [state_pwd, new_name_box], state_pwd).then(update_exp, state_pwd, [file_view, item_selector, ed_path])

        def to_editor(pwd, sel):
            if not sel: return gr.update(), gr.update(), "No select"
            ap, _ = resolve_path(pwd, sel)
            if os.path.isfile(ap): return ap, read_file(ap), "Loaded"
            return gr.update(), gr.update(), "Not file"
        btn_edit.click(to_editor, [state_pwd, item_selector], [ed_path, ed_content, ed_status])

        # Editor Logic Updated for Dropdown
        def do_ed_load(pwd, filename):
            if not filename: return "", "No file selected"
            # Handle both direct filename selection (relative to pwd) and custom path input
            path = os.path.join(pwd, filename) if pwd else filename
            # We rely on read_file's internal checks, but can pre-check existence
            ap, _ = resolve_path(pwd, filename)
            if os.path.isfile(ap):
                return read_file(ap), f"Loaded {filename}"
            return "", f"Error: {filename} not found"

        ed_load.click(do_ed_load, [state_pwd, ed_path], [ed_content, ed_status])
        
        def do_ed_save(pwd, filename, content):
            if not filename: return "No file selected"
            path = os.path.join(pwd, filename) if pwd else filename
            # resolve_path check is good practice before write_file
            ap, _ = resolve_path(pwd, filename)
            return write_file(ap, content)
            
        ed_save.click(do_ed_save, [state_pwd, ed_path, ed_content], ed_status)

        # Settings
        model_dd.change(lambda m: setattr(client, 'model', m), model_dd)
        
        def load_p(k): return get_saved_prompts().get(k, "")
        prompt_sel.change(load_p, prompt_sel, sys_box)
        
        def save_p(n, c):
            ok, m = save_custom_prompt(n, c)
            d = get_saved_prompts()
            return gr.update(choices=list(d.keys()), value=n), gr.update(choices=[k for k in d if k!="Default"]), m
        s_btn.click(save_p, [s_name, sys_box], [prompt_sel, d_sel, sys_stat])
        
        def del_p(n):
            ok, m = delete_custom_prompt(n)
            d = get_saved_prompts()
            return gr.update(choices=list(d.keys())), gr.update(choices=[k for k in d if k!="Default"], value=None), m
        d_btn.click(del_p, d_sel, [prompt_sel, d_sel, sys_stat])
        
        def imp_c(f):
            if not f: return gr.update(), "No file"
            # Gradio compatibility: f might be obj or string
            p = f.name if hasattr(f, 'name') else f
            r = parse_character_card(p)
            return (gr.update(), r) if r.startswith("Error") else (r, "Imported")
        imp_btn.click(imp_c, imp_file, [sys_box, sys_stat])
        
        sys_upd.click(lambda p: client.update_system_prompt(p), sys_box, sys_stat)
        
        # Memory Handlers
        mem_ref.click(lambda: retrieve_info(), None, mem_view)
        
        def do_mem_add(k, v):
            res = remember_info(k, v)
            return res, retrieve_info()
        mem_add_btn.click(do_mem_add, [mem_key_box, mem_val_box], [mem_stat, mem_view])
        
        def do_mem_del(k):
            res = forget_info(k)
            return res, retrieve_info()
        mem_del_btn.click(do_mem_del, mem_key_box, [mem_stat, mem_view])

        # MCP Handlers
        def handle_mcp_upload(file_obj):
            if not file_obj: return None, "No file."
            try:
                # Gradio compatibility: file_obj might be obj or string
                path = file_obj.name if hasattr(file_obj, 'name') else file_obj
                
                # Load new config
                with open(path, 'r') as f: new_cfg = json.load(f)
                
                # Load current config
                current_cfg = {}
                if os.path.exists(PATHS["mcp"]):
                    try:
                        with open(PATHS["mcp"], 'r') as f: current_cfg = json.load(f)
                    except: pass
                
                # Merge
                current_cfg.update(new_cfg)
                
                # Save
                with open(PATHS["mcp"], 'w') as f: json.dump(current_cfg, f, indent=2)
                
                # Restart
                cfg = mcp_mgr.restart_all()
                return cfg, f"Merged {len(new_cfg)} servers. Total: {len(cfg)}"
            except Exception as e: return None, f"Error: {e}"

        mcp_upload.upload(handle_mcp_upload, mcp_upload, [mcp_list, mcp_stat])
        mcp_reload.click(lambda: (mcp_mgr.restart_all(), "Reloaded"), None, [mcp_list, mcp_stat])

        # Chat
        def render_chat(h):
            pairs = []
            u, b = None, []
            processed_tool_outputs = set()

            for i, m in enumerate(h):
                role = m['role']
                content = m.get('content', '')

                if role == 'user':
                    if u: pairs.append([u, "\n".join(b)])
                    u, b = content, []
                
                elif role == 'assistant':
                    if m.get('tool_calls'):
                        for t in m['tool_calls']:
                            fname = t['function']['name']
                            args_str = t['function']['arguments']
                            call_id = t['id']
                            
                            # Find output
                            output_content = None
                            for next_m in h[i+1:]:
                                if next_m.get('role') == 'tool' and next_m.get('tool_call_id') == call_id:
                                    output_content = next_m.get('content', '')
                                    processed_tool_outputs.add(call_id)
                                    break
                            
                            # Header styling
                            status = "✅" if output_content is not None else "⏳"
                            summary_text = f"🛠️ {fname}"
                            
                            # Special handling for thinking to make header nicer
                            if fname == 'sequential_thinking':
                                try:
                                    args = json.loads(args_str)
                                    step = args.get('step', '?')
                                    total = args.get('total_steps', '?')
                                    summary_text = f"💭 Thinking (Step {step}/{total})"
                                except: pass
                            
                            content_body = f"**Args:**\n```json\n{args_str}\n```"
                            if output_content is not None:
                                content_body += f"\n**Output:**\n```\n{output_content}\n```"
                            else:
                                content_body += "\n*(Pending Approval)*"

                            b.append(f"<details><summary>{summary_text} {status}</summary>\n\n{content_body}\n</details>")
                    else:
                        b.append(content)
                
                elif role == 'tool':
                    call_id = m.get('tool_call_id')
                    if call_id not in processed_tool_outputs:
                        b.append(f"<details><summary>🔍 Tool Output (Orphan)</summary>\n\n```\n{content}\n```\n</details>")

            if u: pairs.append([u, "\n".join(b)])
            return pairs

        def usr(m):
            client.history.append({"role": "user", "content": m})
            return "", render_chat(client.history)
        
        def bot(pend):
            txt, tools = client.chat_step_1_send()
            if txt: 
                return render_chat(client.history), gr.update(visible=False), None, None
            if tools:
                # Check for run_command with dangerous commands
                should_auto_approve = True
                for t in tools:
                    if t['function']['name'] == "run_command":
                        try:
                            args = json.loads(t['function']['arguments'])
                            cmd = args.get('command', '')
                            is_dangerous = any(d in cmd for d in DANGEROUS_COMMANDS) or "sudo" in cmd
                            if is_dangerous:
                                should_auto_approve = False
                                break
                        except:
                            should_auto_approve = False
                            break
                
                if should_auto_approve:
                    client.chat_step_2_execute(tools, approved=True)
                    return bot(None)
                
                return render_chat(client.history), gr.update(visible=True), json.dumps([t['function'] for t in tools], indent=2), tools
            return render_chat(client.history), gr.update(visible=False), None, None

        def exe(t):
            client.chat_step_2_execute(t, True)
            return bot(None)
        
        def den():
            client.history.append({"role": "user", "content": "Denied."})
            return bot(None)

        msg_input.submit(usr, msg_input, [msg_input, chatbot]).then(bot, state_pending_tools, [chatbot, approval_group, tool_details, state_pending_tools]).then(update_exp, state_pwd, [file_view, item_selector, ed_path])
        btn_approve.click(exe, state_pending_tools, [chatbot, approval_group, tool_details, state_pending_tools]).then(update_exp, state_pwd, [file_view, item_selector, ed_path])
        btn_deny.click(den, None, [chatbot, approval_group, tool_details, state_pending_tools])

        demo.load(update_exp, state_pwd, [file_view, item_selector, ed_path])

    demo.launch(share=False)

# --- Main CLI ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_API_BASE)
    parser.add_argument("--key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--webui", action="store_true")
    parser.add_argument("--safemode", action="store_true")
    parser.add_argument("--static", default=DEFAULT_STATIC_DIR, help="Directory for persistent agent data")
    args = parser.parse_args()

    # Setup Static Directory
    static_dir = os.path.abspath(args.static)
    os.makedirs(static_dir, exist_ok=True)
    
    PATHS["static"] = static_dir
    PATHS["memory"] = os.path.join(static_dir, ".agent_memory.json")
    PATHS["prompts"] = os.path.join(static_dir, "saved_prompts.json")
    PATHS["mcp"] = os.path.join(static_dir, "mcp_servers.json")
    
    PROTECTED_FILES.update({
        PATHS["memory"], os.path.basename(PATHS["memory"]),
        PATHS["prompts"], os.path.basename(PATHS["prompts"]),
        PATHS["mcp"], os.path.basename(PATHS["mcp"])
    })

    # Start MCP Manager
    mcp_mgr = MCPManager()
    mcp_mgr.start_all()

    client = AgentClient(args.url, args.key, args.model, SafetyManager(args.safemode), mcp_mgr)

    if args.webui or DEFAULT_WEBUI:
        launch_webui(client, mcp_mgr)
    else:
        print(f"{Colors.HEADER}--- Local Code Agent (CLI) ---{Colors.ENDC}")
        print(f"Static Dir: {static_dir}")
        print(f"MCP Servers: {list(mcp_mgr.clients.keys())}")
        
        while True:
            try:
                u_in = input(f"\n{Colors.GREEN}>> {Colors.ENDC}")
                if u_in.lower() in ["/exit", "quit"]: break
                
                txt, tools = client.chat_step_1_send(u_in)
                while tools or (not txt and not tools):
                    if tools:
                        print(f"{Colors.CYAN}Tools: {[t['function']['name'] for t in tools]}{Colors.ENDC}")
                        client.chat_step_2_execute(tools)
                    txt, tools = client.chat_step_1_send()
                    if txt: print(f"{Colors.BLUE}Agent:{Colors.ENDC} {txt}")
                    if not txt and not tools: break
            except KeyboardInterrupt: break

if __name__ == "__main__":
    main()