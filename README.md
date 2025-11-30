# Local Code Agent

**An autonomous, local pair-programmer that bridges the gap between your LLM and your operating system.**

This script (`local_code_agent.py`) creates a standalone AI developer that runs entirely on your local machine. It connects to local LLM servers (like LM Studio or Ollama) and acts as a bridge, giving the AI "hands" to edit files, run terminal commands, and manage projects safely.

---

## 🌟 Core Features

* **Dual Interface:**
    * **CLI Mode:** Fast, terminal-based interaction with color-coded outputs.
    * **Web UI (Gradio):** A "Hacker" themed graphical interface (Black/Aqua) with a built-in file explorer, code editor, and visual chat.
* **Safety First:**
    * **Visual Diffs:** You see exactly what will change before it happens. #TODO / Broken / Implemented Wrong
    * **Auto-Backups:** Every file edit creates a `.old` backup.
    * **Action Approval:** Risky commands (like `rm` or `sudo`) require explicit user permission. (usually)
* **Smart Memory:**
    * **Context Management:** Automatically handles long conversations by pruning history while keeping the system prompt.
    * **Project Memory:** Persists information in `.agent_memory.json` to remember details across sessions.
* **Persona Support:** Import "SillyTavern" character cards to give the agent a specific personality (e.g., "Strict Linter" or "Cyberpunk Netrunner").

---

## 🛠️ Tools Available to the AI

The agent can utilize the following tools to interact with your system:

* `read_file(path)`: Reads file content (capped at 100KB).
* `write_file(path, content)`: Writes/Overwrites files (auto-backups enabled).
* `replace_in_file(path, old, new)`: Exact string replacement (auto-backups enabled).
* `list_files(path)`: Lists files and directories.
* `search_code(directory, term)`: Grep-style recursive search.
* `run_command(command)`: Executes shell commands (subject to safety checks).
* `clone_repo(url)`: Git clone wrapper.
* `lint_file(path)`: Python syntax checker.
* `sequential_thinking`: Forces the model to plan step-by-step before acting.
* `remember_info` / `retrieve_info`: Long-term memory storage.
*
* `MCP Importing` : <-

---

## 📦 Installation

### 1. Prerequisites
* Python 3.10 or higher.
* Git.

### 2. Setup Script
```
# Clone the repository
git clone [https://github.com/NoxCreations/local-code-agent.git](https://github.com/NoxCreations/local-code-agent.git)
cd local-code-agent

# Install Python dependencies
pip install -r requirements.txt
```

🔌 Connecting a Backend (LLM Provider)

You must run a local LLM server (recommended) or use another openai-compatible API endpoint for the agent to function. Choose ONE of the following options for self hosting your ai:

Option A: LM Studio (Recommended)

Best for: Windows/Mac users, ease of use, and visual management.

    Download LM Studio.

    Model Selection: Search for and download an "Abliterated" model (e.g., gemma3-4b-it-abliterated or Llama-3-8B-Abliterated).

        Why Abliterated? Standard models often refuse to edit files or run commands due to safety filters. Abliterated models have these refusals removed, making them obedient coding assistants. And also memes.

    Server Config:

        Go to the Local Server tab (the <-> icon).

        Select your model from the top dropdown.

        Port: Set to 1234.

        CORS: Check "Enable Cross-Origin-Resource-Sharing" (Required for Web UI).

        Context Window: Set to 8192 or higher.

        GPU Offload: Maximize this slider.

    Click Start Server.
    No endpoint change in script, its designed to use LM Studio as a backend.

Option B: Ollama

Best for: Linux users or command-line preference.

    Install Ollama.

    Pull a coding-capable model:
    
    ollama pull qwen2.5-coder:7b

    Configuration:

        Ollama runs on port 11434 by default.

        Edit local_code_agent.py to change API_BASE_URL to http://localhost:11434/v1.

Option C: LocalAI

Best for: Docker enthusiasts and CPU-only setups.

    Run LocalAI via Docker:
    Bash

    docker run -p 8080:8080 --name local-ai -ti localai/localai:latest-aio-cpu

    Configuration:

        Edit local_code_agent.py to change API_BASE_URL to http://localhost:8080/v1.

🚀 How to Run

    Ensure your LLM backend is running.

    Run the agent:

```python local_code_agent.py```

OR

```python local_code_agent.py --webui```



    Interfaces:

        The terminal will show the CLI prompt immediately.

        A local URL (e.g., http://127.0.0.1:7860) will be displayed to access the Web UI.

## 🌐 Advanced: Docker + Tailscale/Headscale

**Scenario:** You want the **Agent** running in a secure container, while the heavy **LLM** runs on your powerful local host machine. You also want to access this agent remotely.

### 1. The Architecture
* **Host Machine:** Runs LM Studio/Ollama (GPU accelerated).
* **Container:** Runs `local_code_agent.py` + Tailscale.
* **Networking:**
    * Container talks to Host LLM via `host.docker.internal`.
    * User talks to Container via Tailscale VPN.

### 2. Dockerfile
Create a file named `Dockerfile` in the project root:

```
FROM python:3.10-slim

# Install system tools and Tailscale
RUN apt-get update && apt-get install -y curl git
RUN curl -fsSL [https://tailscale.com/install.sh](https://tailscale.com/install.sh) | sh

WORKDIR /app

# Install Python Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Code
COPY . .

# Expose Gradio Port
EXPOSE 7860

# Define the entrypoint script to handle Tailscale + Python
COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
```

3. Start Script (start.sh)

Create a file named start.sh:

```
#!/bin/bash

# Start Tailscale in userspace mode if AUTH KEY is provided
if [ ! -z "$TS_AUTHKEY" ]; then
    echo "Starting Tailscale..."
    tailscaled --tun=userspace-networking --socks5-server=localhost:1055 &
    tailscale up --authkey=$TS_AUTHKEY --hostname=code-agent-docker
    echo "Tailscale up."
fi

# Run the Agent
# Note: Ensure local_code_agent.py is configured to point to '[http://host.docker.internal:1234/v1](http://host.docker.internal:1234/v1)'
python local_code_agent.py
```

4. Running the Container

Generate an Auth Key from your Tailscale/Headscale admin console.
Bash

docker build -t local-code-agent .

```
docker run -d \
  --name code-agent \
  --add-host=host.docker.internal:host-gateway \
  -e TS_AUTHKEY=tskey-auth-YOUR-KEY-HERE \
  local-code-agent
```

⚠️ OPTIONAL: Public Exposure via VPS (OVH/IONOS)

⛔ NOT RECOMMENDED ⛔

If you wish to expose the Web UI to the open internet (without requiring a VPN client on your accessing device), you can use a cheap VPS (e.g., OVHcloud or IONOS) as a relay.

The Setup:

    VPS: Purchase a cheap VPS ($3-5/mo) and install the Tailscale/Headscale client on it.

    Connect: Join the VPS to the same Tailnet as your Docker Container.

    Reverse Proxy: Install Nginx or Caddy (recommended) on the VPS to forward public traffic (Port 80/443) to the Docker Container's Tailscale IP (Port 7860).

Why this is dangerous:

    Remote Code Execution: This agent allows editing files and running shell commands.

    Zero Security: If you expose this to the public internet without adding your own robust authentication layer (like Authelia or basic auth in Nginx), anyone who finds the URL can wipe your files or install malware on your container. Absolutely do NOT expose it to the internet if the client isn't running in docker or a disposable vps, it could get tragic.

    Use Tailscale Private Access instead. It is significantly safer to only access the agent while connected to your VPN.


Notes:

    Networking: --add-host=host.docker.internal:host-gateway is crucial. It allows the container to talk to LM Studio running on your physical machine.

    Access: Once running, check your Tailscale dashboard for the IP of code-agent-docker. Access the Web UI at http://[TAILSCALE_IP]:7860.
