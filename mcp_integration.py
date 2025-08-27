#!/usr/bin/env python3
"""
🔌 MCP Integration for LLMCord
Supports Model Context Protocol for enhanced LLM capabilities
"""

import json
import asyncio
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import shutil
import yaml

@dataclass
class MCPServer:
    """Configuration for an MCP server"""
    name: str
    command: List[str]
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    timeout: int = 30
    auto_restart: bool = True
    
    # Runtime state
    process: Optional[asyncio.subprocess.Process] = None
    last_start_time: float = 0
    restart_count: int = 0

@dataclass 
class MCPTool:
    """An MCP tool/capability"""
    name: str
    description: str
    parameters: Dict[str, Any]
    server_name: str

class MCPManager:
    """Manages MCP servers and tool integration"""
    
    def __init__(self, config_path: str = "mcp_config.json"):
        self.config_path = config_path
        self.servers: Dict[str, MCPServer] = {}
        self.tools: Dict[str, MCPTool] = {}  # tool_name -> MCPTool
        self.logger = logging.getLogger("MCP.Manager")
        
        # Load configuration
        self.load_config()
    
    def load_config(self):
        """Load MCP configuration from file"""
        
        if not os.path.exists(self.config_path):
            # Create default config if it doesn't exist
            self.create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Parse server configurations
            for server_name, server_config in config.get('servers', {}).items():
                self.servers[server_name] = MCPServer(
                    name=server_name,
                    command=server_config['command'],
                    args=server_config.get('args', []),
                    env=server_config.get('env', {}),
                    working_dir=server_config.get('working_dir'),
                    timeout=server_config.get('timeout', 30),
                    auto_restart=server_config.get('auto_restart', True)
                )
            
            self.logger.info(f"Loaded {len(self.servers)} MCP servers from config")
            
        except Exception as e:
            self.logger.error(f"Failed to load MCP config: {e}")
            self.create_default_config()
    
    def create_default_config(self):
        """Create a default MCP configuration"""
        
        default_config = {
            "servers": {
                "github": {
                    "command": ["npx", "@modelcontextprotocol/server-github"],
                    "env": {
                        "GITHUB_PERSONAL_ACCESS_TOKEN": ""
                    },
                    "description": "GitHub repository access - clone repos and read files"
                },
                "filesystem": {
                    "command": ["npx", "@modelcontextprotocol/server-filesystem"],
                    "args": ["/tmp/mcp-workspace"],
                    "description": "Local filesystem access for workspace"
                },
                "sqlite": {
                    "command": ["npx", "@modelcontextprotocol/server-sqlite"],
                    "args": ["/tmp/databases"],
                    "description": "SQLite database access"
                }
            },
            "settings": {
                "workspace_dir": "/tmp/mcp-workspace",
                "auto_start_servers": True,
                "tool_timeout": 60
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        self.logger.info(f"Created default MCP config at {self.config_path}")
    
    async def start_server(self, server_name: str) -> bool:
        """Start an MCP server"""
        
        if server_name not in self.servers:
            self.logger.error(f"Unknown MCP server: {server_name}")
            return False
        
        server = self.servers[server_name]
        
        if server.process and server.process.returncode is None:
            self.logger.info(f"MCP server {server_name} already running")
            return True
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(server.env)
            
            # Start the server process
            cmd = server.command + server.args
            
            server.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=server.working_dir
            )
            
            server.last_start_time = asyncio.get_event_loop().time()
            
            self.logger.info(f"Started MCP server: {server_name}")
            
            # Discover tools from this server
            await self.discover_tools(server_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start MCP server {server_name}: {e}")
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """Stop an MCP server"""
        
        if server_name not in self.servers:
            return False
        
        server = self.servers[server_name]
        
        if server.process:
            try:
                server.process.terminate()
                await asyncio.wait_for(server.process.wait(), timeout=5.0)
                self.logger.info(f"Stopped MCP server: {server_name}")
                return True
            except asyncio.TimeoutError:
                server.process.kill()
                self.logger.warning(f"Force killed MCP server: {server_name}")
                return True
            except Exception as e:
                self.logger.error(f"Error stopping MCP server {server_name}: {e}")
        
        return False
    
    async def discover_tools(self, server_name: str):
        """Discover available tools from an MCP server"""
        
        if server_name not in self.servers:
            return
        
        server = self.servers[server_name]
        
        try:
            # Send tools/list request to the server
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }
            
            request_json = json.dumps(request) + "\n"
            
            if server.process and server.process.stdin:
                server.process.stdin.write(request_json.encode())
                await server.process.stdin.drain()
                
                # Read response
                response_line = await asyncio.wait_for(
                    server.process.stdout.readline(),
                    timeout=server.timeout
                )
                
                response = json.loads(response_line.decode())
                
                if 'result' in response and 'tools' in response['result']:
                    # Register discovered tools
                    for tool_info in response['result']['tools']:
                        tool = MCPTool(
                            name=tool_info['name'],
                            description=tool_info.get('description', ''),
                            parameters=tool_info.get('inputSchema', {}),
                            server_name=server_name
                        )
                        
                        self.tools[tool.name] = tool
                    
                    self.logger.info(f"Discovered {len(response['result']['tools'])} tools from {server_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to discover tools from {server_name}: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call an MCP tool"""
        
        if tool_name not in self.tools:
            self.logger.error(f"Unknown MCP tool: {tool_name}")
            return None
        
        tool = self.tools[tool_name]
        server = self.servers[tool.server_name]
        
        if not server.process or server.process.returncode is not None:
            # Try to restart server
            if not await self.start_server(tool.server_name):
                return None
        
        try:
            # Send tool call request
            request = {
                "jsonrpc": "2.0",
                "id": asyncio.get_event_loop().time(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            request_json = json.dumps(request) + "\n"
            
            if server.process and server.process.stdin:
                server.process.stdin.write(request_json.encode())
                await server.process.stdin.drain()
                
                # Read response
                response_line = await asyncio.wait_for(
                    server.process.stdout.readline(),
                    timeout=server.timeout
                )
                
                response = json.loads(response_line.decode())
                
                if 'result' in response:
                    self.logger.info(f"Successfully called MCP tool: {tool_name}")
                    return response['result']
                elif 'error' in response:
                    self.logger.error(f"MCP tool error: {response['error']}")
                    return None
                
        except Exception as e:
            self.logger.error(f"Failed to call MCP tool {tool_name}: {e}")
        
        return None
    
    async def setup_github_workspace(self, repo_url: str, branch: str = "main") -> Optional[str]:
        """Setup a GitHub repository workspace using MCP"""
        
        # Ensure workspace directory exists
        workspace_dir = "/tmp/mcp-workspace"
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Start GitHub MCP server if not running
        if not await self.start_server("github"):
            return None
        
        try:
            # Clone repository using GitHub MCP tool
            result = await self.call_tool("clone_repository", {
                "url": repo_url,
                "branch": branch,
                "path": workspace_dir
            })
            
            if result:
                repo_name = repo_url.split('/')[-1].replace('.git', '')
                repo_path = os.path.join(workspace_dir, repo_name)
                
                self.logger.info(f"Successfully cloned {repo_url} to {repo_path}")
                return repo_path
            
        except Exception as e:
            self.logger.error(f"Failed to setup GitHub workspace: {e}")
        
        return None
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available MCP tools"""
        
        tools_list = []
        
        for tool_name, tool in self.tools.items():
            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "server": tool.server_name
            })
        
        return tools_list
    
    async def start_all_servers(self):
        """Start all configured MCP servers"""
        
        for server_name in self.servers:
            await self.start_server(server_name)
        
        self.logger.info(f"Started {len(self.servers)} MCP servers")
    
    async def stop_all_servers(self):
        """Stop all MCP servers"""
        
        for server_name in self.servers:
            await self.stop_server(server_name)
        
        self.logger.info("Stopped all MCP servers")


class MCPIntegration:
    """Integration layer for MCP with Discord bot"""
    
    def __init__(self, config_path: str = "mcp_config.json"):
        self.mcp_manager = MCPManager(config_path)
        self.logger = logging.getLogger("MCP.Integration")
    
    async def initialize(self):
        """Initialize MCP integration"""
        
        self.logger.info("Initializing MCP integration...")
        
        # Start all MCP servers
        await self.mcp_manager.start_all_servers()
        
        # Log available tools
        tools = await self.mcp_manager.get_available_tools()
        self.logger.info(f"Available MCP tools: {[t['name'] for t in tools]}")
    
    async def process_mcp_command(self, command: str, args: Dict[str, Any]) -> Optional[str]:
        """Process MCP-related commands from Discord"""
        
        if command == "list_mcp_tools":
            tools = await self.mcp_manager.get_available_tools()
            
            if not tools:
                return "🔌 No MCP tools available. Check server configuration."
            
            result = "🔌 **Available MCP Tools:**\n\n"
            
            servers = {}
            for tool in tools:
                server = tool['server']
                if server not in servers:
                    servers[server] = []
                servers[server].append(tool)
            
            for server_name, server_tools in servers.items():
                result += f"**{server_name.title()} Server:**\n"
                for tool in server_tools:
                    result += f"• `{tool['name']}` - {tool['description']}\n"
                result += "\n"
            
            return result
        
        elif command == "clone_repo":
            repo_url = args.get('url', '')
            branch = args.get('branch', 'main')
            
            if not repo_url:
                return "❌ Please provide a repository URL"
            
            result_path = await self.mcp_manager.setup_github_workspace(repo_url, branch)
            
            if result_path:
                return f"✅ Successfully cloned repository to: `{result_path}`\n\nYou can now use filesystem MCP tools to explore the codebase!"
            else:
                return "❌ Failed to clone repository. Check the URL and your GitHub token."
        
        elif command == "mcp_status":
            status_text = "🔌 **MCP Server Status:**\n\n"
            
            for server_name, server in self.mcp_manager.servers.items():
                if server.process and server.process.returncode is None:
                    status = "🟢 Running"
                else:
                    status = "🔴 Stopped"
                
                status_text += f"• **{server_name}**: {status}\n"
            
            return status_text
        
        return None
    
    async def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get MCP tools formatted for LLM function calling"""
        
        tools = await self.mcp_manager.get_available_tools()
        
        llm_tools = []
        for tool in tools:
            llm_tool = {
                "type": "function",
                "function": {
                    "name": f"mcp_{tool['name']}",
                    "description": f"[MCP {tool['server']}] {tool['description']}",
                    "parameters": tool['parameters']
                }
            }
            llm_tools.append(llm_tool)
        
        return llm_tools
    
    async def execute_mcp_function(self, function_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Execute an MCP function called by the LLM"""
        
        # Remove mcp_ prefix
        if function_name.startswith("mcp_"):
            tool_name = function_name[4:]
        else:
            tool_name = function_name
        
        result = await self.mcp_manager.call_tool(tool_name, arguments)
        
        if result:
            # Format result for LLM
            if isinstance(result, dict):
                if 'content' in result:
                    return result['content']
                elif 'text' in result:
                    return result['text']
                else:
                    return json.dumps(result, indent=2)
            else:
                return str(result)
        
        return None
    
    async def shutdown(self):
        """Shutdown MCP integration"""
        
        self.logger.info("Shutting down MCP integration...")
        await self.mcp_manager.stop_all_servers()


# Example MCP configuration for GitHub repository access
def create_github_mcp_config(github_token: str = "") -> Dict[str, Any]:
    """Create MCP configuration for GitHub repository access"""
    
    return {
        "servers": {
            "github": {
                "command": ["npx", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": github_token
                },
                "description": "GitHub repository access - clone repos and read files",
                "auto_restart": True
            },
            "filesystem": {
                "command": ["npx", "@modelcontextprotocol/server-filesystem"],
                "args": ["/tmp/mcp-workspace"],
                "description": "Local filesystem access for cloned repositories",
                "auto_restart": True
            }
        },
        "settings": {
            "workspace_dir": "/tmp/mcp-workspace",
            "auto_start_servers": True,
            "tool_timeout": 60
        }
    }


# Discord command integration
async def setup_mcp_commands(bot, mcp_integration: MCPIntegration):
    """Setup Discord commands for MCP"""
    
    @bot.tree.command(name="mcp-tools", description="List available MCP tools")
    async def list_mcp_tools(interaction):
        result = await mcp_integration.process_mcp_command("list_mcp_tools", {})
        await interaction.response.send_message(result or "No MCP tools available", ephemeral=True)
    
    @bot.tree.command(name="clone-repo", description="Clone a GitHub repository to workspace")
    async def clone_repo(interaction, repo_url: str, branch: str = "main"):
        await interaction.response.defer()
        result = await mcp_integration.process_mcp_command("clone_repo", {
            "url": repo_url,
            "branch": branch
        })
        await interaction.followup.send(result or "Failed to clone repository")
    
    @bot.tree.command(name="mcp-status", description="Check MCP server status")
    async def mcp_status(interaction):
        result = await mcp_integration.process_mcp_command("mcp_status", {})
        await interaction.response.send_message(result or "No MCP servers configured", ephemeral=True)