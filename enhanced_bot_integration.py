#!/usr/bin/env python3
"""
🚀 Enhanced Bot Integration
Integrates all new features into LLMCord: HUD, status, state persistence, etc.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from discord.ext import commands
import yaml

# Import all our new systems
from live_parameter_hud import setup_live_hud
from agent_status_command import setup_status_command
from dev_thread_manager import setup_dev_thread_manager
from mcp_integration import MCPIntegration, setup_mcp_commands
from hidden_reflection_display import setup_hidden_reflection_monitoring
from state_persistence import setup_state_persistence, shutdown_state_persistence


class EnhancedBotManager:
    """Manages all enhanced bot features"""
    
    def __init__(self, bot: commands.Bot, config: Dict[str, Any]):
        self.bot = bot
        self.config = config
        self.logger = logging.getLogger("EnhancedBot")
        
        # Feature instances
        self.live_hud: Optional[Any] = None
        self.status_command: Optional[Any] = None
        self.thread_manager: Optional[Any] = None
        self.mcp_integration: Optional[MCPIntegration] = None
        self.reflection_display: Optional[Any] = None
        
        # Agent tracking for state persistence
        self.agents: Dict[str, Any] = {}
    
    async def initialize_all_features(self):
        """Initialize all enhanced features"""
        
        self.logger.info("🚀 Initializing enhanced bot features...")
        
        try:
            # 1. Setup Live Parameter HUD (auto-start)
            hud_config = self._get_hud_config()
            if hud_config:
                self.live_hud = await setup_live_hud(self.bot, hud_config)
                self.logger.info("✅ Live Parameter HUD configured")
            
            # 2. Setup Agent Status Command
            self.status_command = await setup_status_command(self.bot)
            self.logger.info("✅ Agent Status Command registered")
            
            # 3. Setup Dev Thread Manager
            self.thread_manager = await setup_dev_thread_manager(self.bot)
            self.logger.info("✅ Dev Thread Manager initialized")
            
            # 4. Setup MCP Integration
            mcp_config_path = self.config.get('mcp_config_path', 'mcp_config.json')
            self.mcp_integration = MCPIntegration(mcp_config_path)
            await self.mcp_integration.initialize()
            await setup_mcp_commands(self.bot, self.mcp_integration)
            self.logger.info("✅ MCP Integration initialized")
            
            # 5. Setup Hidden Reflection Monitoring
            self.reflection_display = await setup_hidden_reflection_monitoring(self.bot, self.config)
            self.logger.info("✅ Hidden Reflection Display configured")
            
            # 6. Setup State Persistence (will be called later with agents)
            self.logger.info("✅ State Persistence ready")
            
            self.logger.info("🎉 All enhanced features initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing features: {e}")
            raise
    
    def _get_hud_config(self) -> Optional[Dict[str, Any]]:
        """Get HUD configuration from bot config"""
        
        hud_config = self.config.get('live_hud', {})
        
        if not hud_config.get('enabled', True):
            return None
        
        # Default HUD configuration
        default_config = {
            'channel_id': None,
            'update_interval': 30,
            'compact_mode': False
        }
        
        # Override with config values
        for key in default_config:
            if key in hud_config:
                default_config[key] = hud_config[key]
        
        # Try to get channel ID from various sources
        if not default_config['channel_id']:
            # Try status channel first
            if 'status_channel_id' in self.config:
                default_config['channel_id'] = self.config['status_channel_id']
            # Try monitoring channel
            elif 'monitoring_channel_id' in self.config:
                default_config['channel_id'] = self.config['monitoring_channel_id']
            # Try general channel
            elif 'general_channel_id' in self.config:
                default_config['channel_id'] = self.config['general_channel_id']
        
        if default_config['channel_id']:
            return default_config
        else:
            self.logger.warning("No HUD channel configured - HUD will not auto-start")
            return None
    
    async def register_agents(self, agents: Dict[str, Any]):
        """Register agents for monitoring and state persistence"""
        
        self.agents = agents
        self.logger.info(f"Registered {len(agents)} agents for enhanced features")
        
        # Setup state persistence
        await setup_state_persistence(agents)
        
        # Register agents with evolution scheduler
        from evolution.evolution_scheduler import evolution_scheduler
        for agent_id, runtime in agents.items():
            evolution_scheduler.register_agent(agent_id, runtime)
    
    async def shutdown(self):
        """Shutdown all enhanced features"""
        
        self.logger.info("🔄 Shutting down enhanced features...")
        
        try:
            # Stop HUD
            if self.live_hud and self.live_hud.running:
                await self.live_hud.stop_hud()
            
            # Shutdown MCP
            if self.mcp_integration:
                await self.mcp_integration.shutdown()
            
            # Shutdown state persistence (saves final state)
            await shutdown_state_persistence()
            
            # Stop evolution scheduler
            from evolution.evolution_scheduler import evolution_scheduler
            evolution_scheduler.stop_evolution_cycles()
            
            self.logger.info("✅ Enhanced features shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")


def load_enhanced_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """Load enhanced configuration with defaults"""
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}
    
    # Enhanced feature defaults
    enhanced_defaults = {
        'live_hud': {
            'enabled': True,
            'channel_id': None,  # Will try to auto-detect
            'update_interval': 30,
            'compact_mode': False
        },
        'mcp_config_path': 'mcp_config.json',
        'monitoring_channel_id': None,
        'state_persistence': {
            'enabled': True,
            'data_dir': '/tmp/llmcord_state',
            'auto_save_interval': 5  # minutes
        }
    }
    
    # Merge defaults with existing config
    for key, default_value in enhanced_defaults.items():
        if key not in config:
            config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(config[key], dict):
            # Merge nested dictionaries
            for subkey, subvalue in default_value.items():
                if subkey not in config[key]:
                    config[key][subkey] = subvalue
    
    return config


async def setup_enhanced_bot(bot: commands.Bot, config_file: str = "config.yaml") -> EnhancedBotManager:
    """Setup all enhanced bot features"""
    
    # Load enhanced configuration
    config = load_enhanced_config(config_file)
    
    # Create manager
    manager = EnhancedBotManager(bot, config)
    
    # Initialize all features
    await manager.initialize_all_features()
    
    return manager


# Integration hooks for existing bot code
async def on_bot_ready_enhanced(manager: EnhancedBotManager):
    """Call this when the bot is ready"""
    
    # Start evolution system
    from evolution.evolution_scheduler import start_agent_evolution
    start_agent_evolution()
    
    logging.info("🤖 Enhanced bot is ready!")


async def on_bot_shutdown_enhanced(manager: EnhancedBotManager):
    """Call this when the bot is shutting down"""
    
    await manager.shutdown()


# Example integration code for llmcord_multimodel.py:
"""
# Add to imports
from enhanced_bot_integration import setup_enhanced_bot, on_bot_ready_enhanced, on_bot_shutdown_enhanced

# Add global variable
enhanced_manager = None

# Add to on_ready event
@discord_bot.event
async def on_ready():
    global enhanced_manager
    
    # ... existing on_ready code ...
    
    # Setup enhanced features
    enhanced_manager = await setup_enhanced_bot(discord_bot)
    await enhanced_manager.register_agents(virtual_users)  # or whatever your agents dict is called
    await on_bot_ready_enhanced(enhanced_manager)

# Add shutdown handler (you might need to create this)
import signal
import sys

def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    if enhanced_manager:
        asyncio.run(on_bot_shutdown_enhanced(enhanced_manager))
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
"""