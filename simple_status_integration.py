#!/usr/bin/env python3
"""
📊 Simple Status Integration
Quick integration of /status command for existing llmcord_multimodel.py
"""

import discord
from discord.ext import commands
from typing import Dict, Any
from datetime import datetime
import logging


class SimpleAgentStatus(commands.Cog):
    """Simple version of agent status for immediate integration"""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.logger = logging.getLogger("SimpleAgentStatus")
        self.virtual_users = {}  # Will be set by main bot
    
    def set_virtual_users(self, virtual_users: Dict[str, Any]):
        """Set the virtual users dictionary"""
        self.virtual_users = virtual_users
    
    @discord.app_commands.command(name="status", description="Display detailed state of each agent (omitted from their context)")
    async def status_command(self, interaction: discord.Interaction, agent_id: str = ""):
        """Display agent status - hidden from LLM context"""
        
        await interaction.response.defer(ephemeral=True)
        
        try:
            if not self.virtual_users:
                await interaction.followup.send("🤖 No agents currently loaded", ephemeral=True)
                return
            
            if agent_id:
                # Show specific agent
                if agent_id in self.virtual_users:
                    embed = self._create_agent_status(agent_id)
                    await interaction.followup.send(embed=embed, ephemeral=True)
                else:
                    available = ', '.join(self.virtual_users.keys())
                    await interaction.followup.send(
                        f"❌ Agent `{agent_id}` not found.\n**Available:** {available}",
                        ephemeral=True
                    )
            else:
                # Show all agents summary
                embed = self._create_system_overview()
                await interaction.followup.send(embed=embed, ephemeral=True)
                
                # Show individual agent summaries
                count = 0
                for agent_name in self.virtual_users.keys():
                    if count >= 5:  # Limit to prevent spam
                        break
                    agent_embed = self._create_agent_status(agent_name, compact=True)
                    await interaction.followup.send(embed=agent_embed, ephemeral=True)
                    count += 1
                
                if len(self.virtual_users) > 5:
                    await interaction.followup.send(
                        f"Showing first 5 agents. Use `/status agent_id:<name>` for specific agents.\n"
                        f"**Total agents:** {len(self.virtual_users)}", 
                        ephemeral=True
                    )
        
        except Exception as e:
            self.logger.error(f"Error in status command: {e}")
            await interaction.followup.send(f"❌ Error: {str(e)}", ephemeral=True)
    
    def _create_system_overview(self) -> discord.Embed:
        """Create system overview"""
        
        embed = discord.Embed(
            title="🤖 HoroRobo Agent Status Overview",
            description="System status summary",
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )
        
        embed.add_field(
            name="📊 System Info",
            value=f"**Total Agents:** {len(self.virtual_users)}\n"
                  f"**Loaded:** {len(self.virtual_users)}\n"
                  f"**Status:** 🟢 Online",
            inline=True
        )
        
        # List agents
        agent_list = '\n'.join([f"• `{name}`" for name in list(self.virtual_users.keys())[:10]])
        if len(self.virtual_users) > 10:
            agent_list += f"\n• ...and {len(self.virtual_users) - 10} more"
        
        embed.add_field(
            name="🤖 Active Agents",
            value=agent_list or "None",
            inline=True
        )
        
        embed.add_field(
            name="ℹ️ Information",
            value="This status is **hidden from agent context**.\n"
                  "Use `/status agent_id:<name>` for detailed info.",
            inline=False
        )
        
        embed.set_footer(text="📊 Status via /status command • Hidden from agents")
        
        return embed
    
    def _create_agent_status(self, agent_name: str, compact: bool = False) -> discord.Embed:
        """Create status for a specific agent"""
        
        agent_data = self.virtual_users[agent_name]
        
        embed = discord.Embed(
            title=f"🤖 Agent: {agent_name}",
            color=discord.Color.green(),
            timestamp=datetime.utcnow()
        )
        
        # Basic agent info
        embed.add_field(
            name="📋 Basic Info",
            value=f"**Name:** {agent_name}\n"
                  f"**Type:** Virtual User\n"
                  f"**Status:** 🟢 Active",
            inline=True
        )
        
        # Agent configuration (if available)
        config_text = "**Configuration:**\n"
        if hasattr(agent_data, 'model'):
            config_text += f"• Model: `{agent_data.model}`\n"
        if hasattr(agent_data, 'personality'):
            personality = agent_data.personality[:50] + "..." if len(agent_data.personality) > 50 else agent_data.personality
            config_text += f"• Personality: {personality}\n"
        else:
            config_text += "• Basic virtual user\n"
        
        embed.add_field(
            name="⚙️ Configuration",
            value=config_text,
            inline=True
        )
        
        # Activity info (placeholder for now)
        embed.add_field(
            name="📊 Activity",
            value="**Status:** Loaded and ready\n"
                  "**Last Activity:** System startup\n"
                  "**Type:** Background virtual user",
            inline=False if not compact else True
        )
        
        # Feature availability
        features_text = "**Available Features:**\n"
        features_text += "• ✅ Basic conversation\n"
        features_text += "• ✅ Virtual user simulation\n"
        features_text += "• ⚠️ Advanced emotion engine (pending integration)\n"
        features_text += "• ⚠️ Self-evolution (pending integration)\n"
        
        if not compact:
            embed.add_field(
                name="🎯 Features",
                value=features_text,
                inline=False
            )
        
        # Enhancement note
        embed.add_field(
            name="🚀 Enhancement Note",
            value="This agent can be enhanced with:\n"
                  "• Live emotional states\n"
                  "• Performance monitoring\n" 
                  "• Self-evolution capabilities\n"
                  "• Real-time parameter tracking",
            inline=False
        )
        
        embed.set_footer(text=f"📊 Agent status • Hidden from {agent_name}'s context")
        
        return embed


async def add_simple_status_to_bot(bot: commands.Bot, virtual_users: Dict[str, Any]):
    """Add simple status command to existing bot"""
    
    status_cog = SimpleAgentStatus(bot)
    status_cog.set_virtual_users(virtual_users)
    
    await bot.add_cog(status_cog)
    
    logging.info("✅ Added simple /status command to bot")
    
    return status_cog