#!/usr/bin/env python3
"""
👁️ Hidden Reflection Display System
Shows LLM reflection passes to humans without including them in LLM context
"""

import discord
from discord.ext import commands
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime
import json

class HiddenReflectionDisplay:
    """Displays hidden LLM reflection passes for human monitoring"""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.logger = logging.getLogger("HiddenReflection")
        
        # Track reflection data
        self.reflection_log: List[Dict[str, Any]] = []
        self.cost_tracking = {
            'total_requests': 0,
            'hidden_requests': 0,
            'visible_requests': 0,
            'estimated_tokens': 0
        }
        
        # Channel for hidden displays (admin-only)
        self.monitoring_channel_id: Optional[int] = None
        
    def set_monitoring_channel(self, channel_id: int):
        """Set the channel where hidden reflections are displayed"""
        self.monitoring_channel_id = channel_id
        self.logger.info(f"Set monitoring channel to {channel_id}")
    
    async def log_hidden_reflection(self, 
                                   agent_id: str,
                                   reflection_type: str,
                                   reasoning: str,
                                   decision: bool,
                                   confidence: float,
                                   emotional_state: Dict[str, Any],
                                   trigger_context: str,
                                   estimated_tokens: int = 0):
        """Log a hidden reflection pass"""
        
        reflection_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'agent_id': agent_id,
            'type': reflection_type,
            'reasoning': reasoning,
            'decision': 'RESPOND' if decision else 'PASS',
            'confidence': confidence,
            'emotional_state': emotional_state,
            'trigger_context': trigger_context[:200] + "..." if len(trigger_context) > 200 else trigger_context,
            'estimated_tokens': estimated_tokens
        }
        
        self.reflection_log.append(reflection_data)
        
        # Update cost tracking
        self.cost_tracking['total_requests'] += 1
        self.cost_tracking['hidden_requests'] += 1
        self.cost_tracking['estimated_tokens'] += estimated_tokens
        
        # Keep only last 100 reflections
        if len(self.reflection_log) > 100:
            self.reflection_log = self.reflection_log[-100:]
        
        # Display to monitoring channel if set
        if self.monitoring_channel_id:
            await self._display_reflection(reflection_data)
    
    async def log_visible_response(self, agent_id: str, response_length: int, estimated_tokens: int = 0):
        """Log a visible response for cost tracking"""
        
        self.cost_tracking['visible_requests'] += 1
        self.cost_tracking['estimated_tokens'] += estimated_tokens
        
        # Optional: Log visible responses too
        visible_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'agent_id': agent_id,
            'type': 'visible_response',
            'response_length': response_length,
            'estimated_tokens': estimated_tokens
        }
        
        if self.monitoring_channel_id:
            await self._display_visible_response(visible_data)
    
    async def _display_reflection(self, reflection_data: Dict[str, Any]):
        """Display hidden reflection in monitoring channel"""
        
        try:
            channel = self.bot.get_channel(self.monitoring_channel_id)
            if not channel:
                return
            
            # Create embed for reflection
            embed = discord.Embed(
                title="🧠 Hidden Reflection Pass",
                color=discord.Color.blue() if reflection_data['decision'] == 'RESPOND' else discord.Color.grey(),
                timestamp=datetime.fromisoformat(reflection_data['timestamp'])
            )
            
            # Agent info
            embed.add_field(
                name="🤖 Agent",
                value=f"`{reflection_data['agent_id']}`",
                inline=True
            )
            
            # Decision
            decision_emoji = "✅" if reflection_data['decision'] == 'RESPOND' else "⏸️"
            embed.add_field(
                name="🎯 Decision",
                value=f"{decision_emoji} {reflection_data['decision']}",
                inline=True
            )
            
            # Confidence
            confidence_bar = self._create_progress_bar(reflection_data['confidence'])
            embed.add_field(
                name="🎲 Confidence",
                value=f"{confidence_bar} {reflection_data['confidence']:.2f}",
                inline=True
            )
            
            # Emotional state (compact)
            emotional_summary = self._format_emotional_state(reflection_data['emotional_state'])
            embed.add_field(
                name="😊 Emotional State",
                value=emotional_summary,
                inline=False
            )
            
            # Reasoning (truncated)
            reasoning = reflection_data['reasoning'][:300] + "..." if len(reflection_data['reasoning']) > 300 else reflection_data['reasoning']
            embed.add_field(
                name="🧠 Reasoning",
                value=f"```{reasoning}```",
                inline=False
            )
            
            # Context trigger
            embed.add_field(
                name="📝 Context",
                value=f"```{reflection_data['trigger_context']}```",
                inline=False
            )
            
            # Cost info
            if reflection_data['estimated_tokens'] > 0:
                embed.set_footer(text=f"Est. tokens: {reflection_data['estimated_tokens']} | Type: {reflection_data['type']}")
            
            await channel.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Failed to display hidden reflection: {e}")
    
    async def _display_visible_response(self, visible_data: Dict[str, Any]):
        """Display visible response log in monitoring channel"""
        
        try:
            channel = self.bot.get_channel(self.monitoring_channel_id)
            if not channel:
                return
            
            embed = discord.Embed(
                title="💬 Visible Response",
                color=discord.Color.green(),
                timestamp=datetime.fromisoformat(visible_data['timestamp'])
            )
            
            embed.add_field(
                name="🤖 Agent",
                value=f"`{visible_data['agent_id']}`",
                inline=True
            )
            
            embed.add_field(
                name="📏 Length",
                value=f"{visible_data['response_length']} chars",
                inline=True
            )
            
            if visible_data['estimated_tokens'] > 0:
                embed.set_footer(text=f"Est. tokens: {visible_data['estimated_tokens']}")
            
            await channel.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Failed to display visible response: {e}")
    
    def _create_progress_bar(self, value: float, length: int = 10) -> str:
        """Create a text progress bar"""
        filled = int(value * length)
        bar = "█" * filled + "░" * (length - filled)
        return f"`{bar}`"
    
    def _format_emotional_state(self, emotional_state: Dict[str, Any]) -> str:
        """Format emotional state compactly"""
        
        if not emotional_state:
            return "`No emotional data`"
        
        # Key emotions with icons
        emotions = {
            'curiosity': '🔍',
            'confidence': '💪', 
            'social_energy': '⚡',
            'restlessness': '😤',
            'harmony': '☯️',
            'consciousness': '🧠'
        }
        
        formatted = []
        for key, icon in emotions.items():
            if key in emotional_state:
                value = emotional_state[key]
                if isinstance(value, (int, float)):
                    formatted.append(f"{icon}{value:.2f}")
                else:
                    formatted.append(f"{icon}{value}")
        
        return " ".join(formatted) if formatted else "`No data`"
    
    async def display_cost_summary(self, channel: discord.TextChannel):
        """Display cost and usage summary"""
        
        embed = discord.Embed(
            title="📊 LLM Usage & Cost Summary",
            color=discord.Color.gold(),
            timestamp=datetime.utcnow()
        )
        
        # Request breakdown
        total_requests = self.cost_tracking['total_requests']
        hidden_requests = self.cost_tracking['hidden_requests'] 
        visible_requests = self.cost_tracking['visible_requests']
        
        embed.add_field(
            name="🔢 Total Requests",
            value=f"`{total_requests:,}`",
            inline=True
        )
        
        embed.add_field(
            name="👁️ Hidden Passes",
            value=f"`{hidden_requests:,}` ({hidden_requests/max(total_requests,1)*100:.1f}%)",
            inline=True
        )
        
        embed.add_field(
            name="💬 Visible Responses", 
            value=f"`{visible_requests:,}` ({visible_requests/max(total_requests,1)*100:.1f}%)",
            inline=True
        )
        
        # Token estimation
        estimated_tokens = self.cost_tracking['estimated_tokens']
        embed.add_field(
            name="🧮 Est. Total Tokens",
            value=f"`{estimated_tokens:,}`",
            inline=True
        )
        
        # Rough cost estimate (varies by model)
        estimated_cost = estimated_tokens * 0.000002  # Rough estimate for mid-tier models
        embed.add_field(
            name="💰 Est. Cost",
            value=f"`${estimated_cost:.4f}`",
            inline=True
        )
        
        # Efficiency metrics
        if total_requests > 0:
            tokens_per_request = estimated_tokens / total_requests
            embed.add_field(
                name="⚡ Avg Tokens/Request",
                value=f"`{tokens_per_request:.0f}`",
                inline=True
            )
        
        # Recent activity
        recent_reflections = [r for r in self.reflection_log if 
                            (datetime.utcnow() - datetime.fromisoformat(r['timestamp'])).total_seconds() < 3600]
        
        embed.add_field(
            name="🕐 Last Hour Activity",
            value=f"`{len(recent_reflections)}` hidden passes",
            inline=False
        )
        
        await channel.send(embed=embed)
    
    async def get_reflection_history(self, agent_id: Optional[str] = None, 
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent reflection history"""
        
        reflections = self.reflection_log
        
        if agent_id:
            reflections = [r for r in reflections if r['agent_id'] == agent_id]
        
        return reflections[-limit:] if reflections else []
    
    def export_reflection_data(self) -> Dict[str, Any]:
        """Export all reflection data for analysis"""
        
        return {
            'reflection_log': self.reflection_log,
            'cost_tracking': self.cost_tracking,
            'export_timestamp': datetime.utcnow().isoformat(),
            'total_reflections': len(self.reflection_log)
        }


class ReflectionMonitoringCommands(commands.Cog):
    """Discord commands for reflection monitoring"""
    
    def __init__(self, bot: commands.Bot, reflection_display: HiddenReflectionDisplay):
        self.bot = bot
        self.reflection_display = reflection_display
    
    @discord.app_commands.command(name="set-monitoring-channel", description="Set channel for hidden reflection display")
    async def set_monitoring_channel(self, interaction: discord.Interaction):
        """Set current channel as monitoring channel"""
        
        # Check if user is admin
        if interaction.user.id not in [123456789]:  # Replace with actual admin IDs
            await interaction.response.send_message("❌ Admin only command", ephemeral=True)
            return
        
        self.reflection_display.set_monitoring_channel(interaction.channel.id)
        await interaction.response.send_message(
            f"✅ Set {interaction.channel.mention} as hidden reflection monitoring channel",
            ephemeral=True
        )
    
    @discord.app_commands.command(name="reflection-summary", description="Show LLM usage and cost summary")
    async def reflection_summary(self, interaction: discord.Interaction):
        """Show reflection and cost summary"""
        
        await interaction.response.defer(ephemeral=True)
        await self.reflection_display.display_cost_summary(interaction.channel)
        await interaction.followup.send("📊 Summary displayed above", ephemeral=True)
    
    @discord.app_commands.command(name="recent-reflections", description="Show recent hidden reflections")
    async def recent_reflections(self, interaction: discord.Interaction, 
                                agent: str = "", limit: int = 5):
        """Show recent reflections"""
        
        await interaction.response.defer(ephemeral=True)
        
        agent_id = agent if agent else None
        reflections = await self.reflection_display.get_reflection_history(agent_id, limit)
        
        if not reflections:
            await interaction.followup.send("No recent reflections found", ephemeral=True)
            return
        
        embed = discord.Embed(
            title=f"🧠 Recent Hidden Reflections{f' - {agent}' if agent else ''}",
            color=discord.Color.purple()
        )
        
        for i, reflection in enumerate(reflections[-5:], 1):
            decision_emoji = "✅" if reflection['decision'] == 'RESPOND' else "⏸️"
            
            embed.add_field(
                name=f"{i}. {decision_emoji} {reflection['agent_id']}",
                value=f"**Decision:** {reflection['decision']}\n"
                      f"**Confidence:** {reflection['confidence']:.2f}\n"
                      f"**Reasoning:** {reflection['reasoning'][:100]}...\n"
                      f"**Time:** <t:{int(datetime.fromisoformat(reflection['timestamp']).timestamp())}:R>",
                inline=False
            )
        
        await interaction.followup.send(embed=embed, ephemeral=True)


# Integration example
async def setup_hidden_reflection_monitoring(bot: commands.Bot, config: Dict[str, Any]) -> HiddenReflectionDisplay:
    """Setup hidden reflection monitoring system"""
    
    reflection_display = HiddenReflectionDisplay(bot)
    
    # Set monitoring channel from config if available
    if monitoring_channel := config.get('monitoring_channel_id'):
        reflection_display.set_monitoring_channel(monitoring_channel)
    
    # Add commands
    await bot.add_cog(ReflectionMonitoringCommands(bot, reflection_display))
    
    return reflection_display


# Example usage in emotion engine integration
async def integrate_with_emotion_engine(reflection_display: HiddenReflectionDisplay):
    """Example of how to integrate with emotion engine"""
    
    from emotion_engine.reflection import MetaReflector
    
    # Patch the reflection system to log hidden passes
    original_perform_reflection = MetaReflector.perform_reflection
    
    async def patched_perform_reflection(self, runtime, context, reflection_type):
        """Patched reflection that logs hidden passes"""
        
        # Perform normal reflection
        result = await original_perform_reflection(self, runtime, context, reflection_type)
        
        # Log the hidden reflection pass
        await reflection_display.log_hidden_reflection(
            agent_id=runtime.agent_id,
            reflection_type=reflection_type.value,
            reasoning=result.reasoning,
            decision=result.should_respond,
            confidence=result.confidence,
            emotional_state={
                'curiosity': runtime.current_state.curiosity,
                'confidence': runtime.current_state.confidence,
                'social_energy': runtime.current_state.social_energy,
                'restlessness': runtime.current_state.restlessness,
                'harmony': runtime.current_state.harmony,
                'consciousness': runtime.current_state.consciousness.name
            },
            trigger_context=str(context.conversation_history[-1]) if context.conversation_history else "No context",
            estimated_tokens=150  # Rough estimate for reflection pass
        )
        
        return result
    
    # Apply the patch
    MetaReflector.perform_reflection = patched_perform_reflection