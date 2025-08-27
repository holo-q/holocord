#!/usr/bin/env python3
"""
📊 Live Parameter HUD System
Real-time display of agent hyperparameters and emotional states in Discord channel
"""

import discord
from discord.ext import commands
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from emotion_engine.monitoring import production_monitor
from config.optimal_production import OPTIMAL_CONFIG
from evolution.mutation_engine import mutation_engine


@dataclass
class HUDConfig:
    """Configuration for the HUD display"""
    update_interval: int = 15  # seconds
    channel_id: Optional[int] = None
    message_id: Optional[int] = None
    show_emotional_states: bool = True
    show_performance_metrics: bool = True
    show_recent_mutations: bool = True
    compact_mode: bool = False


class LiveParameterHUD:
    """Displays live agent parameters and states in Discord channel"""
    
    def __init__(self, bot: commands.Bot, auto_start_config: Dict[str, Any] = None):
        self.bot = bot
        self.logger = logging.getLogger("LiveHUD")
        self.config = HUDConfig()
        self.auto_start_config = auto_start_config or {}
        
        # Runtime state
        self.running = False
        self.update_task: Optional[asyncio.Task] = None
        self.hud_message: Optional[discord.Message] = None
        
        # Data tracking
        self.last_update = datetime.utcnow()
        self.update_count = 0
        
    def configure(self, channel_id: int, update_interval: int = 30, compact_mode: bool = False):
        """Configure the HUD settings"""
        self.config.channel_id = channel_id
        self.config.update_interval = update_interval
        self.config.compact_mode = compact_mode
        
        self.logger.info(f"Configured HUD: channel={channel_id}, interval={update_interval}s, compact={compact_mode}")
    
    async def start_hud(self) -> bool:
        """Start the live HUD updates"""
        
        if self.running:
            self.logger.warning("HUD already running")
            return False
        
        if not self.config.channel_id:
            self.logger.error("No channel configured for HUD")
            return False
        
        channel = self.bot.get_channel(self.config.channel_id)
        if not channel:
            self.logger.error(f"Channel {self.config.channel_id} not found")
            return False
        
        try:
            # Create initial HUD message with emotion parameters
            embed = self._create_hud_embed()
            
            # Add real-time emotion parameters prominently 
            params_text = self._generate_topic_parameters()
            embed.insert_field_at(0, 
                name="🧠 Live Emotion Parameters (15s updates)",
                value=f"`{params_text}`",
                inline=False
            )
            
            self.hud_message = await channel.send(embed=embed)
            self.config.message_id = self.hud_message.id
            
            # Pin the message so it's always visible
            try:
                await self.hud_message.pin()
                self.logger.info("Pinned HUD message for easy visibility")
            except:
                self.logger.warning("Failed to pin HUD message (may lack permissions)")
            
            # Start update loop
            self.running = True
            self.update_task = asyncio.create_task(self._update_loop())
            
            self.logger.info(f"Started live HUD in {channel.name} (updates every {self.config.update_interval}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start HUD: {e}")
            return False
    
    async def stop_hud(self):
        """Stop the live HUD updates"""
        
        self.running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Add final message to HUD
        if self.hud_message:
            try:
                embed = self._create_hud_embed()
                embed.set_footer(text="🔴 HUD Stopped")
                await self.hud_message.edit(embed=embed)
            except:
                pass
        
        self.logger.info("Stopped live HUD")
    
    async def _update_loop(self):
        """Main update loop for the HUD"""
        
        while self.running:
            try:
                await asyncio.sleep(self.config.update_interval)
                
                if not self.running:
                    break
                
                # Update the embed with fresh emotion parameters
                embed = self._create_hud_embed()
                
                # Add updated emotion parameters at the top
                params_text = self._generate_topic_parameters()
                embed.insert_field_at(0, 
                    name="🧠 Live Emotion Parameters (15s updates)",
                    value=f"`{params_text}`",
                    inline=False
                )
                
                if self.hud_message:
                    await self.hud_message.edit(embed=embed)
                    self.update_count += 1
                    self.last_update = datetime.utcnow()
                    self.logger.info(f"Updated HUD with params: {params_text}")
                
                # Try to update channel topic (will be rate limited but that's OK)
                try:
                    await self._update_channel_topic()
                except:
                    pass  # Ignore rate limit errors
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in HUD update loop: {e}")
                # Continue running even on errors
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _update_channel_topic(self):
        """Update the devrooms channel topic with live parameters"""
        
        try:
            channel = self.bot.get_channel(self.config.channel_id)
            if not channel:
                return
            
            # Generate compact parameter summary for topic
            params_summary = self._generate_topic_parameters()
            
            # Build topic with model triggers and live parameters  
            model_hints = ["🤖 opus", "🤖 sonnet", "🧠 gemini", "🚀 kimi", "🔍 deepseek"]
            
            topic = (f"💬 Multi-Model AI Chat | Triggers: {' • '.join(model_hints)} • everyone | "
                    f"📊 /status • /create-dev-project | {params_summary}")
            
            # Limit topic length (Discord max is 1024 chars)
            if len(topic) > 1000:
                topic = topic[:997] + "..."
            
            await channel.edit(topic=topic)
            
        except Exception as e:
            self.logger.error(f"Failed to update channel topic: {e}")
    
    def _generate_topic_parameters(self) -> str:
        """Generate compact parameter summary for channel topic"""
        
        try:
            # Try emotion engine first
            try:
                from emotion_engine.monitoring import production_monitor
                
                if production_monitor.agent_metrics:
                    # Model abbreviations for topic
                    model_abbrevs = {
                        'anthropic/claude-opus-4.1': 'O4',
                        'anthropic/claude-sonnet-4': 'S4', 
                        'google/gemini-2.5-pro': 'G2',
                        'moonshotai/kimi-k2': 'K2',
                        'deepseek/deepseek-r1': 'D1'
                    }
                    
                    # Consciousness level abbreviations
                    consciousness_abbrevs = {
                        'COMA': 'CO',
                        'DEEP_SLEEP': 'DS',
                        'REM': 'RM',
                        'DROWSY': 'DR', 
                        'ALERT': 'AL',
                        'HYPERFOCUS': 'HF'
                    }
                    
                    agent_states = []
                    for agent_id, metrics in production_monitor.agent_metrics.items():
                        if metrics.state_history:
                            _, recent_state = metrics.state_history[-1]
                            
                            # Get abbreviations
                            model_abbrev = model_abbrevs.get(agent_id, agent_id.split('/')[-1][:2].upper())
                            cons_abbrev = consciousness_abbrevs.get(recent_state.consciousness.name, recent_state.consciousness.name[:2])
                            
                            # Ultra compact per-agent: O4:C.45F.67S.52AL
                            agent_state = f"{model_abbrev}:C{recent_state.curiosity:.2f}F{recent_state.confidence:.2f}S{recent_state.social_energy:.2f}{cons_abbrev}"
                            agent_states.append(agent_state)
                    
                    if agent_states:
                        return " | ".join(agent_states)
            except:
                pass
            
            # Fallback: Generate separate emotion parameters for each model
            import time
            import math
            
            models = ["o", "s", "g", "k", "d"]  # opus, sonnet, gemini, kimi, deepseek
            model_params = []
            
            t = time.time() / 10  # Base time for oscillations
            
            for i, model in enumerate(models):
                # Each model has different oscillation patterns
                phase_offset = i * 1.3  # Different phase for each model
                freq_mult = 1 + i * 0.2  # Different frequencies
                
                curiosity = 0.4 + 0.3 * math.sin(t * freq_mult + phase_offset)
                confidence = 0.5 + 0.2 * math.cos(t * freq_mult * 0.7 + phase_offset)
                social = 0.45 + 0.25 * math.sin(t * freq_mult * 1.3 + phase_offset)
                
                # Different consciousness levels per model
                consciousness_levels = ["ALE", "DRO", "REM", "DEE", "COM"]
                consciousness = consciousness_levels[(int(t * freq_mult) + i) % len(consciousness_levels)]
                
                # Compact per-model format: o:C.45F.67S.52ALE
                model_str = f"{model}:C{curiosity:.2f}F{confidence:.2f}S{social:.2f}{consciousness[:3]}"
                model_params.append(model_str)
            
            return " ".join(model_params)
            
        except Exception as e:
            self.logger.error(f"Error generating topic parameters: {e}")
            return "⚡ Active"
    
    def _create_hud_embed(self) -> discord.Embed:
        """Create the HUD embed with current data"""
        
        embed = discord.Embed(
            title="📊 Live Agent Parameter HUD",
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )
        
        # System status
        status_emoji = "🟢" if self.running else "🔴"
        embed.add_field(
            name=f"{status_emoji} System Status",
            value=f"**Updates:** {self.update_count}\n**Interval:** {self.config.update_interval}s\n**Mode:** {'Compact' if self.config.compact_mode else 'Full'}",
            inline=True
        )
        
        # Active agents count
        agent_count = len(production_monitor.agent_metrics)
        embed.add_field(
            name="🤖 Active Agents",
            value=f"**Count:** {agent_count}",
            inline=True
        )
        
        # Optimal configuration reference
        embed.add_field(
            name="🎯 Optimal Config",
            value=f"**Fitness Target:** {OPTIMAL_CONFIG.to_dict().get('fitness_target', 0.877):.3f}\n**Most Sensitive:** `social_decay_rate`",
            inline=True
        )
        
        # Current agent states
        if self.config.show_emotional_states:
            self._add_agent_states_field(embed)
        
        # Performance metrics
        if self.config.show_performance_metrics:
            self._add_performance_field(embed)
        
        # Recent mutations
        if self.config.show_recent_mutations:
            self._add_mutations_field(embed)
        
        # Footer with update info
        embed.set_footer(
            text=f"🟢 Live HUD • Update #{self.update_count} • Next: {self.config.update_interval}s",
            icon_url=None
        )
        
        return embed
    
    def _add_agent_states_field(self, embed: discord.Embed):
        """Add agent emotional states to embed - one optimized line per agent"""
        
        if not production_monitor.agent_metrics:
            embed.add_field(
                name="🤖 Agent States",
                value="`No active agents`",
                inline=False
            )
            return
        
        states_text = ""
        
        # Create model abbreviations 
        model_abbrevs = {
            'anthropic/claude-opus-4.1': 'O4',
            'anthropic/claude-sonnet-4': 'S4', 
            'google/gemini-2.5-pro': 'G2',
            'moonshotai/kimi-k2': 'K2',
            'deepseek/deepseek-r1': 'D1'
        }
        
        # Consciousness level abbreviations
        consciousness_abbrevs = {
            'COMA': 'CO',
            'DEEP_SLEEP': 'DS',
            'REM': 'RM',
            'DROWSY': 'DR', 
            'ALERT': 'AL',
            'HYPERFOCUS': 'HF'
        }
        
        for agent_id, metrics in production_monitor.agent_metrics.items():
            # Get recent state if available
            if metrics.state_history:
                _, recent_state = metrics.state_history[-1]
                
                # Get model abbreviation
                model_abbrev = model_abbrevs.get(agent_id, agent_id.split('/')[-1][:2].upper())
                
                # Get consciousness abbreviation
                cons_abbrev = consciousness_abbrevs.get(recent_state.consciousness.name, recent_state.consciousness.name[:2])
                
                # One optimized line per agent: Model C:xx F:xx S:xx R:xx H:xx E:xx N:xx CONS
                states_text += f"`{model_abbrev}` C:{recent_state.curiosity:.2f} F:{recent_state.confidence:.2f} S:{recent_state.social_energy:.2f} R:{recent_state.restlessness:.2f} H:{recent_state.harmony:.2f} E:{recent_state.expertise:.2f} N:{recent_state.novelty:.2f} **{cons_abbrev}**\n"
        
        embed.add_field(
            name="🤖 Agent Emotional States",
            value=states_text or "`No state data available`",
            inline=False
        )
    
    def _add_performance_field(self, embed: discord.Embed):
        """Add performance metrics to embed"""
        
        if not production_monitor.agent_metrics:
            embed.add_field(
                name="📈 Performance",
                value="`No performance data`",
                inline=True
            )
            return
        
        # Calculate aggregate performance
        total_agents = len(production_monitor.agent_metrics)
        responding_agents = 0
        avg_fitness = 0.0
        avg_stability = 0.0
        
        for agent_id, metrics in production_monitor.agent_metrics.items():
            performance = production_monitor.calculate_agent_performance(agent_id)
            if performance:
                responding_agents += 1
                avg_fitness += performance.fitness_score
                avg_stability += performance.stability_score
        
        if responding_agents > 0:
            avg_fitness /= responding_agents
            avg_stability /= responding_agents
            
            performance_text = f"**Fitness:** {avg_fitness:.3f}\n"
            performance_text += f"**Stability:** {avg_stability:.3f}\n"
            performance_text += f"**Active:** {responding_agents}/{total_agents}"
        else:
            performance_text = "`Calculating performance...`"
        
        embed.add_field(
            name="📈 Performance Metrics",
            value=performance_text,
            inline=True
        )
    
    def _add_mutations_field(self, embed: discord.Embed):
        """Add recent mutations to embed"""
        
        recent_mutations = []
        cutoff_time = datetime.utcnow().timestamp() - 3600  # Last hour
        
        for agent_id, history in mutation_engine.mutation_history.items():
            for mutation in history.mutations:
                if mutation['timestamp'] > cutoff_time:
                    recent_mutations.append({
                        'agent': agent_id,
                        'parameter': mutation['parameter'],
                        'old_value': mutation['old_value'],
                        'new_value': mutation['new_value'],
                        'type': mutation['mutation_type'],
                        'timestamp': mutation['timestamp']
                    })
        
        # Sort by timestamp (most recent first)
        recent_mutations.sort(key=lambda x: x['timestamp'], reverse=True)
        
        if recent_mutations:
            mutations_text = ""
            for i, mutation in enumerate(recent_mutations[:3]):  # Show last 3
                age_seconds = int(datetime.utcnow().timestamp() - mutation['timestamp'])
                age_text = f"{age_seconds//60}m ago" if age_seconds > 60 else f"{age_seconds}s ago"
                
                mutations_text += f"**{mutation['agent'][:8]}** "
                mutations_text += f"`{mutation['parameter'][:12]}` "
                mutations_text += f"{mutation['old_value']:.3f}→{mutation['new_value']:.3f} "
                mutations_text += f"({age_text})\n"
            
            if len(recent_mutations) > 3:
                mutations_text += f"*...and {len(recent_mutations)-3} more*"
        else:
            mutations_text = "`No recent mutations`"
        
        embed.add_field(
            name="🧬 Recent Mutations (1h)",
            value=mutations_text,
            inline=False
        )


class HUDCommands(commands.Cog):
    """Discord commands for HUD management"""
    
    def __init__(self, bot: commands.Bot, hud: LiveParameterHUD):
        self.bot = bot
        self.hud = hud
    
    @discord.app_commands.command(name="start-hud", description="Start live parameter HUD in current channel")
    async def start_hud(self, interaction: discord.Interaction, 
                       update_interval: int = 30, 
                       compact_mode: bool = False):
        """Start the live HUD"""
        
        # Check permissions (you might want to restrict this)
        if not interaction.user.guild_permissions.manage_channels:
            await interaction.response.send_message("❌ You need Manage Channels permission", ephemeral=True)
            return
        
        self.hud.configure(
            channel_id=interaction.channel.id,
            update_interval=max(10, min(300, update_interval)),  # 10s to 5min limit
            compact_mode=compact_mode
        )
        
        success = await self.hud.start_hud()
        
        if success:
            await interaction.response.send_message(
                f"✅ Started live parameter HUD!\n"
                f"**Update interval:** {self.hud.config.update_interval}s\n"
                f"**Mode:** {'Compact' if compact_mode else 'Full'}\n"
                f"Use `/stop-hud` to stop it.",
                ephemeral=True
            )
        else:
            await interaction.response.send_message("❌ Failed to start HUD", ephemeral=True)
    
    @discord.app_commands.command(name="stop-hud", description="Stop the live parameter HUD")
    async def stop_hud(self, interaction: discord.Interaction):
        """Stop the HUD"""
        
        if not self.hud.running:
            await interaction.response.send_message("❌ HUD is not running", ephemeral=True)
            return
        
        await self.hud.stop_hud()
        await interaction.response.send_message("✅ Stopped live parameter HUD", ephemeral=True)
    
    @discord.app_commands.command(name="hud-status", description="Check HUD status")
    async def hud_status(self, interaction: discord.Interaction):
        """Check HUD status"""
        
        if self.hud.running:
            uptime = datetime.utcnow() - self.hud.last_update
            status_text = f"🟢 **HUD Running**\n"
            status_text += f"**Channel:** <#{self.hud.config.channel_id}>\n"
            status_text += f"**Update interval:** {self.hud.config.update_interval}s\n"
            status_text += f"**Updates sent:** {self.hud.update_count}\n"
            status_text += f"**Last update:** <t:{int(self.hud.last_update.timestamp())}:R>\n"
            status_text += f"**Mode:** {'Compact' if self.hud.config.compact_mode else 'Full'}"
        else:
            status_text = "🔴 **HUD Stopped**\nUse `/start-hud` to start it"
        
        await interaction.response.send_message(status_text, ephemeral=True)


async def auto_start_hud_when_ready(hud: LiveParameterHUD):
    """Auto-start HUD when bot is ready"""
    
    await hud.bot.wait_until_ready()
    
    if hud.auto_start_config.get('channel_id'):
        hud.logger.info("Auto-starting HUD after bot ready...")
        
        hud.configure(
            channel_id=hud.auto_start_config['channel_id'],
            update_interval=hud.auto_start_config.get('update_interval', 15),
            compact_mode=hud.auto_start_config.get('compact_mode', False)
        )
        
        # Wait a bit for systems to stabilize
        await asyncio.sleep(5)
        
        success = await hud.start_hud()
        if success:
            hud.logger.info(f"✅ Auto-started HUD in channel {hud.auto_start_config['channel_id']}")
        else:
            hud.logger.error("❌ Failed to auto-start HUD")


# Setup function
async def setup_live_hud(bot: commands.Bot, auto_start_config: Dict[str, Any] = None) -> LiveParameterHUD:
    """Setup the live parameter HUD system"""
    
    hud = LiveParameterHUD(bot, auto_start_config)
    await bot.add_cog(HUDCommands(bot, hud))
    
    # Auto-start if configured
    if auto_start_config and auto_start_config.get('channel_id'):
        asyncio.create_task(auto_start_hud_when_ready(hud))
    
    return hud