#!/usr/bin/env python3
"""
📊 Agent Status Command
Detailed agent state display that's omitted from LLM context
"""

import discord
from discord.ext import commands
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from emotion_engine.monitoring import production_monitor
from evolution.mutation_engine import mutation_engine
from evolution.evolution_scheduler import evolution_scheduler
from config.optimal_production import OPTIMAL_CONFIG, EXPECTED_PERFORMANCE


class AgentStatusCommand(commands.Cog):
    """Discord command for detailed agent status"""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.logger = logging.getLogger("AgentStatus")
    
    @discord.app_commands.command(name="status", description="Display detailed state of each agent")
    async def status_command(self, interaction: discord.Interaction, 
                           agent_id: str = "", 
                           detailed: bool = True):
        """
        Display detailed agent status (omitted from LLM context)
        
        Args:
            agent_id: Specific agent to show (optional, shows all if empty)
            detailed: Show detailed breakdown vs summary
        """
        
        await interaction.response.defer(ephemeral=True)
        
        try:
            if agent_id:
                # Show specific agent
                if agent_id in production_monitor.agent_metrics:
                    embed = self._create_single_agent_status(agent_id, detailed)
                    await interaction.followup.send(embed=embed, ephemeral=True)
                else:
                    await interaction.followup.send(
                        f"❌ Agent `{agent_id}` not found.\n**Available agents:** {', '.join(production_monitor.agent_metrics.keys()) or 'None'}",
                        ephemeral=True
                    )
            else:
                # Show all agents
                embeds = self._create_all_agents_status(detailed)
                
                if not embeds:
                    await interaction.followup.send("🤖 No active agents found", ephemeral=True)
                    return
                
                # Send first embed
                await interaction.followup.send(embed=embeds[0], ephemeral=True)
                
                # Send additional embeds if any (Discord limit is 10 embeds per message)
                for embed in embeds[1:10]:  # Limit to prevent spam
                    await interaction.followup.send(embed=embed, ephemeral=True)
                
                if len(embeds) > 10:
                    await interaction.followup.send(
                        f"📊 Showing first 10 agents. Use `/status agent_id:<name>` for specific agents.\n"
                        f"**Total agents:** {len(production_monitor.agent_metrics)}",
                        ephemeral=True
                    )
        
        except Exception as e:
            self.logger.error(f"Error in status command: {e}")
            await interaction.followup.send(f"❌ Error generating status: {str(e)}", ephemeral=True)
    
    def _create_all_agents_status(self, detailed: bool) -> List[discord.Embed]:
        """Create status embeds for all agents"""
        
        embeds = []
        
        # System overview embed
        system_embed = self._create_system_overview()
        embeds.append(system_embed)
        
        # Individual agent embeds
        agent_count = 0
        for agent_id in production_monitor.agent_metrics.keys():
            if agent_count >= 9:  # Leave room for system overview
                break
            
            agent_embed = self._create_single_agent_status(agent_id, detailed, compact=not detailed)
            embeds.append(agent_embed)
            agent_count += 1
        
        return embeds
    
    def _create_system_overview(self) -> discord.Embed:
        """Create system overview embed"""
        
        embed = discord.Embed(
            title="🤖 HoroRobo Agent Status Dashboard",
            description="Complete system status overview",
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )
        
        # Agent statistics
        total_agents = len(production_monitor.agent_metrics)
        active_agents = 0
        awake_agents = 0
        mutating_agents = len(evolution_scheduler.agents_in_mutation)
        
        # Calculate aggregate metrics
        total_fitness = 0.0
        total_stability = 0.0
        total_response_rate = 0.0
        agents_with_metrics = 0
        
        for agent_id, metrics in production_monitor.agent_metrics.items():
            performance = production_monitor.calculate_agent_performance(agent_id)
            if performance:
                active_agents += 1
                total_fitness += performance.fitness_score
                total_stability += performance.stability_score
                total_response_rate += performance.response_rate
                agents_with_metrics += 1
            
            # Check if agent is awake (has recent activity)
            if metrics.state_history:
                latest_time, latest_state = metrics.state_history[-1]
                if (datetime.utcnow().timestamp() - latest_time) < 300:  # Active in last 5 minutes
                    awake_agents += 1
        
        # Agent counts
        embed.add_field(
            name="📊 Agent Overview",
            value=f"**Total Agents:** {total_agents}\n"
                  f"**Active:** {active_agents}\n"
                  f"**Awake:** {awake_agents}\n"
                  f"**Mutating:** {mutating_agents}",
            inline=True
        )
        
        # Performance averages
        if agents_with_metrics > 0:
            avg_fitness = total_fitness / agents_with_metrics
            avg_stability = total_stability / agents_with_metrics
            avg_response_rate = total_response_rate / agents_with_metrics
            
            fitness_status = "✅" if avg_fitness >= EXPECTED_PERFORMANCE['fitness_target'] * 0.9 else "⚠️" if avg_fitness >= 0.7 else "❌"
            stability_status = "✅" if avg_stability >= EXPECTED_PERFORMANCE['stability_target'] * 0.9 else "⚠️" if avg_stability >= 0.8 else "❌"
            
            embed.add_field(
                name="📈 Performance",
                value=f"{fitness_status} **Fitness:** {avg_fitness:.3f}\n"
                      f"{stability_status} **Stability:** {avg_stability:.3f}\n"
                      f"📊 **Response Rate:** {avg_response_rate:.3f}",
                inline=True
            )
        
        # Evolution statistics
        evolution_stats = evolution_scheduler.evolution_stats
        embed.add_field(
            name="🧬 Evolution Stats",
            value=f"**Mutations Applied:** {evolution_stats['mutations_applied']}\n"
                  f"**Successful:** {evolution_stats['successful_mutations']}\n"
                  f"**Failed:** {evolution_stats['failed_mutations']}\n"
                  f"**Agents Evolved:** {len(evolution_stats['agents_evolved'])}",
            inline=True
        )
        
        # System health indicators
        system_health = "🟢 Healthy"
        health_issues = []
        
        if agents_with_metrics > 0:
            avg_fitness = total_fitness / agents_with_metrics
            if avg_fitness < 0.6:
                system_health = "🔴 Critical"
                health_issues.append("Low average fitness")
            elif avg_fitness < 0.75:
                system_health = "🟡 Warning"
                health_issues.append("Below target fitness")
        
        if mutating_agents > total_agents * 0.5:
            health_issues.append("High mutation activity")
        
        if awake_agents < total_agents * 0.3:
            health_issues.append("Many agents sleeping")
        
        embed.add_field(
            name="🏥 System Health",
            value=f"**Status:** {system_health}\n" + 
                  ("\n".join([f"• {issue}" for issue in health_issues[:3]]) if health_issues else "All systems normal"),
            inline=False
        )
        
        # Optimal configuration reference
        embed.add_field(
            name="🎯 Target Configuration",
            value=f"**Fitness Target:** {EXPECTED_PERFORMANCE['fitness_target']:.3f}\n"
                  f"**Stability Target:** {EXPECTED_PERFORMANCE['stability_target']:.3f}\n"
                  f"**Most Sensitive Param:** `social_decay_rate` ({OPTIMAL_CONFIG.social_decay_rate:.4f})",
            inline=True
        )
        
        embed.set_footer(text="📊 Status requested via /status command • Hidden from agent context")
        
        return embed
    
    def _create_single_agent_status(self, agent_id: str, detailed: bool, compact: bool = False) -> discord.Embed:
        """Create detailed status for a single agent"""
        
        metrics = production_monitor.agent_metrics[agent_id]
        performance = production_monitor.calculate_agent_performance(agent_id)
        
        # Color based on performance
        if performance:
            if performance.fitness_score >= 0.8:
                color = discord.Color.green()
            elif performance.fitness_score >= 0.6:
                color = discord.Color.orange()
            else:
                color = discord.Color.red()
        else:
            color = discord.Color.grey()
        
        embed = discord.Embed(
            title=f"🤖 Agent: {agent_id}",
            color=color,
            timestamp=datetime.utcnow()
        )
        
        # Current emotional state
        if metrics.state_history:
            timestamp, current_state = metrics.state_history[-1]
            age_seconds = int(datetime.utcnow().timestamp() - timestamp)
            age_text = f"{age_seconds//60}m {age_seconds%60}s ago" if age_seconds > 60 else f"{age_seconds}s ago"
            
            if compact:
                # Compact emotional state
                state_text = f"🔍 **Curiosity:** {current_state.curiosity:.3f}\n"
                state_text += f"💪 **Confidence:** {current_state.confidence:.3f}\n"
                state_text += f"⚡ **Social Energy:** {current_state.social_energy:.3f}\n"
                state_text += f"🧠 **Consciousness:** {current_state.consciousness.name}"
            else:
                # Detailed emotional state
                state_text = f"🔍 **Curiosity:** {current_state.curiosity:.3f} "
                state_text += f"({'↗️' if current_state.curiosity > OPTIMAL_CONFIG.curiosity_base else '↘️' if current_state.curiosity < OPTIMAL_CONFIG.curiosity_base * 0.8 else '➡️'})\n"
                
                state_text += f"💪 **Confidence:** {current_state.confidence:.3f} "
                state_text += f"({'↗️' if current_state.confidence > OPTIMAL_CONFIG.confidence_base * 1.2 else '↘️' if current_state.confidence < OPTIMAL_CONFIG.confidence_base else '➡️'})\n"
                
                state_text += f"⚡ **Social Energy:** {current_state.social_energy:.3f} "
                state_text += f"({'🔋' if current_state.social_energy > 0.7 else '🪫' if current_state.social_energy < 0.3 else '🔋'})\n"
                
                state_text += f"😤 **Restlessness:** {current_state.restlessness:.3f}\n"
                state_text += f"☯️ **Harmony:** {current_state.harmony:.3f}\n"
                state_text += f"🧠 **Consciousness:** {current_state.consciousness.name}\n"
                state_text += f"🎓 **Expertise:** {current_state.expertise:.3f}\n"
                state_text += f"✨ **Novelty:** {current_state.novelty:.3f}"
            
            embed.add_field(
                name=f"😊 Emotional State ({age_text})",
                value=state_text,
                inline=not detailed
            )
        else:
            embed.add_field(
                name="😊 Emotional State",
                value="`No state data available`",
                inline=True
            )
        
        # Performance metrics
        if performance:
            perf_text = f"🎯 **Fitness:** {performance.fitness_score:.3f}"
            perf_text += f" ({'✅' if performance.fitness_score >= 0.8 else '⚠️' if performance.fitness_score >= 0.6 else '❌'})\n"
            
            perf_text += f"🏗️ **Stability:** {performance.stability_score:.3f}"
            perf_text += f" ({'✅' if performance.stability_score >= 0.9 else '⚠️' if performance.stability_score >= 0.7 else '❌'})\n"
            
            perf_text += f"💬 **Response Rate:** {performance.response_rate:.3f}"
            perf_text += f" ({'📢' if performance.response_rate > 0.8 else '🤐' if performance.response_rate < 0.4 else '💬'})\n"
            
            perf_text += f"🧠 **Consciousness:** {performance.consciousness_score:.3f}"
            perf_text += f" ({'✅' if performance.consciousness_score >= 0.8 else '⚠️' if performance.consciousness_score >= 0.5 else '❌'})"
            
            embed.add_field(
                name="📈 Performance Metrics",
                value=perf_text,
                inline=not detailed
            )
        
        # Activity statistics
        activity_text = f"📊 **State Updates:** {len(metrics.state_history)}\n"
        activity_text += f"💭 **Decisions:** {len(metrics.response_history)}\n"
        activity_text += f"🔄 **Consciousness Changes:** {len(metrics.consciousness_transitions)}"
        
        if metrics.response_history:
            recent_responses = len([r for r in metrics.response_history if r['responded']])
            response_rate = recent_responses / len(metrics.response_history)
            activity_text += f"\n🗣️ **Recent Response Rate:** {response_rate:.2%}"
        
        embed.add_field(
            name="📊 Activity Stats",
            value=activity_text,
            inline=True
        )
        
        # Evolution/mutation history
        if agent_id in mutation_engine.mutation_history:
            evolution_history = mutation_engine.mutation_history[agent_id]
            
            evolution_text = f"🧬 **Total Mutations:** {len(evolution_history.mutations)}\n"
            evolution_text += f"✅ **Successful:** {evolution_history.successful_mutations}\n"
            evolution_text += f"❌ **Failed:** {evolution_history.failed_mutations}"
            
            if evolution_history.mutations:
                latest_mutation = evolution_history.mutations[-1]
                time_since = datetime.utcnow().timestamp() - latest_mutation['timestamp']
                age_text = f"{int(time_since//60)}m ago" if time_since > 60 else f"{int(time_since)}s ago"
                
                evolution_text += f"\n🕐 **Last Mutation:** {age_text}"
                evolution_text += f"\n🎛️ **Parameter:** `{latest_mutation['parameter']}`"
                evolution_text += f"\n📈 **Change:** {latest_mutation['old_value']:.3f} → {latest_mutation['new_value']:.3f}"
            
            embed.add_field(
                name="🧬 Evolution History",
                value=evolution_text,
                inline=True
            )
        
        # Deviation from optimal configuration
        if metrics.state_history and detailed:
            _, current_state = metrics.state_history[-1]
            
            deviations = []
            
            curiosity_dev = abs(current_state.curiosity - OPTIMAL_CONFIG.curiosity_base)
            if curiosity_dev > 0.2:
                deviations.append(f"🔍 Curiosity: {curiosity_dev:+.3f}")
            
            confidence_dev = abs(current_state.confidence - OPTIMAL_CONFIG.confidence_base)
            if confidence_dev > 0.2:
                deviations.append(f"💪 Confidence: {confidence_dev:+.3f}")
            
            social_dev = abs(current_state.social_energy - OPTIMAL_CONFIG.social_energy_base)
            if social_dev > 0.2:
                deviations.append(f"⚡ Social: {social_dev:+.3f}")
            
            if deviations:
                embed.add_field(
                    name="⚠️ Deviations from Optimal",
                    value="\n".join(deviations[:3]) + ("..." if len(deviations) > 3 else ""),
                    inline=False
                )
            else:
                embed.add_field(
                    name="✅ Configuration Status",
                    value="Within optimal parameters",
                    inline=False
                )
        
        # Footer
        is_mutating = agent_id in evolution_scheduler.agents_in_mutation
        mutation_status = "🧬 Currently mutating" if is_mutating else "🔒 Stable"
        
        embed.set_footer(text=f"{mutation_status} • Hidden from agent context")
        
        return embed


async def setup_status_command(bot: commands.Bot) -> AgentStatusCommand:
    """Setup the agent status command"""
    
    status_cog = AgentStatusCommand(bot)
    await bot.add_cog(status_cog)
    
    return status_cog