#!/usr/bin/env python3
"""
Emotional Virtual Users Manager for Discord Multi-Model Bot
Integrates emotion engine with virtual users for intelligent, emotion-driven AI agents
"""

import asyncio
import json
import re
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
import logging

import discord
from discord import app_commands
from discord.ext import commands, tasks
import httpx
import yaml

# Import emotion engine components
from genome import ExtendedGenome, AgentRuntime, EmotionState, ConsciousnessLevel, DNAParser
from emotion_engine import (
    consciousness_manager, dynamics_processor, meta_reflector, emotion_injector,
    cross_agent_dynamics, ReflectionContext, EnvironmentalInput, InjectionStyle
)
from modules.social import social_orchestrator, initialize_agent_social
from modules.cognitive import cognitive_processor, initialize_agent_cognition

@dataclass
class EmotionalVirtualUser:
    """Enhanced virtual user with emotional capabilities"""
    model_id: str
    discord_name: str
    webhook_url: str
    webhook_id: str
    avatar_url: Optional[str] = None
    description: str = ""
    pricing: Dict[str, float] = None
    context_length: int = 0
    active: bool = True
    
    # Emotion engine integration
    genome: Optional[ExtendedGenome] = None
    runtime: Optional[AgentRuntime] = None
    dna_string: Optional[str] = None
    personality_traits: Dict[str, float] = None
    last_reflection: float = 0.0
    reflection_interval: float = 45.0  # seconds between reflections
    
    def __post_init__(self):
        if self.pricing is None:
            self.pricing = {}
        if self.personality_traits is None:
            self.personality_traits = {}
    
    @property
    def is_awake(self) -> bool:
        """Check if agent is awake based on consciousness level"""
        if not self.runtime:
            return False
        return self.runtime.is_awake()
    
    @property
    def emotional_state(self) -> Optional[EmotionState]:
        """Get current emotional state"""
        return self.runtime.current_state if self.runtime else None
    
    @property
    def consciousness_level(self) -> Optional[ConsciousnessLevel]:
        """Get current consciousness level"""
        return self.runtime.current_state.consciousness if self.runtime else None
    
    def get_status_emoji(self) -> str:
        """Get emoji representing current state"""
        if not self.runtime:
            return "💤"
        
        state = self.runtime.current_state
        if state.consciousness <= ConsciousnessLevel.DEEP_SLEEP:
            return "💤"
        elif state.consciousness == ConsciousnessLevel.REM:
            return "😴"
        elif state.consciousness == ConsciousnessLevel.DROWSY:
            return "😪"
        elif state.consciousness == ConsciousnessLevel.ALERT:
            return "✨"
        else:  # HYPERFOCUS
            return "🔥"
    
    def get_emotional_summary(self) -> str:
        """Get brief emotional state summary"""
        if not self.runtime:
            return "offline"
        
        state = self.runtime.current_state
        
        # Find dominant emotion
        emotions = {
            'curious': state.curiosity,
            'confident': state.confidence,
            'restless': state.restlessness,
            'social': state.social_energy,
            'harmonious': state.harmony
        }
        
        dominant = max(emotions.items(), key=lambda x: x[1])
        if dominant[1] > 0.7:
            return f"{dominant[0]} ({dominant[1]:.1f})"
        elif dominant[1] > 0.5:
            return f"somewhat {dominant[0]}"
        else:
            return "neutral"

class EmotionalVirtualUserManager:
    """Enhanced virtual user manager with emotion engine"""
    
    def __init__(self, bot: commands.Bot, openrouter_api_key: str):
        self.bot = bot
        self.openrouter_api_key = openrouter_api_key
        self.virtual_users: Dict[str, EmotionalVirtualUser] = {}
        self.active_channels: Set[str] = set()
        self.dna_parser = DNAParser()
        
        # Reflection and dynamics tracking
        self.conversation_context: List[Dict[str, Any]] = []
        self.last_message_time = 0.0
        
        # Load existing users and start background tasks
        self.load_virtual_users()
        self.start_background_tasks()
    
    def load_virtual_users(self):
        """Load virtual users from JSON file and initialize emotion engines"""
        try:
            with open('emotional_virtual_users.json', 'r') as f:
                data = json.load(f)
                for user_data in data.get('users', []):
                    # Convert back to EmotionalVirtualUser
                    user = EmotionalVirtualUser(**user_data)
                    
                    # Initialize emotion engine if DNA string exists
                    if user.dna_string:
                        self._initialize_emotion_engine(user)
                    
                    self.virtual_users[user.model_id] = user
            
            logging.info(f"Loaded {len(self.virtual_users)} emotional virtual users")
        
        except FileNotFoundError:
            # Try to import from old format
            self._migrate_from_old_format()
        except Exception as e:
            logging.error(f"Error loading emotional virtual users: {e}")
    
    def _migrate_from_old_format(self):
        """Migrate from old virtual_users.json format"""
        try:
            with open('virtual_users.json', 'r') as f:
                data = json.load(f)
                for user_data in data.get('users', []):
                    # Create emotional user from old format
                    emotional_user = EmotionalVirtualUser(
                        model_id=user_data['model_id'],
                        discord_name=user_data['discord_name'],
                        webhook_url=user_data['webhook_url'],
                        webhook_id=user_data['webhook_id'],
                        avatar_url=user_data.get('avatar_url'),
                        description=user_data.get('description', ''),
                        pricing=user_data.get('pricing', {}),
                        context_length=user_data.get('context_length', 0),
                        active=user_data.get('active', True),
                    )
                    
                    # Generate default DNA for model type
                    emotional_user.dna_string = self._generate_default_dna(user_data['model_id'])
                    self._initialize_emotion_engine(emotional_user)
                    
                    self.virtual_users[emotional_user.model_id] = emotional_user
            
            logging.info(f"Migrated {len(self.virtual_users)} users from old format")
            self.save_virtual_users()
            
        except FileNotFoundError:
            logging.info("No existing virtual users found, starting fresh")
        except Exception as e:
            logging.error(f"Error migrating users: {e}")
    
    def _generate_default_dna(self, model_id: str) -> str:
        """Generate default DNA string based on model type"""
        model_dna = {
            'anthropic/claude-opus-4.1': 'cA9fB8sB7rB5hB6|C>R@70|A:30~90|H:45~75|D>F@55',
            'anthropic/claude-sonnet-4': 'cB8fB7sB8rA6hA7|C>H@65|S:40~80|F:50~85|R>S@40',
            'google/gemini-2.5-pro': 'cA7fB6sB6rB6hB5|R>C@60|F:45~75|S:35~65|C>F@70',
            'moonshotai/kimi-k2': 'cB6fB5sA8rA5hB7|H>S@75|C:40~70|F:50~80|S>H@65',
            'deepseek/deepseek-r1': 'cB9fA8sA6rA4hA5|C>F@80|R:25~55|H:40~60|F>C@85'
        }
        
        return model_dna.get(model_id, 'cB6fB6sB6rB5hB5|C>F@60|R:30~70|H:40~80|S:35~75')
    
    def _initialize_emotion_engine(self, user: EmotionalVirtualUser):
        """Initialize emotion engine components for a user"""
        if not user.dna_string:
            return
        
        try:
            # Parse DNA and create genome
            user.genome = self.dna_parser.parse(user.model_id, user.dna_string)
            
            # Create runtime
            user.runtime = AgentRuntime(
                agent_id=user.model_id,
                genome=user.genome,
                current_state=EmotionState()  # Default starting state
            )
            
            # Initialize social and cognitive profiles
            initialize_agent_social(user.model_id, user.discord_name, user.personality_traits)
            initialize_agent_cognition(user.model_id, user.model_id.split('/')[-1])
            
            logging.info(f"Initialized emotion engine for {user.discord_name}")
            
        except Exception as e:
            logging.error(f"Error initializing emotion engine for {user.model_id}: {e}")
    
    def save_virtual_users(self):
        """Save emotional virtual users to JSON file"""
        try:
            # Convert users to serializable format
            users_data = []
            for user in self.virtual_users.values():
                user_dict = asdict(user)
                # Remove non-serializable fields
                user_dict.pop('genome', None)
                user_dict.pop('runtime', None)
                users_data.append(user_dict)
            
            data = {
                'users': users_data,
                'active_channels': list(self.active_channels)
            }
            
            with open('emotional_virtual_users.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            logging.info(f"Saved {len(self.virtual_users)} emotional virtual users")
        
        except Exception as e:
            logging.error(f"Error saving emotional virtual users: {e}")
    
    def start_background_tasks(self):
        """Start background tasks for emotion processing"""
        self.reflection_loop.start()
        self.dynamics_update_loop.start()
        self.status_update_loop.start()
    
    @tasks.loop(seconds=15.0)  # Reflection every 15 seconds
    async def reflection_loop(self):
        """Background reflection loop for all agents"""
        try:
            current_time = time.time()
            
            for user in self.virtual_users.values():
                if not user.runtime or not user.active:
                    continue
                
                # Check if it's time for reflection
                if current_time - user.last_reflection >= user.reflection_interval:
                    await self._perform_agent_reflection(user)
                    user.last_reflection = current_time
            
        except Exception as e:
            logging.error(f"Error in reflection loop: {e}")
    
    @tasks.loop(seconds=30.0)  # Dynamics update every 30 seconds
    async def dynamics_update_loop(self):
        """Background dynamics update loop"""
        try:
            # Update all agent states
            for user in self.virtual_users.values():
                if user.runtime and user.active:
                    await self._update_agent_dynamics(user)
            
            # Apply cross-agent dynamics
            active_runtimes = {
                user.model_id: user.runtime 
                for user in self.virtual_users.values() 
                if user.runtime and user.active and user.runtime.is_awake()
            }
            
            if len(active_runtimes) > 1:
                updated_states = cross_agent_dynamics.apply_cross_agent_effects(active_runtimes)
                
                # Apply updated states
                for model_id, new_state in updated_states.items():
                    if model_id in self.virtual_users:
                        self.virtual_users[model_id].runtime.update_state(new_state)
        
        except Exception as e:
            logging.error(f"Error in dynamics update loop: {e}")
    
    @tasks.loop(minutes=2.0)  # Status update every 2 minutes
    async def status_update_loop(self):
        """Update channel status with agent information"""
        try:
            await self._update_channel_status()
        except Exception as e:
            logging.error(f"Error in status update loop: {e}")
    
    async def _perform_agent_reflection(self, user: EmotionalVirtualUser):
        """Perform reflection for a specific agent"""
        if not user.runtime:
            return
        
        try:
            # Create reflection context
            context = ReflectionContext.from_conversation(
                self.conversation_context,
                user.model_id,
                {u.model_id: u.emotional_state for u in self.virtual_users.values() 
                 if u.emotional_state and u.model_id != user.model_id}
            )
            
            # Perform reflection
            reflection_result = meta_reflector.perform_reflection(
                user.runtime, context
            )
            
            # If agent decides to respond, wake them up
            if reflection_result.should_respond and user.runtime.current_state.consciousness <= ConsciousnessLevel.REM:
                consciousness_manager.trigger_transition(user.runtime, 'conversation_activity')
                logging.info(f"🤔 {user.discord_name}: {reflection_result.reasoning}")
            
        except Exception as e:
            logging.error(f"Error in agent reflection for {user.model_id}: {e}")
    
    async def _update_agent_dynamics(self, user: EmotionalVirtualUser):
        """Update dynamics for a specific agent"""
        if not user.runtime:
            return
        
        try:
            # Create environmental input
            env_input = EnvironmentalInput.from_conversation_context(
                self.conversation_context,
                user.model_id,
                {u.model_id: u.emotional_state for u in self.virtual_users.values() 
                 if u.emotional_state and u.model_id != user.model_id}
            )
            
            # Update emotional state
            delta_time = time.time() - user.runtime.current_state.last_tick
            new_state = dynamics_processor.update_state(user.runtime, env_input, delta_time)
            user.runtime.update_state(new_state)
            
            # Update consciousness state
            consciousness_manager.update_consciousness(user.runtime, env_input)
            
            # Update cognitive state
            cognitive_processor.update_cognitive_state(
                user.model_id, user.runtime.current_state, 
                {'conversation': self.conversation_context}, delta_time
            )
            
        except Exception as e:
            logging.error(f"Error updating dynamics for {user.model_id}: {e}")
    
    async def _update_channel_status(self):
        """Update channel status message with agent states"""
        try:
            # Create status message
            awake_agents = []
            sleeping_agents = []
            
            for user in self.virtual_users.values():
                if not user.active:
                    continue
                
                status_info = f"{user.get_status_emoji()} **{user.discord_name}** - {user.get_emotional_summary()}"
                
                if user.is_awake:
                    awake_agents.append(status_info)
                else:
                    sleeping_agents.append(status_info)
            
            # Build status message
            status_parts = []
            if awake_agents:
                status_parts.append("**🟢 Awake Agents:**\n" + "\n".join(awake_agents))
            if sleeping_agents:
                status_parts.append("**🔴 Sleeping Agents:**\n" + "\n".join(sleeping_agents))
            
            if status_parts:
                status_message = "\n\n".join(status_parts)
                # TODO: Update actual channel status/pinned message
                logging.debug(f"Status update: {len(awake_agents)} awake, {len(sleeping_agents)} sleeping")
        
        except Exception as e:
            logging.error(f"Error updating channel status: {e}")
    
    async def create_virtual_user(self, channel: discord.TextChannel, model_id: str,
                                discord_name: Optional[str] = None, 
                                dna_string: Optional[str] = None) -> EmotionalVirtualUser:
        """Create a new emotional virtual user with webhook"""
        try:
            # Load model info from OpenRouter data
            with open('openrouter_models.json', 'r') as f:
                model_data = json.load(f)
            
            model_info = None
            for model in model_data['models']:
                if model['id'] == model_id:
                    model_info = model
                    break
            
            if not model_info:
                raise ValueError(f"Model {model_id} not found in OpenRouter data")
            
            # Use provided name or generate from model
            if not discord_name:
                discord_name = model_info['discord_name']
            
            # Create webhook
            webhook = await channel.create_webhook(
                name=discord_name,
                reason=f"Emotional virtual user for {model_id}"
            )
            
            # Get avatar URL based on model
            avatar_url = self.get_model_avatar_url(model_id)
            
            # Generate DNA string if not provided
            if not dna_string:
                dna_string = self._generate_default_dna(model_id)
            
            # Create emotional virtual user
            user = EmotionalVirtualUser(
                model_id=model_id,
                discord_name=discord_name,
                webhook_url=webhook.url,
                webhook_id=str(webhook.id),
                avatar_url=avatar_url,
                description=model_info.get('description', ''),
                pricing=model_info.get('pricing', {}),
                context_length=model_info.get('context_length', 0),
                dna_string=dna_string
            )
            
            # Initialize emotion engine
            self._initialize_emotion_engine(user)
            
            self.virtual_users[model_id] = user
            self.save_virtual_users()
            
            logging.info(f"Created emotional virtual user {discord_name} for {model_id}")
            return user
            
        except Exception as e:
            logging.error(f"Error creating emotional virtual user: {e}")
            raise
    
    async def send_as_virtual_user(self, model_id: str, content: str,
                                 embeds: Optional[List[discord.Embed]] = None,
                                 apply_emotions: bool = True) -> bool:
        """Send a message as a virtual user via webhook with optional emotion injection"""
        if model_id not in self.virtual_users:
            return False
        
        try:
            user = self.virtual_users[model_id]
            
            # Apply emotion injection if enabled and user has runtime
            final_content = content
            if apply_emotions and user.runtime:
                # Choose injection style based on consciousness level
                if user.runtime.current_state.consciousness >= ConsciousnessLevel.HYPERFOCUS:
                    style = InjectionStyle.ABSTRACT
                elif user.runtime.current_state.consciousness >= ConsciousnessLevel.ALERT:
                    style = InjectionStyle.NATURAL
                else:
                    style = InjectionStyle.SUBLIMINAL
                
                # Inject emotions
                injection = emotion_injector.create_contextual_injection(
                    user.runtime.current_state,
                    self.conversation_context,
                    user.personality_traits.get('style', 'balanced')
                )
                
                final_content = injection.apply_to_prompt(content)
            
            webhook_data = {
                'content': final_content,
                'username': user.discord_name,
            }
            
            if user.avatar_url:
                webhook_data['avatar_url'] = user.avatar_url
            
            if embeds:
                webhook_data['embeds'] = [embed.to_dict() for embed in embeds]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(user.webhook_url, json=webhook_data)
                response.raise_for_status()
            
            # Update user activity
            if user.runtime:
                user.runtime.last_spoke = time.time()
                # Refresh wake time if awake
                if user.runtime.is_awake():
                    consciousness_manager.trigger_transition(user.runtime, 'successful_interaction')
            
            return True
            
        except Exception as e:
            logging.error(f"Error sending message as {model_id}: {e}")
            return False
    
    async def query_openrouter_with_emotions(self, model_id: str, messages: List[Dict],
                                           system_prompt: Optional[str] = None,
                                           apply_emotions: bool = True) -> Optional[str]:
        """Query OpenRouter API with emotional context injection"""
        try:
            user = self.virtual_users.get(model_id)
            if not user:
                return await self.query_openrouter(model_id, messages, system_prompt)
            
            # Prepare messages with emotional context
            final_messages = messages.copy()
            final_system_prompt = system_prompt
            
            if apply_emotions and user.runtime:
                # Generate cognitive approach
                topic_context = [msg.get('content', '') for msg in messages[-3:]]
                thinking_approach = cognitive_processor.generate_reasoning_approach(
                    model_id, topic_context
                )
                
                # Add emotional system prompt
                emotion_context = f"""
{final_system_prompt or "You are a helpful AI assistant."}

{thinking_approach or ""}

Current emotional state: {user.get_emotional_summary()}
Consciousness level: {user.consciousness_level.name if user.consciousness_level else 'unknown'}
"""
                final_system_prompt = emotion_context.strip()
            
            return await self.query_openrouter(model_id, final_messages, final_system_prompt)
            
        except Exception as e:
            logging.error(f"Error querying with emotions {model_id}: {e}")
            return None
    
    async def query_openrouter(self, model_id: str, messages: List[Dict],
                             system_prompt: Optional[str] = None) -> Optional[str]:
        """Query OpenRouter API for model response"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/holo-q/llmcord",
                "X-Title": "HOLO-Q Discord Emotional Multi-Model Bot"
            }
            
            request_data = {
                "model": model_id,
                "messages": messages,
                "stream": False,
                "max_tokens": 2000,
            }
            
            if system_prompt:
                request_data["messages"] = [
                    {"role": "system", "content": system_prompt}
                ] + messages
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=request_data
                )
                response.raise_for_status()
                
                data = response.json()
                return data['choices'][0]['message']['content']
                
        except Exception as e:
            logging.error(f"Error querying {model_id}: {e}")
            return None
    
    def update_conversation_context(self, messages: List[Dict[str, Any]]):
        """Update conversation context for all agents"""
        self.conversation_context = messages[-10:]  # Keep last 10 messages
        self.last_message_time = time.time()
        
        # Wake up mentioned agents
        for message in messages[-1:]:  # Check last message
            content = message.get('content', '').lower()
            for user in self.virtual_users.values():
                if user.runtime and user.discord_name.lower() in content:
                    consciousness_manager.trigger_transition(user.runtime, 'direct_mention')
    
    def wake_user(self, model_id: str) -> bool:
        """Wake up a virtual user"""
        if model_id in self.virtual_users:
            user = self.virtual_users[model_id]
            if user.runtime:
                consciousness_manager.trigger_transition(user.runtime, 'external_wake')
                logging.info(f"🔥 {user.discord_name} is now AWAKE")
                return True
        return False
    
    def get_active_models(self) -> List[str]:
        """Get list of active model IDs"""
        return [model_id for model_id, user in self.virtual_users.items() if user.active]
    
    def get_awake_models(self) -> List[str]:
        """Get list of awake model IDs"""
        return [model_id for model_id, user in self.virtual_users.items() 
                if user.active and user.is_awake]
    
    def get_sleeping_models(self) -> List[str]:
        """Get list of sleeping model IDs"""
        return [model_id for model_id, user in self.virtual_users.items()
                if user.active and not user.is_awake]
    
    def get_virtual_user(self, model_id: str) -> Optional[EmotionalVirtualUser]:
        """Get virtual user by model ID"""
        return self.virtual_users.get(model_id)
    
    def get_model_avatar_url(self, model_id: str) -> Optional[str]:
        """Get appropriate avatar URL for a model"""
        avatar_map = {
            'anthropic/claude-opus-4.1': 'https://cdn.discordapp.com/attachments/1049724679377649697/1410025571803848847/claude-opus.png',
            'anthropic/claude-sonnet-4': 'https://cdn.discordapp.com/attachments/1049724679377649697/1410025571443125348/claude-sonnet.png',
            'google/gemini-2.5-pro': 'https://cdn.discordapp.com/attachments/1049724679377649697/1410025571099193364/gemini.png',
            'moonshotai/kimi-k2': 'https://cdn.discordapp.com/attachments/1049724679377649697/1410025570918449192/kimi.png',
            'deepseek/deepseek-r1': 'https://cdn.discordapp.com/attachments/1049724679377649697/1410025570897481838/deepseek.png'
        }
        return avatar_map.get(model_id)
    
    def get_emotional_summary(self) -> Dict[str, Any]:
        """Get emotional summary of all agents"""
        summary = {
            'total_agents': len(self.virtual_users),
            'active_agents': len(self.get_active_models()),
            'awake_agents': len(self.get_awake_models()),
            'agent_states': {}
        }
        
        for model_id, user in self.virtual_users.items():
            if user.runtime:
                state = user.runtime.current_state
                summary['agent_states'][model_id] = {
                    'name': user.discord_name,
                    'consciousness': state.consciousness.name,
                    'emotions': {
                        'curiosity': round(state.curiosity, 2),
                        'confidence': round(state.confidence, 2),
                        'social_energy': round(state.social_energy, 2),
                        'restlessness': round(state.restlessness, 2),
                        'harmony': round(state.harmony, 2)
                    },
                    'status': user.get_emotional_summary()
                }
        
        return summary

# Enhanced Discord slash commands with emotion features
class EmotionalVirtualUserCommands(commands.Cog):
    def __init__(self, bot: commands.Bot, manager: EmotionalVirtualUserManager):
        self.bot = bot
        self.manager = manager
    
    @app_commands.command(
        name="add_emotional_model",
        description="Add a new AI model with emotional capabilities (Admin only)"
    )
    @app_commands.describe(
        model_id="OpenRouter model ID",
        name="Custom Discord name (optional)",
        dna_string="Custom DNA string for emotions (optional)"
    )
    async def add_emotional_model(self, interaction: discord.Interaction, 
                                model_id: str, name: Optional[str] = None,
                                dna_string: Optional[str] = None):
        """Add a new emotional model as virtual user"""
        await interaction.response.defer()
        
        # Check admin permissions
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            admin_ids = config['permissions']['users']['admin_ids']
            if interaction.user.id not in admin_ids:
                await interaction.followup.send("❌ Admin only.", ephemeral=True)
                return
        except Exception as e:
            await interaction.followup.send("❌ Error checking permissions", ephemeral=True)
            return
        
        try:
            if model_id in self.manager.virtual_users:
                await interaction.followup.send(f"❌ Model `{model_id}` already exists!")
                return
            
            user = await self.manager.create_virtual_user(
                interaction.channel, model_id, name, dna_string
            )
            
            embed = discord.Embed(
                title="✨ Emotional Virtual User Added",
                description=f"**{user.discord_name}** (`{model_id}`)",
                color=discord.Color.green()
            )
            
            if user.runtime:
                state = user.runtime.current_state
                embed.add_field(
                    name="🧠 Initial State",
                    value=f"Consciousness: {state.consciousness.name}\nEmotions: {user.get_emotional_summary()}",
                    inline=True
                )
            
            embed.add_field(
                name="🧬 DNA",
                value=f"`{user.dna_string[:50]}{'...' if len(user.dna_string) > 50 else ''}`",
                inline=True
            )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            await interaction.followup.send(f"❌ Error adding emotional model: {str(e)}")
    
    @app_commands.command(
        name="emotional_status",
        description="Show detailed emotional status of all agents"
    )
    async def emotional_status(self, interaction: discord.Interaction):
        """Show emotional status of all agents"""
        await interaction.response.defer()
        
        summary = self.manager.get_emotional_summary()
        
        embed = discord.Embed(
            title="🧠 Emotional Agent Status",
            description=f"**{summary['awake_agents']}/{summary['active_agents']}** agents awake",
            color=discord.Color.blue()
        )
        
        for model_id, state_info in summary['agent_states'].items():
            consciousness_emoji = {
                'COMA': '💀', 'DEEP_SLEEP': '💤', 'REM': '😴',
                'DROWSY': '😪', 'ALERT': '✨', 'HYPERFOCUS': '🔥'
            }
            
            emoji = consciousness_emoji.get(state_info['consciousness'], '❓')
            
            emotion_text = ', '.join([
                f"{k}: {v}" for k, v in state_info['emotions'].items() 
                if v > 0.5
            ])
            
            embed.add_field(
                name=f"{emoji} {state_info['name']}",
                value=f"**{state_info['consciousness']}**\n{emotion_text or 'neutral'}\n*{state_info['status']}*",
                inline=True
            )
        
        await interaction.followup.send(embed=embed)
    
    @app_commands.command(
        name="wake_agent",
        description="Wake up a specific agent (Admin only)"
    )
    @app_commands.describe(model_id="Model ID to wake up")
    async def wake_agent(self, interaction: discord.Interaction, model_id: str):
        """Wake up a specific agent"""
        await interaction.response.defer()
        
        # Check admin permissions (simplified for brevity)
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            admin_ids = config['permissions']['users']['admin_ids']
            if interaction.user.id not in admin_ids:
                await interaction.followup.send("❌ Admin only.", ephemeral=True)
                return
        except:
            await interaction.followup.send("❌ Error checking permissions", ephemeral=True)
            return
        
        success = self.manager.wake_user(model_id)
        
        if success:
            user = self.manager.get_virtual_user(model_id)
            await interaction.followup.send(f"🔥 **{user.discord_name}** has been awakened!")
        else:
            await interaction.followup.send(f"❌ Could not wake `{model_id}` - not found or already awake")

if __name__ == "__main__":
    print("Emotional Virtual User Manager ready for integration with llmcord.py")