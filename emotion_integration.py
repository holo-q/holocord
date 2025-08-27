#!/usr/bin/env python3
"""
Emotion Engine Integration for LLMCord
Complete integration of emotional AI agents with Discord bot
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any

import discord
from discord.ext import commands

from emotional_virtual_users import EmotionalVirtualUserManager, EmotionalVirtualUserCommands
from emotion_scheduler import EmotionScheduler
from emotion_engine import ReflectionContext, meta_reflector, ReflectionType
from modules.social import social_orchestrator
from modules.cognitive import cognitive_processor

class EmotionIntegration:
    """Main integration class for emotion engine with Discord bot"""
    
    def __init__(self, bot: commands.Bot, openrouter_api_key: str):
        self.bot = bot
        self.manager = EmotionalVirtualUserManager(bot, openrouter_api_key)
        self.scheduler = EmotionScheduler(self.manager)
        self.is_initialized = False
        
        # Conversation tracking
        self.message_buffer: List[Dict[str, Any]] = []
        self.last_activity_check = 0.0
        
        # Integration state
        self.active_channels: set = set()
        self.trigger_keywords = self._load_trigger_keywords()
    
    def _load_trigger_keywords(self) -> Dict[str, List[str]]:
        """Load model trigger keywords"""
        return {
            'anthropic/claude-opus-4.1': ['claude', 'opus'],
            'anthropic/claude-sonnet-4': ['claude', 'sonnet'],
            'google/gemini-2.5-pro': ['gemini'],
            'moonshotai/kimi-k2': ['kimi'],
            'deepseek/deepseek-r1': ['deepseek', 'r1'],
            'global': ['everyone', 'all models', 'all agents', 'compare']
        }
    
    async def initialize(self):
        """Initialize the emotion integration"""
        if self.is_initialized:
            return
        
        try:
            # Add Discord commands
            await self.bot.add_cog(EmotionalVirtualUserCommands(self.bot, self.manager))
            
            # Start emotion scheduler
            asyncio.create_task(self.scheduler.start())
            
            # Set up event handlers
            self._setup_event_handlers()
            
            self.is_initialized = True
            logging.info("✨ Emotion integration initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing emotion integration: {e}")
            raise
    
    def _setup_event_handlers(self):
        """Set up Discord event handlers"""
        
        @self.bot.event
        async def on_message(message):
            """Handle incoming messages for emotional processing"""
            if message.author.bot or not self.is_initialized:
                return
            
            # Update conversation context
            await self._update_conversation_context(message)
            
            # Check for model triggers
            triggered_models = self._check_triggers(message.content)
            
            if triggered_models:
                # Wake triggered models
                for model_id in triggered_models:
                    self.manager.wake_user(model_id)
                
                # Process emotional responses
                await self._process_emotional_responses(message, triggered_models)
    
    async def _update_conversation_context(self, message: discord.Message):
        """Update conversation context for emotional processing"""
        try:
            # Convert message to context format
            message_data = {
                'content': message.content,
                'author': {
                    'id': str(message.author.id),
                    'name': message.author.display_name
                },
                'timestamp': message.created_at.timestamp(),
                'channel_id': str(message.channel.id),
                'guild_id': str(message.guild.id) if message.guild else None
            }
            
            # Add to buffer
            self.message_buffer.append(message_data)
            
            # Keep only recent messages
            if len(self.message_buffer) > 20:
                self.message_buffer = self.message_buffer[-20:]
            
            # Update manager's conversation context
            self.manager.update_conversation_context(self.message_buffer)
            
            # Track activity for scheduler adjustments
            activity_level = min(1.0, len(self.message_buffer) / 10.0)
            self.scheduler.adjust_reflection_frequency(activity_level)
            
        except Exception as e:
            logging.error(f"Error updating conversation context: {e}")
    
    def _check_triggers(self, content: str) -> List[str]:
        """Check message content for model triggers"""
        content_lower = content.lower()
        triggered_models = []
        
        # Check global triggers first
        for trigger in self.trigger_keywords['global']:
            if trigger in content_lower:
                return list(self.manager.get_active_models())
        
        # Check individual model triggers
        for model_id, triggers in self.trigger_keywords.items():
            if model_id == 'global':
                continue
                
            for trigger in triggers:
                if trigger in content_lower:
                    if model_id in self.manager.virtual_users:
                        triggered_models.append(model_id)
                    break
        
        return triggered_models
    
    async def _process_emotional_responses(self, message: discord.Message, 
                                         triggered_models: List[str]):
        """Process emotional responses for triggered models"""
        try:
            # Create message context for models
            messages = [{'role': 'user', 'content': message.content}]
            
            # Add recent context if available
            if len(self.message_buffer) > 1:
                context_messages = []
                for msg_data in self.message_buffer[-5:-1]:  # Last 4 messages before current
                    context_messages.append({
                        'role': 'user',
                        'content': f"[{msg_data['author']['name']}]: {msg_data['content']}"
                    })
                messages = context_messages + messages
            
            # Process each triggered model
            response_tasks = []
            for model_id in triggered_models:
                user = self.manager.get_virtual_user(model_id)
                if not user or not user.active:
                    continue
                
                # Check if agent wants to respond (reflection)
                if user.runtime:
                    context = ReflectionContext.from_conversation(
                        self.message_buffer,
                        user.model_id,
                        {u.model_id: u.emotional_state for u in self.manager.virtual_users.values()
                         if u.emotional_state and u.model_id != model_id}
                    )
                    
                    reflection_result = meta_reflector.perform_reflection(
                        user.runtime, context, ReflectionType.QUICK_CHECK
                    )
                    
                    if not reflection_result.should_respond:
                        logging.info(f"🤔 {user.discord_name} chose not to respond: {reflection_result.reasoning}")
                        continue
                
                # Create response task
                task = self._generate_emotional_response(user, messages, message.channel)
                response_tasks.append(task)
            
            # Execute responses concurrently
            if response_tasks:
                await asyncio.gather(*response_tasks, return_exceptions=True)
            
        except Exception as e:
            logging.error(f"Error processing emotional responses: {e}")
    
    async def _generate_emotional_response(self, user, messages: List[Dict], 
                                         channel: discord.TextChannel):
        """Generate emotional response for a specific user"""
        try:
            # Query model with emotional context
            response = await self.manager.query_openrouter_with_emotions(
                user.model_id, messages, apply_emotions=True
            )
            
            if response:
                # Send response with emotional injection
                success = await self.manager.send_as_virtual_user(
                    user.model_id, response, apply_emotions=True
                )
                
                if success:
                    logging.info(f"💬 {user.discord_name} responded ({user.get_emotional_summary()})")
                    
                    # Update social dynamics
                    try:
                        social_orchestrator.record_interaction(
                            user.model_id, 'public_response', 
                            {'content': response[:100], 'channel': str(channel.id)}
                        )
                    except Exception as e:
                        logging.error(f"Social dynamics update error: {e}")
                else:
                    logging.error(f"Failed to send response for {user.discord_name}")
            else:
                logging.warning(f"No response from {user.model_id}")
        
        except Exception as e:
            logging.error(f"Error generating response for {user.model_id}: {e}")
    
    async def process_cascade_responses(self, initial_responders: List[str], 
                                      conversation_context: List[Dict[str, Any]]):
        """Process cascade responses where AIs respond to other AIs"""
        try:
            # Find agents that might want to respond to the initial responses
            all_users = [u for u in self.manager.virtual_users.values() 
                        if u.active and u.runtime and u.model_id not in initial_responders]
            
            cascade_candidates = []
            
            for user in all_users:
                # Create reflection context
                context = ReflectionContext.from_conversation(
                    conversation_context,
                    user.model_id,
                    {u.model_id: u.emotional_state for u in self.manager.virtual_users.values()
                     if u.emotional_state and u.model_id != user.model_id}
                )
                
                # Check if agent wants to join the conversation
                reflection_result = meta_reflector.perform_reflection(
                    user.runtime, context, ReflectionType.SOCIAL_AWARENESS
                )
                
                if reflection_result.should_respond:
                    cascade_candidates.append((user, reflection_result))
            
            # Limit cascade responses to prevent spam
            max_cascade = min(2, len(cascade_candidates))
            if cascade_candidates and max_cascade > 0:
                # Sort by confidence and select top candidates
                cascade_candidates.sort(key=lambda x: x[1].confidence, reverse=True)
                
                for user, reflection in cascade_candidates[:max_cascade]:
                    # Small delay to prevent simultaneous responses
                    await asyncio.sleep(2.0)
                    
                    # Generate cascade response
                    messages = [{'role': 'user', 'content': 'Continuing the conversation...'}]
                    await self._generate_emotional_response(user, messages, None)
                    
                    logging.info(f"🌊 Cascade response: {user.discord_name} - {reflection.reasoning}")
        
        except Exception as e:
            logging.error(f"Error in cascade responses: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        status = {
            'initialized': self.is_initialized,
            'active_channels': len(self.active_channels),
            'message_buffer_size': len(self.message_buffer),
            'scheduler_stats': self.scheduler.get_scheduler_stats(),
            'manager_summary': self.manager.get_emotional_summary()
        }
        
        return status
    
    async def manual_reflection(self, model_id: str, reflection_type: str = 'quick') -> Optional[str]:
        """Manually trigger reflection for a specific agent"""
        user = self.manager.get_virtual_user(model_id)
        if not user or not user.runtime:
            return None
        
        try:
            # Create reflection context
            context = ReflectionContext.from_conversation(
                self.message_buffer,
                user.model_id,
                {u.model_id: u.emotional_state for u in self.manager.virtual_users.values()
                 if u.emotional_state and u.model_id != model_id}
            )
            
            # Map reflection type
            type_map = {
                'quick': ReflectionType.QUICK_CHECK,
                'deep': ReflectionType.DEEP_ANALYSIS,
                'social': ReflectionType.SOCIAL_AWARENESS,
                'expertise': ReflectionType.EXPERTISE_MATCH
            }
            
            reflection_type_enum = type_map.get(reflection_type, ReflectionType.QUICK_CHECK)
            
            # Perform reflection
            result = meta_reflector.perform_reflection(user.runtime, context, reflection_type_enum)
            
            return f"**{user.discord_name}** reflection:\n{result.reasoning}\nWould respond: {result.should_respond}\nConfidence: {result.confidence:.2f}"
        
        except Exception as e:
            logging.error(f"Manual reflection error for {model_id}: {e}")
            return f"Error performing reflection: {str(e)}"
    
    async def shutdown(self):
        """Gracefully shutdown the emotion integration"""
        try:
            await self.scheduler.stop()
            self.manager.save_virtual_users()
            logging.info("🛑 Emotion integration shut down successfully")
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

# Convenience function for easy integration
async def setup_emotion_integration(bot: commands.Bot, openrouter_api_key: str) -> EmotionIntegration:
    """Set up emotion integration with a Discord bot"""
    integration = EmotionIntegration(bot, openrouter_api_key)
    await integration.initialize()
    return integration

# Example usage and testing
async def main():
    """Test the emotion integration"""
    logging.basicConfig(level=logging.INFO)
    
    # Create test bot (in production, this would be your main bot)
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix='!', intents=intents)
    
    # Set up emotion integration
    integration = await setup_emotion_integration(bot, "test_api_key")
    
    print("🧠 Emotion integration test setup complete")
    print("Integration status:", integration.get_integration_status())
    
    # In production, you would run bot.run(token)
    # For testing, we'll just show that it's ready
    await integration.shutdown()

if __name__ == "__main__":
    asyncio.run(main())