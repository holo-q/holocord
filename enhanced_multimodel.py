#!/usr/bin/env python3
"""
Enhanced Multi-Model Processor with Wake/Sleep States and Meta-Reasoning
Handles cascading model interactions and intelligent response decisions
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import time

from virtual_users import VirtualUserManager

class EnhancedMultiModelProcessor:
    """Enhanced multi-model processor with wake/sleep states and cascade reactions"""
    
    def __init__(self, virtual_manager: VirtualUserManager):
        self.virtual_manager = virtual_manager
        self.active_conversations: Dict[str, List[Dict]] = {}
        # Multi-model trigger patterns
        self.triggers = {
            'all': ['everyone', 'all models', 'compare', 'all:', 'everyone:'],
            'claude': ['claude', 'anthropic'],
            'sonnet': ['sonnet', 'claude-sonnet'],
            'opus': ['opus', 'claude-opus'],
            'gemini': ['gemini', 'google', 'gemini-2.5'],
            'kimi': ['kimi', 'moonshot', 'kimi-k2', 'k2'],
            'deepseek': ['deepseek', 'r1', 'deepseek-r1'],
        }
    
    def detect_model_triggers(self, content: str, is_from_virtual_user: bool = False) -> Set[str]:
        """Detect which models should respond based on message content"""
        content_lower = content.lower()
        triggered_models = set()
        
        # Check for specific model triggers
        for model_key, triggers in self.triggers.items():
            if any(trigger in content_lower for trigger in triggers):
                if model_key == 'all':
                    if is_from_virtual_user:
                        # Virtual users can wake up sleeping models
                        all_models = set(self.virtual_manager.get_active_models())
                        for model_id in all_models:
                            self.virtual_manager.wake_user(model_id)
                        return all_models
                    else:
                        # Regular users only get awake models, but can wake specific ones
                        awake_models = set(self.virtual_manager.get_awake_models())
                        if not awake_models:  # If no models awake, wake one random model
                            all_active = self.virtual_manager.get_active_models()
                            if all_active:
                                import random
                                wake_model = random.choice(all_active)
                                self.virtual_manager.wake_user(wake_model)
                                awake_models.add(wake_model)
                        return awake_models
                else:
                    # Find specific model
                    matching_models = [
                        m for m in self.virtual_manager.get_active_models() 
                        if self._model_matches_key(m, model_key)
                    ]
                    for model_id in matching_models:
                        self.virtual_manager.wake_user(model_id)  # Wake up mentioned models
                    triggered_models.update(matching_models)
        
        return triggered_models
    
    def _model_matches_key(self, model_id: str, model_key: str) -> bool:
        """Check if a model ID matches a trigger key"""
        model_lower = model_id.lower()
        if model_key == 'claude':
            return 'claude' in model_lower
        elif model_key == 'sonnet':
            return 'sonnet' in model_lower
        elif model_key == 'opus':
            return 'opus' in model_lower
        elif model_key == 'gemini':
            return 'gemini' in model_lower
        elif model_key == 'kimi':
            return 'kimi' in model_lower
        elif model_key == 'deepseek':
            return 'deepseek' in model_lower
        return False
    
    async def should_model_respond(self, model_id: str, message_content: str, 
                                 conversation_history: List[Dict], 
                                 is_directly_triggered: bool) -> tuple[bool, str]:
        """Meta-reasoning: Should this model respond to the message?"""
        
        virtual_user = self.virtual_manager.get_virtual_user(model_id)
        if not virtual_user or not virtual_user.is_awake:
            return False, ""
        
        # If directly triggered, always respond
        if is_directly_triggered:
            return True, f"[🎯 Directly mentioned, engaging]"
        
        # Build meta-reasoning prompt
        recent_messages = conversation_history[-3:] if conversation_history else []
        context = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')[:100]}..." 
                           for msg in recent_messages])
        
        meta_prompt = f"""
You are {virtual_user.discord_name}, currently AWAKE and listening to the conversation.

Recent context:
{context}

Latest message: "{message_content}"

Should you respond to this message? Consider:
- Is this message relevant to your expertise?
- Would your response add value to the conversation?
- Are other models better suited to respond?
- Is this conversation getting too crowded with responses?

Respond with ONLY:
"RESPOND: [brief reason why you should respond]"
OR 
"PASS: [brief reason why you're passing]"
"""
        
        try:
            # Quick meta-reasoning call
            meta_response = await self.virtual_manager.query_openrouter(
                model_id, 
                [{"role": "user", "content": meta_prompt}],
                system_prompt="You are deciding whether to participate in a conversation. Be concise and decisive."
            )
            
            if meta_response and "RESPOND:" in meta_response.upper():
                reason = meta_response.split(":", 1)[1].strip()
                return True, f"[🧠 {reason[:50]}...]"
            elif meta_response and "PASS:" in meta_response.upper():
                reason = meta_response.split(":", 1)[1].strip()
                logging.info(f"{virtual_user.discord_name} chose to pass: {reason}")
                return False, ""
            else:
                # Default to not responding if unclear
                return False, ""
                
        except Exception as e:
            logging.error(f"Meta-reasoning failed for {model_id}: {e}")
            return False, ""
    
    async def process_message(self, message, messages: List[Dict], system_prompt: Optional[str] = None) -> bool:
        """Process message with enhanced wake/sleep and meta-reasoning"""
        
        # Check if message is from a virtual user (webhook)
        is_from_virtual_user = bool(message.webhook_id)
        
        # Check if bot was mentioned or replying to virtual user
        bot_mentioned = hasattr(message, 'mentions') and any(
            mention.id == self.virtual_manager.bot.user.id for mention in message.mentions
        )
        
        replying_to_virtual_user = False
        if message.reference:
            try:
                referenced_msg = await message.channel.fetch_message(message.reference.message_id)
                if referenced_msg.webhook_id:
                    replying_to_virtual_user = True
            except:
                pass
        
        # Detect which models should respond
        triggered_models = self.detect_model_triggers(message.content, is_from_virtual_user)
        
        # If bot mentioned or replying to virtual user, consider awake models
        if (bot_mentioned or replying_to_virtual_user) and not triggered_models:
            awake_models = set(self.virtual_manager.get_awake_models())
            if awake_models:
                triggered_models = awake_models
            else:
                # No models awake, wake a random one
                all_models = self.virtual_manager.get_active_models()
                if all_models:
                    import random
                    wake_model = random.choice(all_models)
                    self.virtual_manager.wake_user(wake_model)
                    triggered_models = {wake_model}
        
        if not triggered_models:
            return False
        
        logging.info(f"🎯 Models triggered: {triggered_models}")
        
        # Show typing indicator while processing models
        async with message.channel.typing():
            # Build conversation context
            channel_id = str(message.channel.id)
            conversation_messages = self.build_conversation_context(messages, channel_id)
            
            # Determine which models should actually respond using meta-reasoning
            responding_models = []
            for model_id in triggered_models:
                is_directly_mentioned = any(
                    trigger in message.content.lower() 
                    for trigger_list in self.triggers.values() 
                    for trigger in trigger_list
                    if self._model_matches_key(model_id, trigger)
                )
                
                should_respond, reasoning = await self.should_model_respond(
                    model_id, message.content, conversation_messages, is_directly_mentioned
                )
                
                if should_respond:
                    responding_models.append((model_id, reasoning))
                    # Refresh wake time for responding models
                    self.virtual_manager.refresh_wake_time(model_id)
            
            if not responding_models:
                logging.info("🤔 All models chose to pass on this message")
                return False
            
            # Process each model response concurrently
            tasks = []
            for model_id, reasoning in responding_models:
                task = self.query_and_respond_with_reasoning(
                    model_id, conversation_messages, system_prompt, reasoning
                )
                tasks.append(task)
            
            if tasks:
                # Execute all model queries concurrently
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update conversation context
                self.active_conversations[channel_id] = conversation_messages
                
                return True
        
        return False
    
    async def query_and_respond_with_reasoning(self, model_id: str, messages: List[Dict], 
                                            system_prompt: Optional[str], reasoning: str):
        """Query model and send response with meta-reasoning"""
        try:
            # Query the model
            response = await self.virtual_manager.query_openrouter(model_id, messages, system_prompt)
            
            if response:
                # Add reasoning prefix to response
                full_response = f"{reasoning}\n\n{response}" if reasoning else response
                
                # Send response via webhook
                success = await self.virtual_manager.send_as_virtual_user(model_id, full_response)
                
                if success:
                    virtual_user = self.virtual_manager.get_virtual_user(model_id)
                    logging.info(f"✅ {virtual_user.discord_name} responded with reasoning")
                    
                    # Add response to conversation context
                    messages.append({
                        "role": "assistant", 
                        "content": response,
                        "name": virtual_user.discord_name
                    })
                else:
                    logging.error(f"❌ Failed to send response from {model_id}")
            else:
                logging.error(f"❌ No response from model {model_id}")
                
        except Exception as e:
            logging.error(f"❌ Error in query_and_respond_with_reasoning for {model_id}: {e}")
    
    def build_conversation_context(self, messages: List[Dict], channel_id: str) -> List[Dict]:
        """Build conversation context with recent history"""
        # Get recent conversation from this channel
        recent_conversation = self.active_conversations.get(channel_id, [])
        
        # Combine with current messages (limit to last 10 total)
        all_messages = recent_conversation + messages
        return all_messages[-10:] if len(all_messages) > 10 else all_messages

# Helper function to update channel status with wake/sleep indicators
async def update_devrooms_status(discord_bot, virtual_manager):
    """Update devrooms channel topic with wake/sleep status"""
    try:
        # Find devrooms channel
        devrooms_channel = None
        for guild in discord_bot.guilds:
            for channel in guild.text_channels:
                if channel.name == "devrooms":
                    devrooms_channel = channel
                    break
            if devrooms_channel:
                break
        
        if not devrooms_channel:
            return
        
        # Get wake/sleep status
        awake_models = virtual_manager.get_awake_models()
        sleeping_models = virtual_manager.get_sleeping_models()
        
        awake_hints = []
        sleeping_hints = []
        
        for model_id in awake_models:
            user = virtual_manager.get_virtual_user(model_id)
            if user:
                if "claude-opus" in model_id.lower():
                    awake_hints.append("🔥opus")
                elif "claude-sonnet" in model_id.lower():
                    awake_hints.append("🔥sonnet") 
                elif "gemini" in model_id.lower():
                    awake_hints.append("🔥gemini")
                elif "kimi" in model_id.lower():
                    awake_hints.append("🔥kimi")
                elif "deepseek" in model_id.lower():
                    awake_hints.append("🔥deepseek")
        
        for model_id in sleeping_models:
            user = virtual_manager.get_virtual_user(model_id)
            if user:
                if "claude-opus" in model_id.lower():
                    sleeping_hints.append("😴opus")
                elif "claude-sonnet" in model_id.lower():
                    sleeping_hints.append("😴sonnet") 
                elif "gemini" in model_id.lower():
                    sleeping_hints.append("😴gemini")
                elif "kimi" in model_id.lower():
                    sleeping_hints.append("😴kimi")
                elif "deepseek" in model_id.lower():
                    sleeping_hints.append("😴deepseek")
        
        status_parts = []
        if awake_hints:
            status_parts.append(f"AWAKE: {' '.join(awake_hints)}")
        if sleeping_hints:
            status_parts.append(f"SLEEPING: {' '.join(sleeping_hints)}")
        
        topic = f"🧠 Smart Multi-Model Chat | {' | '.join(status_parts)} | Mention models to wake them!"
        
        await devrooms_channel.edit(topic=topic)
        logging.info(f"🔄 Updated status: {topic}")
        
    except Exception as e:
        logging.error(f"Failed to update devrooms status: {e}")