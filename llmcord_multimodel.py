#!/usr/bin/env python3
"""
Enhanced LLMCord with Multi-Model Virtual Users
Integrates with virtual_users.py for auto-virtual user management
"""

import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional, Dict, List, Set
import json
import re

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml

from virtual_users import VirtualUserManager, VirtualUserCommands
from simple_status_integration import add_simple_status_to_bot
from live_parameter_hud import setup_live_hud
from emotion_engine.monitoring import production_monitor
from emotion_plotting import start_emotion_plotting, stop_emotion_plotting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500

# Multi-model trigger patterns
MULTI_MODEL_TRIGGERS = {
    'all': ['everyone', 'all models', 'compare', 'all:', 'everyone:'],
    'claude': ['claude', 'anthropic'],
    'sonnet': ['sonnet', 'claude-sonnet'],
    'opus': ['opus', 'claude-opus'],
    'gemini': ['gemini', 'google', 'gemini-2.5'],
    'kimi': ['kimi', 'moonshot', 'kimi-k2', 'k2'],
    'deepseek': ['deepseek', 'r1', 'deepseek-r1'],
}

def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)

config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True  # Required to read message content for trigger detection
activity = discord.CustomActivity(name=(config["status_message"] or "Multi-Model HOLO-Q Discord Bot")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()

# Initialize virtual user manager
openrouter_key = config["providers"]["openrouter"]["api_key"]
virtual_manager = VirtualUserManager(discord_bot, openrouter_key)

@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)
    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    parent_msg: Optional[discord.Message] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

class MultiModelProcessor:
    """Handles multi-model responses and context sharing"""
    
    def __init__(self, virtual_manager: VirtualUserManager):
        self.virtual_manager = virtual_manager
        self.active_conversations: Dict[str, List[Dict]] = {}
    
    def detect_model_triggers(self, content: str) -> Set[str]:
        """Detect which models should respond based on message content"""
        content_lower = content.lower()
        triggered_models = set()
        
        # Check for specific model triggers
        for model_key, triggers in MULTI_MODEL_TRIGGERS.items():
            if any(trigger in content_lower for trigger in triggers):
                if model_key == 'all':
                    # Trigger all active models
                    return set(self.virtual_manager.get_active_models())
                elif model_key == 'claude':
                    triggered_models.update([
                        m for m in self.virtual_manager.get_active_models() 
                        if 'claude' in m.lower()
                    ])
                elif model_key == 'sonnet':
                    triggered_models.update([
                        m for m in self.virtual_manager.get_active_models() 
                        if 'sonnet' in m.lower()
                    ])
                elif model_key == 'opus':
                    triggered_models.update([
                        m for m in self.virtual_manager.get_active_models() 
                        if 'opus' in m.lower()
                    ])
                elif model_key == 'gemini':
                    triggered_models.update([
                        m for m in self.virtual_manager.get_active_models() 
                        if 'gemini' in m.lower()
                    ])
                elif model_key == 'kimi':
                    triggered_models.update([
                        m for m in self.virtual_manager.get_active_models() 
                        if 'kimi' in m.lower()
                    ])
                elif model_key == 'deepseek':
                    triggered_models.update([
                        m for m in self.virtual_manager.get_active_models() 
                        if 'deepseek' in m.lower()
                    ])
        
        # If no specific triggers found, check if it's in devrooms and bot was mentioned
        if not triggered_models:
            return set()
        
        return triggered_models
    
    def build_conversation_context(self, messages: List[Dict], channel_id: str) -> List[Dict]:
        """Build conversation context including previous virtual user messages"""
        # Get stored context for this channel
        stored_context = self.active_conversations.get(channel_id, [])
        
        # Combine with current messages, keeping context manageable
        all_messages = stored_context + messages
        
        # Keep last 20 messages to maintain context but not exceed limits
        if len(all_messages) > 20:
            all_messages = all_messages[-20:]
        
        return all_messages
    
    async def process_multi_model_response(self, message: discord.Message, 
                                         messages: List[Dict], 
                                         system_prompt: Optional[str] = None):
        """Process message and generate responses from multiple models"""
        
        if message.channel.name != 'devrooms':
            return False  # Only work in devrooms for now
        
        # Check if bot was mentioned or if it's a reply to a virtual user
        bot_mentioned = discord_bot.user in message.mentions
        replying_to_virtual_user = False
        
        if message.reference:
            try:
                referenced_msg = await message.channel.fetch_message(message.reference.message_id)
                # Check if referenced message was from a webhook (virtual user)
                if referenced_msg.webhook_id:
                    replying_to_virtual_user = True
            except:
                pass
        
        # Detect which models should respond
        triggered_models = self.detect_model_triggers(message.content)
        
        # If bot mentioned or replying to virtual user, trigger all models
        if (bot_mentioned or replying_to_virtual_user) and not triggered_models:
            triggered_models = set(self.virtual_manager.get_active_models())
        
        if not triggered_models:
            return False
        
        logging.info(f"Multi-model response triggered for models: {triggered_models}")
        
        # Show typing indicator while processing models
        async with message.channel.typing():
            # Build conversation context
            channel_id = str(message.channel.id)
            conversation_messages = self.build_conversation_context(messages, channel_id)
            
            # Process each model response concurrently
            tasks = []
            for model_id in triggered_models:
                virtual_user = self.virtual_manager.get_virtual_user(model_id)
                if virtual_user and virtual_user.active:
                    task = self.query_and_respond(model_id, conversation_messages, system_prompt)
                    tasks.append(task)
            
            if tasks:
                # Execute all model queries concurrently
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update conversation context
                self.active_conversations[channel_id] = conversation_messages
                
                return True
        
        return False
    
    async def query_and_respond(self, model_id: str, messages: List[Dict], 
                              system_prompt: Optional[str] = None):
        """Query a specific model and send response via webhook"""
        try:
            # Add typing indicator
            # Note: Webhooks can't show typing, but we could add a brief delay
            
            # Query the model
            response = await self.virtual_manager.query_openrouter(model_id, messages, system_prompt)
            
            if response:
                # Send response via webhook
                success = await self.virtual_manager.send_as_virtual_user(model_id, response)
                
                if success:
                    logging.info(f"Model {model_id} responded successfully")
                    
                    # Add the virtual user's response to conversation context
                    virtual_user = self.virtual_manager.get_virtual_user(model_id)
                    if virtual_user:
                        messages.append({
                            "role": "assistant", 
                            "content": response,
                            "name": virtual_user.discord_name  # Track which model responded
                        })
                else:
                    logging.error(f"Failed to send response from {model_id}")
            else:
                logging.error(f"No response from model {model_id}")
                
        except Exception as e:
            logging.error(f"Error processing {model_id}: {e}")

# Initialize multi-model processor
multi_model_processor = MultiModelProcessor(virtual_manager)

@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))

@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()][:24]
    choices += [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []

    return choices

async def update_devrooms_topic():
    """Update devrooms channel topic with available model triggers"""
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
            logging.warning("Devrooms channel not found")
            return
        
        # Build topic with model triggers
        active_models = virtual_manager.get_active_models()
        model_hints = []
        
        for model_id in active_models:
            user = virtual_manager.get_virtual_user(model_id)
            if user:
                if "claude-opus" in model_id.lower():
                    model_hints.append("🤖 opus")
                elif "claude-sonnet" in model_id.lower():
                    model_hints.append("🤖 sonnet") 
                elif "gemini" in model_id.lower():
                    model_hints.append("🧠 gemini")
                elif "kimi" in model_id.lower():
                    model_hints.append("🚀 kimi")
                elif "deepseek" in model_id.lower():
                    model_hints.append("🔍 deepseek")
        
        # Enhanced topic with new features
        topic = f"💬 Multi-Model AI Chat | Triggers: {' • '.join(model_hints)} • everyone | 📊 /status • /create-dev-project"
        
        await devrooms_channel.edit(topic=topic)
        logging.info(f"Updated devrooms topic: {topic}")
        
    except Exception as e:
        logging.error(f"Failed to update devrooms topic: {e}")

@discord_bot.event
async def on_ready() -> None:
    if client_id := config["client_id"]:
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    # Add virtual user commands
    await discord_bot.add_cog(VirtualUserCommands(discord_bot, virtual_manager))
    
    await discord_bot.tree.sync()
    
    # Update devrooms channel topic with model hints
    await update_devrooms_topic()
    
    logging.info(f"Multi-Model Discord Bot ready!")
    logging.info(f"Active virtual users: {len(virtual_manager.virtual_users)}")
    for model_id, user in virtual_manager.virtual_users.items():
        logging.info(f"  - {user.discord_name} ({model_id})")
    
    # Add /status command for agent monitoring
    try:
        await add_simple_status_to_bot(discord_bot, virtual_manager.virtual_users)
        logging.info("✅ Added /status command - request detailed agent states (hidden from LLM context)")
    except Exception as e:
        logging.error(f"❌ Failed to add /status command: {e}")
    
    # Initialize emotion monitoring with real agents
    try:
        from genome.base import AgentRuntime, BaseStats, ExtendedGenome, GenomeCore
        from genome.types import EmotionState, ConsciousnessLevel
        import time
        
        # Create real AgentRuntime for each virtual user
        for agent_id, virtual_user in virtual_manager.virtual_users.items():
            # Create basic genome and runtime for this agent
            base_stats = BaseStats(
                curiosity_base=0.5574,  # From optimal config
                confidence_base=0.3433,
                social_energy_base=0.7805,
                restlessness_amplitude=0.2,
                harmony_factor=0.4639
            )
            
            genome_core = GenomeCore(
                model_id=agent_id,
                base_stats=base_stats
            )
            
            extended_genome = ExtendedGenome(core=genome_core)
            
            initial_state = base_stats.to_initial_state()
            
            runtime = AgentRuntime(
                agent_id=agent_id,
                genome=extended_genome,
                current_state=initial_state
            )
            
            # Register with production monitor
            production_monitor.register_agent(agent_id, runtime)
            
            # Initial state update to populate history
            production_monitor.record_state_update(
                agent_id, 
                initial_state,
                initial_state
            )
        
        logging.info(f"✅ Initialized emotion monitoring for {len(virtual_manager.virtual_users)} agents")
    except Exception as e:
        logging.error(f"❌ Failed to initialize emotion monitoring: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    # Setup live parameter HUD (auto-starts)
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
        
        auto_start_config = {"enabled": True, "channel_id": devrooms_channel.id if devrooms_channel else None}
        await setup_live_hud(discord_bot, auto_start_config)
        logging.info("✅ Setup live parameter HUD with auto-start")
    except Exception as e:
        logging.error(f"❌ Failed to setup live parameter HUD: {e}")
    
    # Start real-time emotion plotting system
    try:
        await start_emotion_plotting()
        logging.info("✅ Started real-time emotion parameter plotting system")
    except Exception as e:
        logging.error(f"❌ Failed to start emotion plotting system: {e}")
    
    # Add manual topic update command
    @discord_bot.tree.command(name="update-topic", description="Update devrooms channel topic with latest features")
    async def update_topic_command(interaction: discord.Interaction):
        """Manually update devrooms topic"""
        if interaction.user.id not in config["permissions"]["users"]["admin_ids"]:
            await interaction.response.send_message("❌ Admin only command", ephemeral=True)
            return
        
        await interaction.response.defer(ephemeral=True)
        
        try:
            await update_devrooms_topic()
            await interaction.followup.send("✅ Updated devrooms topic with latest features", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to update topic: {str(e)}", ephemeral=True)
    
    await discord_bot.tree.sync()
    logging.info("✅ Added /update-topic command")

@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        # Skip webhook messages (from virtual users) to avoid loops
        if new_msg.webhook_id:
            return
    
    # Permission checking (existing code)
    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)
    allow_dms = config.get("allow_dms", True)
    permissions = config["permissions"]
    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    # Build message chain (existing code)
    messages = []
    user_warnings = set()
    curr_msg = new_msg
    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5)
    max_messages = config.get("max_messages", 25)

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]
                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"
                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                # Parent message fetching (existing code)
                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                messages.append(message)

            # Warning handling (existing code)
            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := config["system_prompt"]:
        now = datetime.now().astimezone()
        # Add user context
        user_name = new_msg.author.display_name or new_msg.author.name
        user_context = f"You are currently talking with {user_name} ({new_msg.author.name})"
        
        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        system_prompt += f"\n\n{user_context}"
        
        if PROVIDERS_SUPPORTING_USERNAMES:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."
        system_prompt += "\n\nYou are participating in a multi-model conversation. Other AI models may also respond to this message."

    # TRY MULTI-MODEL PROCESSING FIRST
    multi_model_handled = await multi_model_processor.process_multi_model_response(new_msg, messages, system_prompt)
    
    if multi_model_handled:
        logging.info("Message handled by multi-model system")
        return
    
    # FALLBACK TO ORIGINAL SINGLE-MODEL RESPONSE (existing code continues...)
    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
    provider_config = config["providers"][provider]
    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)
    extra_headers = provider_config.get("extra_headers", None)
    extra_query = provider_config.get("extra_query", None)
    extra_body = (provider_config.get("extra_body", None) or {}) | (model_parameters or {}) or None

    if system_prompt:
        messages.append(dict(role="system", content=system_prompt))

    # Generate single-model response (rest of existing code)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)
        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason != None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason
                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""
                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time
                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception:
        logging.exception("Error while generating response")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (existing code)
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)

async def main() -> None:
    await discord_bot.start(config["bot_token"])

try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass