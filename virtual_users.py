#!/usr/bin/env python3
"""
Virtual Users Manager for Discord Multi-Model Bot
Manages virtual users (AI models) via webhooks and slash commands
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
import logging

import discord
from discord import app_commands
from discord.ext import commands
import httpx
import yaml

@dataclass
class VirtualUser:
    model_id: str
    discord_name: str
    webhook_url: str
    webhook_id: str
    avatar_url: Optional[str] = None
    description: str = ""
    pricing: Dict[str, float] = None
    context_length: int = 0
    active: bool = True
    is_awake: bool = False
    last_activity: float = 0.0  # timestamp
    wake_duration: int = 600  # 10 minutes in seconds
    
    def __post_init__(self):
        if self.pricing is None:
            self.pricing = {}

class VirtualUserManager:
    def __init__(self, bot: commands.Bot, openrouter_api_key: str):
        self.bot = bot
        self.openrouter_api_key = openrouter_api_key
        self.virtual_users: Dict[str, VirtualUser] = {}
        self.active_channels: Set[str] = set()
        self.load_virtual_users()
    
    def load_virtual_users(self):
        """Load virtual users from JSON file"""
        try:
            with open('virtual_users.json', 'r') as f:
                data = json.load(f)
                for user_data in data.get('users', []):
                    user = VirtualUser(**user_data)
                    self.virtual_users[user.model_id] = user
            logging.info(f"Loaded {len(self.virtual_users)} virtual users")
        except FileNotFoundError:
            logging.info("No virtual users file found, starting fresh")
        except Exception as e:
            logging.error(f"Error loading virtual users: {e}")
    
    def save_virtual_users(self):
        """Save virtual users to JSON file"""
        try:
            data = {
                'users': [asdict(user) for user in self.virtual_users.values()],
                'active_channels': list(self.active_channels)
            }
            with open('virtual_users.json', 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Saved {len(self.virtual_users)} virtual users")
        except Exception as e:
            logging.error(f"Error saving virtual users: {e}")
    
    async def create_virtual_user(self, channel: discord.TextChannel, model_id: str, 
                                discord_name: Optional[str] = None) -> VirtualUser:
        """Create a new virtual user with webhook"""
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
                reason=f"Virtual user for {model_id}"
            )
            
            # Get avatar URL based on model
            avatar_url = self.get_model_avatar_url(model_id)
            
            # Create virtual user
            user = VirtualUser(
                model_id=model_id,
                discord_name=discord_name,
                webhook_url=webhook.url,
                webhook_id=str(webhook.id),
                avatar_url=avatar_url,
                description=model_info.get('description', ''),
                pricing=model_info.get('pricing', {}),
                context_length=model_info.get('context_length', 0)
            )
            
            self.virtual_users[model_id] = user
            self.save_virtual_users()
            
            logging.info(f"Created virtual user {discord_name} for {model_id}")
            return user
            
        except Exception as e:
            logging.error(f"Error creating virtual user: {e}")
            raise
    
    async def remove_virtual_user(self, model_id: str) -> bool:
        """Remove a virtual user and delete its webhook"""
        if model_id not in self.virtual_users:
            return False
        
        try:
            user = self.virtual_users[model_id]
            
            # Delete webhook
            async with httpx.AsyncClient() as client:
                await client.delete(f"https://discord.com/api/webhooks/{user.webhook_id}")
            
            del self.virtual_users[model_id]
            self.save_virtual_users()
            
            logging.info(f"Removed virtual user {user.discord_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error removing virtual user: {e}")
            return False
    
    async def send_as_virtual_user(self, model_id: str, content: str, 
                                  embeds: Optional[List[discord.Embed]] = None) -> bool:
        """Send a message as a virtual user via webhook"""
        if model_id not in self.virtual_users:
            return False
        
        try:
            user = self.virtual_users[model_id]
            
            webhook_data = {
                'content': content,
                'username': user.discord_name,
            }
            
            if user.avatar_url:
                webhook_data['avatar_url'] = user.avatar_url
            
            if embeds:
                webhook_data['embeds'] = [embed.to_dict() for embed in embeds]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(user.webhook_url, json=webhook_data)
                response.raise_for_status()
            
            return True
            
        except Exception as e:
            logging.error(f"Error sending message as {model_id}: {e}")
            return False
    
    async def query_openrouter(self, model_id: str, messages: List[Dict], 
                             system_prompt: Optional[str] = None) -> Optional[str]:
        """Query OpenRouter API for model response"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/holo-q/llmcord",
                "X-Title": "HOLO-Q Discord Multi-Model Bot"
            }
            
            request_data = {
                "model": model_id,
                "messages": messages,
                "stream": False,
                # "max_tokens": 4000,  # Removed - let models use their natural length
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
    
    def get_active_models(self) -> List[str]:
        """Get list of active model IDs"""
        return [model_id for model_id, user in self.virtual_users.items() if user.active]
    
    def get_virtual_user(self, model_id: str) -> Optional[VirtualUser]:
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
    
    def wake_user(self, model_id: str) -> bool:
        """Wake up a virtual user"""
        import time
        if model_id in self.virtual_users:
            user = self.virtual_users[model_id]
            user.is_awake = True
            user.last_activity = time.time()
            logging.info(f"🔥 {user.discord_name} is now AWAKE")
            return True
        return False
    
    def refresh_wake_time(self, model_id: str) -> bool:
        """Refresh wake time for a virtual user"""
        import time
        if model_id in self.virtual_users:
            user = self.virtual_users[model_id]
            if user.is_awake:
                user.last_activity = time.time()
                return True
        return False
    
    def check_sleep_timers(self):
        """Check and sleep users who have been inactive"""
        import time
        current_time = time.time()
        for model_id, user in self.virtual_users.items():
            if user.is_awake and (current_time - user.last_activity) > user.wake_duration:
                user.is_awake = False
                logging.info(f"😴 {user.discord_name} fell asleep")
    
    def get_awake_models(self) -> List[str]:
        """Get list of awake model IDs"""
        self.check_sleep_timers()  # Check for sleeping models first
        return [model_id for model_id, user in self.virtual_users.items() if user.active and user.is_awake]
    
    def get_sleeping_models(self) -> List[str]:
        """Get list of sleeping model IDs"""
        self.check_sleep_timers()  # Check for sleeping models first
        return [model_id for model_id, user in self.virtual_users.items() if user.active and not user.is_awake]

# Discord slash commands
class VirtualUserCommands(commands.Cog):
    def __init__(self, bot: commands.Bot, manager: VirtualUserManager):
        self.bot = bot
        self.manager = manager
    
    @app_commands.command(
        name="add_model",
        description="Add a new AI model as a virtual user (Admin only)"
    )
    @app_commands.describe(
        model_id="OpenRouter model ID (e.g., anthropic/claude-3-5-sonnet)",
        name="Custom Discord name (optional)"
    )
    async def add_model(self, interaction: discord.Interaction, model_id: str, name: Optional[str] = None):
        """Add a new model as virtual user"""
        await interaction.response.defer()
        
        # Check admin permissions
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            admin_ids = config['permissions']['users']['admin_ids']
            if interaction.user.id not in admin_ids:
                await interaction.followup.send("❌ You don't have permission to add models. Admin only.", ephemeral=True)
                return
        except Exception as e:
            await interaction.followup.send("❌ Error checking permissions", ephemeral=True)
            return
        
        try:
            if model_id in self.manager.virtual_users:
                await interaction.followup.send(f"❌ Model `{model_id}` already exists as virtual user!")
                return
            
            user = await self.manager.create_virtual_user(
                interaction.channel, model_id, name
            )
            
            embed = discord.Embed(
                title="✅ Virtual User Added",
                description=f"**{user.discord_name}** (`{model_id}`)",
                color=discord.Color.green()
            )
            embed.add_field(name="Context", value=f"{user.context_length:,}", inline=True)
            
            if user.pricing.get('prompt'):
                price = float(user.pricing['prompt']) * 1000000
                embed.add_field(name="Cost", value=f"${price:.2f}/1M tokens", inline=True)
            
            if user.description:
                embed.add_field(
                    name="Description", 
                    value=user.description[:200] + "..." if len(user.description) > 200 else user.description,
                    inline=False
                )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            await interaction.followup.send(f"❌ Error adding model: {str(e)}")
    
    @app_commands.command(
        name="remove_model",
        description="Remove an AI model virtual user (Admin only)"
    )
    @app_commands.describe(model_id="Model ID to remove")
    async def remove_model(self, interaction: discord.Interaction, model_id: str):
        """Remove a virtual user"""
        await interaction.response.defer()
        
        # Check admin permissions
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            admin_ids = config['permissions']['users']['admin_ids']
            if interaction.user.id not in admin_ids:
                await interaction.followup.send("❌ You don't have permission to remove models. Admin only.", ephemeral=True)
                return
        except Exception as e:
            await interaction.followup.send("❌ Error checking permissions", ephemeral=True)
            return
        
        success = await self.manager.remove_virtual_user(model_id)
        
        if success:
            await interaction.followup.send(f"✅ Removed virtual user for `{model_id}`")
        else:
            await interaction.followup.send(f"❌ Virtual user `{model_id}` not found")
    
    @app_commands.command(
        name="list_models",
        description="List all active virtual users"
    )
    async def list_models(self, interaction: discord.Interaction):
        """List all virtual users"""
        await interaction.response.defer()
        
        if not self.manager.virtual_users:
            await interaction.followup.send("No virtual users configured yet!")
            return
        
        embed = discord.Embed(
            title="🤖 Active Virtual Users",
            color=discord.Color.blue()
        )
        
        for model_id, user in self.manager.virtual_users.items():
            status = "✅" if user.active else "❌"
            price_info = ""
            
            if user.pricing.get('prompt'):
                try:
                    price = float(user.pricing['prompt']) * 1000000
                    price_info = f" (${price:.2f}/1M)"
                except:
                    pass
            
            embed.add_field(
                name=f"{status} {user.discord_name}",
                value=f"`{model_id}`{price_info}",
                inline=True
            )
        
        await interaction.followup.send(embed=embed)
    
    @app_commands.command(
        name="available_models",
        description="Show available OpenRouter models that can be added"
    )
    async def available_models(self, interaction: discord.Interaction):
        """Show available models from OpenRouter"""
        await interaction.response.defer()
        
        try:
            with open('openrouter_models.json', 'r') as f:
                data = json.load(f)
            
            embed = discord.Embed(
                title="📋 Available Models",
                description=f"Top {min(15, len(data['models']))} models from OpenRouter",
                color=discord.Color.blue()
            )
            
            for i, model in enumerate(data['models'][:15], 1):
                status = "✅" if model['id'] in self.manager.virtual_users else "⭕"
                
                try:
                    price = float(model['pricing'].get('prompt', 0)) * 1000000
                    price_str = f" (${price:.2f}/1M)"
                except:
                    price_str = ""
                
                embed.add_field(
                    name=f"{status} {model['discord_name']}",
                    value=f"`{model['id']}`{price_str}",
                    inline=True
                )
            
            embed.set_footer(text="Use /add_model <model_id> to add a model")
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            await interaction.followup.send(f"❌ Error loading available models: {str(e)}")
    
    @app_commands.command(
        name="triggers",
        description="Show all available model triggers for text-based activation"
    )
    async def show_triggers(self, interaction: discord.Interaction):
        """Show available model triggers"""
        await interaction.response.defer()
        
        embed = discord.Embed(
            title="🎯 Model Triggers",
            description="Type these words in your message to activate AI models:",
            color=discord.Color.green()
        )
        
        # Add trigger examples
        trigger_examples = {
            "🤖 **Claude Models**": "`claude` - Both Claude models\n`opus` - Claude Opus only\n`sonnet` - Claude Sonnet only",
            "🧠 **Other Models**": "`gemini` - Google Gemini\n`kimi` - Moonshot Kimi\n`deepseek` or `r1` - DeepSeek R1",
            "🌐 **All Models**": "`everyone` or `all models` - Triggers ALL models\n`compare` - Good for comparisons"
        }
        
        for category, triggers in trigger_examples.items():
            embed.add_field(
                name=category,
                value=triggers,
                inline=False
            )
        
        embed.add_field(
            name="📝 **Example Messages**",
            value='• "Hey claude, what\'s 2+2?"\n• "opus and gemini, compare these approaches"\n• "everyone: solve this problem"\n• "deepseek, help me debug this code"',
            inline=False
        )
        
        embed.set_footer(text="💡 Tip: You can combine multiple triggers in one message!")
        await interaction.followup.send(embed=embed)
    
    @app_commands.command(
        name="test_model",
        description="Test a specific model with a prompt (Admin only)"
    )
    @app_commands.describe(
        model_id="Model ID to test",
        prompt="Test prompt to send"
    )
    async def test_model(self, interaction: discord.Interaction, model_id: str, prompt: str):
        """Test a specific model"""
        await interaction.response.defer()
        
        # Check admin permissions
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            admin_ids = config['permissions']['users']['admin_ids']
            if interaction.user.id not in admin_ids:
                await interaction.followup.send("❌ You don't have permission to test models. Admin only.", ephemeral=True)
                return
        except Exception as e:
            await interaction.followup.send("❌ Error checking permissions", ephemeral=True)
            return
        
        if model_id not in self.manager.virtual_users:
            await interaction.followup.send(f"❌ Virtual user `{model_id}` not found. Add it first with `/add_model`")
            return
        
        try:
            user = self.manager.virtual_users[model_id]
            
            # Query the model
            messages = [{"role": "user", "content": prompt}]
            response = await self.manager.query_openrouter(model_id, messages)
            
            if response:
                # Send via webhook to simulate real usage
                success = await self.manager.send_as_virtual_user(model_id, response)
                
                if success:
                    await interaction.followup.send(f"✅ Test message sent as **{user.discord_name}**")
                else:
                    await interaction.followup.send(f"❌ Failed to send via webhook, but model responded: {response[:200]}...")
            else:
                await interaction.followup.send(f"❌ Model `{model_id}` failed to respond")
                
        except Exception as e:
            await interaction.followup.send(f"❌ Error testing model: {str(e)}")

if __name__ == "__main__":
    # Test the virtual user manager
    print("Virtual User Manager ready for integration with llmcord.py")