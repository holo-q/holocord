#!/usr/bin/env python3
"""
🧵 Dev Thread Manager for LLMCord
Enhanced thread support for development work projects in devrooms
"""

import discord
from discord.ext import commands
from typing import Dict, List, Optional, Set
import asyncio
import logging
from datetime import datetime, timedelta

class DevThreadManager:
    """Manages development threads and project assignments"""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.logger = logging.getLogger("DevThreadManager")
        
        # Track active dev threads
        self.dev_threads: Dict[int, Dict] = {}  # thread_id -> thread_info
        self.project_assignments: Dict[str, Set[int]] = {}  # project_name -> set of user_ids
        self.thread_models: Dict[int, Set[str]] = {}  # thread_id -> set of model_names
        
    def is_devroom(self, channel: discord.TextChannel) -> bool:
        """Check if channel is a devroom"""
        return hasattr(channel, 'name') and 'dev' in channel.name.lower()
    
    async def create_dev_thread(self, channel: discord.TextChannel, 
                               project_name: str, 
                               description: str,
                               assigned_models: List[str] = None,
                               ping_users: List[discord.User] = None) -> Optional[discord.Thread]:
        """Create a new development thread for a project"""
        
        if not self.is_devroom(channel):
            self.logger.warning(f"Attempted to create dev thread in non-devroom: {channel.name}")
            return None
        
        try:
            # Create the thread
            thread_name = f"🛠️ {project_name}"
            
            # Create initial message for the thread
            initial_content = f"# 🛠️ Project: {project_name}\n\n**Description:** {description}\n\n"
            
            if assigned_models:
                model_mentions = " ".join([f"`{model}`" for model in assigned_models])
                initial_content += f"**Assigned Models:** {model_mentions}\n\n"
            
            if ping_users:
                user_mentions = " ".join([user.mention for user in ping_users])
                initial_content += f"**Team:** {user_mentions}\n\n"
            
            initial_content += "**Status:** 🟡 Starting\n\n---\n\nReady to collaborate! 🚀"
            
            # Send message first, then create thread from it
            initial_message = await channel.send(initial_content)
            thread = await initial_message.create_thread(
                name=thread_name,
                auto_archive_duration=10080  # 7 days
            )
            
            # Track the thread
            self.dev_threads[thread.id] = {
                'project_name': project_name,
                'description': description,
                'created_at': datetime.utcnow(),
                'status': 'active',
                'assigned_models': assigned_models or [],
                'initial_message_id': initial_message.id,
                'creator_id': None  # Will be set by caller
            }
            
            if assigned_models:
                self.thread_models[thread.id] = set(assigned_models)
            
            # Send welcome message to models in thread
            if assigned_models:
                model_welcome = f"👋 **Models assigned to this project:**\n"
                for model in assigned_models:
                    model_welcome += f"• `{model}` - Ready to assist with {project_name}!\n"
                
                model_welcome += f"\n💡 **Tip:** You can discuss implementation, ask questions, share code, and collaborate on this project here."
                await thread.send(model_welcome)
            
            self.logger.info(f"Created dev thread '{project_name}' in {channel.name}")
            return thread
            
        except Exception as e:
            self.logger.error(f"Failed to create dev thread: {e}")
            return None
    
    async def ping_models_in_thread(self, thread: discord.Thread, 
                                   models: List[str], 
                                   message: str,
                                   urgent: bool = False) -> bool:
        """Ping specific models in a thread with a message"""
        
        try:
            if thread.id not in self.dev_threads:
                self.logger.warning(f"Thread {thread.id} not tracked as dev thread")
                return False
            
            urgency_icon = "🚨" if urgent else "📢"
            ping_content = f"{urgency_icon} **Model Ping**\n\n"
            
            if models:
                model_list = ", ".join([f"`{model}`" for model in models])
                ping_content += f"**Calling:** {model_list}\n\n"
                
                # Add models to thread tracking if not already there
                if thread.id not in self.thread_models:
                    self.thread_models[thread.id] = set()
                self.thread_models[thread.id].update(models)
            
            ping_content += f"**Message:** {message}\n\n"
            
            if urgent:
                ping_content += "⏰ **Priority:** High - Please respond when available"
            else:
                ping_content += "💬 **Please respond when you have a moment**"
            
            await thread.send(ping_content)
            
            self.logger.info(f"Pinged models {models} in thread {thread.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to ping models in thread: {e}")
            return False
    
    async def update_thread_status(self, thread: discord.Thread, 
                                  status: str, 
                                  details: str = None) -> bool:
        """Update the status of a dev thread"""
        
        try:
            if thread.id not in self.dev_threads:
                return False
            
            thread_info = self.dev_threads[thread.id]
            
            status_icons = {
                'active': '🟡',
                'in_progress': '🔵', 
                'blocked': '🔴',
                'review': '🟠',
                'completed': '🟢',
                'archived': '⚫'
            }
            
            status_icon = status_icons.get(status, '❓')
            
            # Update thread info
            thread_info['status'] = status
            thread_info['last_updated'] = datetime.utcnow()
            
            # Post status update
            status_message = f"📊 **Status Update**\n\n"
            status_message += f"**Project:** {thread_info['project_name']}\n"
            status_message += f"**Status:** {status_icon} {status.title()}\n"
            
            if details:
                status_message += f"**Details:** {details}\n"
            
            status_message += f"**Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
            
            await thread.send(status_message)
            
            # Update thread name if status changed significantly
            if status in ['completed', 'blocked', 'archived']:
                new_name = f"{status_icon} {thread_info['project_name']}"
                await thread.edit(name=new_name)
            
            self.logger.info(f"Updated thread {thread.name} status to {status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update thread status: {e}")
            return False
    
    def get_thread_models(self, thread_id: int) -> Set[str]:
        """Get models assigned to a thread"""
        return self.thread_models.get(thread_id, set())
    
    def is_model_in_thread(self, thread_id: int, model_name: str) -> bool:
        """Check if a model is assigned to a thread"""
        return model_name in self.get_thread_models(thread_id)
    
    async def list_active_dev_threads(self, channel: discord.TextChannel) -> List[Dict]:
        """List all active dev threads in a channel"""
        
        active_threads = []
        
        try:
            # Get active threads from Discord
            async for thread in channel.guild.active_threads():
                if thread.parent_id == channel.id and thread.id in self.dev_threads:
                    thread_info = self.dev_threads[thread.id].copy()
                    thread_info['discord_thread'] = thread
                    active_threads.append(thread_info)
            
            return active_threads
            
        except Exception as e:
            self.logger.error(f"Failed to list active dev threads: {e}")
            return []
    
    async def create_project_summary(self, thread: discord.Thread) -> Optional[str]:
        """Generate a summary of project discussion in a thread"""
        
        if thread.id not in self.dev_threads:
            return None
        
        try:
            thread_info = self.dev_threads[thread.id]
            
            # Get recent messages
            messages = []
            async for msg in thread.history(limit=50):
                if not msg.author.bot or "Status Update" not in msg.content:
                    messages.append({
                        'author': msg.author.display_name,
                        'content': msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                        'timestamp': msg.created_at
                    })
            
            messages.reverse()  # Chronological order
            
            summary = f"# 📋 Project Summary: {thread_info['project_name']}\n\n"
            summary += f"**Description:** {thread_info['description']}\n"
            summary += f"**Status:** {thread_info.get('status', 'active')}\n"
            summary += f"**Created:** {thread_info['created_at'].strftime('%Y-%m-%d %H:%M UTC')}\n"
            
            if assigned_models := thread_info.get('assigned_models'):
                summary += f"**Assigned Models:** {', '.join(assigned_models)}\n"
            
            summary += f"**Messages:** {len(messages)} recent messages\n\n"
            
            if messages:
                summary += "## 💬 Recent Activity\n\n"
                for msg in messages[-10:]:  # Last 10 messages
                    summary += f"**{msg['author']}**: {msg['content']}\n\n"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to create project summary: {e}")
            return None


# Discord commands for dev thread management
class DevThreadCommands(commands.Cog):
    """Discord commands for managing dev threads"""
    
    def __init__(self, bot: commands.Bot, thread_manager: DevThreadManager):
        self.bot = bot
        self.thread_manager = thread_manager
    
    @discord.app_commands.command(name="create-dev-project", description="Create a new development project thread")
    async def create_dev_project(self, interaction: discord.Interaction, 
                                project_name: str,
                                description: str,
                                models: str = "",
                                ping_users: str = ""):
        """Create a new dev project thread"""
        
        if not self.thread_manager.is_devroom(interaction.channel):
            await interaction.response.send_message("❌ This command can only be used in dev channels!", ephemeral=True)
            return
        
        # Parse models
        assigned_models = [m.strip() for m in models.split(",") if m.strip()] if models else []
        
        # Parse users to ping (mentions or usernames)
        ping_user_objects = []
        if ping_users:
            for user_str in ping_users.split(","):
                user_str = user_str.strip()
                if user_str.startswith("<@") and user_str.endswith(">"):
                    # Discord mention format
                    user_id = int(user_str[2:-1].replace("!", ""))
                    user = interaction.guild.get_member(user_id)
                    if user:
                        ping_user_objects.append(user)
        
        thread = await self.thread_manager.create_dev_thread(
            channel=interaction.channel,
            project_name=project_name,
            description=description,
            assigned_models=assigned_models,
            ping_users=ping_user_objects
        )
        
        if thread:
            await interaction.response.send_message(
                f"✅ Created dev project thread: {thread.mention}\n"
                f"**Project:** {project_name}\n"
                f"**Models:** {', '.join(assigned_models) if assigned_models else 'None assigned'}"
            )
        else:
            await interaction.response.send_message("❌ Failed to create dev project thread", ephemeral=True)
    
    @discord.app_commands.command(name="ping-models", description="Ping specific models in current thread")
    async def ping_models(self, interaction: discord.Interaction,
                         models: str,
                         message: str,
                         urgent: bool = False):
        """Ping models in the current thread"""
        
        if not isinstance(interaction.channel, discord.Thread):
            await interaction.response.send_message("❌ This command can only be used in threads!", ephemeral=True)
            return
        
        if interaction.channel.id not in self.thread_manager.dev_threads:
            await interaction.response.send_message("❌ This is not a tracked dev thread!", ephemeral=True)
            return
        
        model_list = [m.strip() for m in models.split(",")]
        
        success = await self.thread_manager.ping_models_in_thread(
            thread=interaction.channel,
            models=model_list,
            message=message,
            urgent=urgent
        )
        
        if success:
            urgency = "🚨 **URGENT**" if urgent else "📢"
            await interaction.response.send_message(
                f"{urgency} Pinged models: {', '.join(model_list)}", 
                ephemeral=True
            )
        else:
            await interaction.response.send_message("❌ Failed to ping models", ephemeral=True)
    
    @discord.app_commands.command(name="update-status", description="Update the status of current dev thread")
    async def update_status(self, interaction: discord.Interaction,
                           status: str,
                           details: str = ""):
        """Update thread status"""
        
        if not isinstance(interaction.channel, discord.Thread):
            await interaction.response.send_message("❌ This command can only be used in threads!", ephemeral=True)
            return
        
        success = await self.thread_manager.update_thread_status(
            thread=interaction.channel,
            status=status.lower(),
            details=details
        )
        
        if success:
            await interaction.response.send_message(f"✅ Updated status to: **{status}**", ephemeral=True)
        else:
            await interaction.response.send_message("❌ Failed to update status", ephemeral=True)
    
    @discord.app_commands.command(name="dev-summary", description="Generate a summary of current dev thread")
    async def dev_summary(self, interaction: discord.Interaction):
        """Generate thread summary"""
        
        if not isinstance(interaction.channel, discord.Thread):
            await interaction.response.send_message("❌ This command can only be used in threads!", ephemeral=True)
            return
        
        await interaction.response.defer(ephemeral=True)
        
        summary = await self.thread_manager.create_project_summary(interaction.channel)
        
        if summary:
            # Send as file if too long
            if len(summary) > 2000:
                import io
                summary_file = discord.File(
                    io.StringIO(summary), 
                    filename=f"{interaction.channel.name}_summary.md"
                )
                await interaction.followup.send("📋 Project summary:", file=summary_file, ephemeral=True)
            else:
                await interaction.followup.send(f"```markdown\n{summary}\n```", ephemeral=True)
        else:
            await interaction.followup.send("❌ Failed to generate summary", ephemeral=True)


# Usage example
async def setup_dev_thread_manager(bot: commands.Bot) -> DevThreadManager:
    """Setup dev thread manager with bot"""
    
    thread_manager = DevThreadManager(bot)
    
    # Add commands
    await bot.add_cog(DevThreadCommands(bot, thread_manager))
    
    return thread_manager