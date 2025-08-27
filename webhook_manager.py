#!/usr/bin/env python3
"""
Discord Webhook Manager
Handles webhook creation, deletion and message sending
"""

import httpx
import discord
from typing import Optional, List
import logging

class WebhookManager:
    @staticmethod
    async def create_webhook(channel: discord.TextChannel, name: str, 
                           avatar_url: Optional[str] = None) -> discord.Webhook:
        """Create a new webhook in the channel"""
        try:
            webhook = await channel.create_webhook(
                name=name,
                reason=f"Virtual user webhook for {name}"
            )
            return webhook
        except Exception as e:
            logging.error(f"Error creating webhook: {e}")
            raise
    
    @staticmethod
    async def delete_webhook(webhook_id: str) -> bool:
        """Delete a webhook by ID"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"https://discord.com/api/webhooks/{webhook_id}"
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logging.error(f"Error deleting webhook {webhook_id}: {e}")
            return False
    
    @staticmethod
    async def send_webhook_message(webhook_url: str, content: str, 
                                 username: str, avatar_url: Optional[str] = None,
                                 embeds: Optional[List[discord.Embed]] = None) -> bool:
        """Send a message via webhook"""
        try:
            webhook_data = {
                'content': content,
                'username': username,
            }
            
            if avatar_url:
                webhook_data['avatar_url'] = avatar_url
            
            if embeds:
                webhook_data['embeds'] = [embed.to_dict() for embed in embeds]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=webhook_data)
                response.raise_for_status()
            
            return True
            
        except Exception as e:
            logging.error(f"Error sending webhook message: {e}")
            return False