#!/usr/bin/env python3
"""
Admin Permission Checker
Centralized admin permission management
"""

import yaml
from typing import List, Optional
import discord

class AdminChecker:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._admin_ids: Optional[List[int]] = None
    
    def _load_admin_ids(self) -> List[int]:
        """Load admin IDs from config"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('permissions', {}).get('users', {}).get('admin_ids', [])
        except Exception:
            return []
    
    def get_admin_ids(self) -> List[int]:
        """Get admin IDs (cached)"""
        if self._admin_ids is None:
            self._admin_ids = self._load_admin_ids()
        return self._admin_ids
    
    def refresh_admin_ids(self):
        """Force refresh admin IDs from config"""
        self._admin_ids = None
        return self.get_admin_ids()
    
    def is_admin(self, user_id: int) -> bool:
        """Check if user is admin"""
        return user_id in self.get_admin_ids()
    
    def is_interaction_admin(self, interaction: discord.Interaction) -> bool:
        """Check if Discord interaction user is admin"""
        return self.is_admin(interaction.user.id)
    
    async def require_admin(self, interaction: discord.Interaction, 
                          action_name: str = "perform this action") -> bool:
        """Check admin permission and respond with error if not admin"""
        if not self.is_interaction_admin(interaction):
            await interaction.followup.send(
                f"❌ You don't have permission to {action_name}. Admin only.", 
                ephemeral=True
            )
            return False
        return True

# Global instance
admin_checker = AdminChecker()