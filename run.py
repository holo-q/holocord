#!/usr/bin/env python3
"""
Entry point for HoloCord multi-model Discord bot
"""
import sys
import asyncio

def main():
    """Main entry point"""
    try:
        from holocord.main import discord_bot
        
        # Start the bot
        import yaml
        with open("config.yaml", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        
        discord_bot.run(config["bot_token"])
        
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
        return 0
    except Exception as e:
        print(f"Error starting bot: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())