#!/usr/bin/env python3
"""
OpenRouter Models Fetcher
Fetches available models from OpenRouter API and processes them for Discord virtual users
"""

import httpx
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class ModelInfo:
    id: str
    name: str
    description: str
    pricing: Dict[str, float]
    context_length: int
    avatar_url: Optional[str] = None
    discord_name: str = ""
    
    def __post_init__(self):
        """Generate Discord-friendly username from model ID"""
        # Convert model ID to Discord username
        # e.g. "anthropic/claude-3-5-sonnet" → "Claude-Sonnet"
        base_name = self.id.split('/')[-1]  # Get part after /
        
        # Clean up common patterns
        base_name = re.sub(r'-\d+k$', '', base_name)  # Remove context length suffix
        base_name = re.sub(r'-\d{4}-\d{2}$', '', base_name)  # Remove date suffix
        base_name = re.sub(r'-instruct$', '', base_name)  # Remove instruct suffix
        base_name = re.sub(r'-chat$', '', base_name)  # Remove chat suffix
        
        # Convert to title case and clean up
        parts = base_name.split('-')
        cleaned_parts = []
        
        for part in parts:
            # Skip version numbers and common suffixes
            if re.match(r'^\d+$', part) or part in ['v', 'preview', 'beta', 'alpha']:
                continue
            
            # Handle special cases
            if part.lower() == 'gpt':
                cleaned_parts.append('GPT')
            elif part.lower() in ['claude', 'gemini', 'llama', 'qwen', 'mistral']:
                cleaned_parts.append(part.capitalize())
            else:
                cleaned_parts.append(part.capitalize())
        
        self.discord_name = '-'.join(cleaned_parts[:2])  # Limit to 2 parts for readability
        
        # Fallback if name is too short
        if len(self.discord_name) < 3:
            self.discord_name = self.name.split()[0] if self.name else self.id.split('/')[-1]

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        
    async def fetch_models(self) -> List[ModelInfo]:
        """Fetch all available models from OpenRouter"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/holo-q/llmcord",
            "X-Title": "HOLO-Q Discord Multi-Model Bot"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/models", headers=headers)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get('data', []):
                try:
                    model = ModelInfo(
                        id=model_data['id'],
                        name=model_data.get('name', ''),
                        description=model_data.get('description', ''),
                        pricing=model_data.get('pricing', {}),
                        context_length=model_data.get('context_length', 0)
                    )
                    models.append(model)
                except Exception as e:
                    print(f"Error processing model {model_data.get('id', 'unknown')}: {e}")
                    continue
            
            return models
    
    def filter_models(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Filter models for Discord use - remove NSFW, very expensive, or problematic ones"""
        filtered = []
        
        for model in models:
            # Skip NSFW models
            if any(term in model.id.lower() or term in model.name.lower() or term in model.description.lower() 
                   for term in ['nsfw', 'uncensored', 'erotica', 'adult']):
                continue
                
            # Skip very expensive models (>$50 per 1M tokens)
            prompt_cost = model.pricing.get('prompt', 0)
            try:
                if float(prompt_cost) > 0.05:  # $50 per 1M tokens
                    continue
            except (ValueError, TypeError):
                # Skip models with invalid pricing data
                continue
                
            # Skip models with very small context (likely old/bad)
            if model.context_length < 2000:
                continue
                
            # Skip obvious duplicates or test models
            if any(term in model.id.lower() for term in ['test', 'demo', 'example', 'deprecated']):
                continue
                
            filtered.append(model)
        
        # Sort by popularity/quality (rough heuristic)
        def model_score(m: ModelInfo) -> float:
            score = 0
            
            # Prefer well-known providers
            provider_bonus = {
                'anthropic': 100,
                'openai': 90, 
                'google': 80,
                'meta-llama': 70,
                'mistralai': 60,
                'qwen': 50,
            }
            
            provider = m.id.split('/')[0]
            score += provider_bonus.get(provider, 0)
            
            # Prefer higher context length (up to a point)
            score += min(m.context_length / 1000, 50)
            
            # Prefer newer models (rough heuristic)
            if '3.5' in m.id or '4' in m.id or '2024' in m.id:
                score += 20
            
            return score
        
        filtered.sort(key=model_score, reverse=True)
        
        # Take top models but ensure diversity
        final_models = []
        seen_providers = set()
        
        for model in filtered:
            provider = model.id.split('/')[0]
            
            # Always include top models from major providers
            if len(final_models) < 20 and (len(final_models) < 8 or provider not in seen_providers):
                final_models.append(model)
                seen_providers.add(provider)
        
        return final_models

async def main():
    """Test the OpenRouter model fetcher"""
    # Get API key from config
    import yaml
    
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            api_key = config['providers']['openrouter']['api_key']
            
        if not api_key or api_key.startswith('sk-or-v1-'):
            print("⚠️  Using default API key from config - replace with your own!")
        
        client = OpenRouterClient(api_key)
        print("🔄 Fetching models from OpenRouter...")
        
        models = await client.fetch_models()
        print(f"📥 Fetched {len(models)} total models")
        
        filtered_models = client.filter_models(models)
        print(f"✅ Filtered to {len(filtered_models)} suitable models")
        
        print("\n🤖 Top Discord Virtual Users:")
        for i, model in enumerate(filtered_models[:15], 1):
            try:
                price = float(model.pricing.get('prompt', 0)) * 1000000  # Convert to per 1M tokens
                price_str = f"${price:.2f}/1M tokens"
            except (ValueError, TypeError):
                price_str = "Price unknown"
            
            print(f"{i:2d}. {model.discord_name:<20} ({model.id})")
            print(f"    💰 {price_str} | 🧠 {model.context_length:,} context")
            if model.description:
                desc = model.description[:80] + "..." if len(model.description) > 80 else model.description
                print(f"    📝 {desc}")
            print()
        
        # Save to JSON for Discord bot to use
        model_data = {
            'models': [
                {
                    'id': m.id,
                    'discord_name': m.discord_name,
                    'name': m.name,
                    'description': m.description,
                    'pricing': m.pricing,
                    'context_length': m.context_length
                }
                for m in filtered_models
            ],
            'total_count': len(filtered_models),
            'updated_at': httpx.get('http://worldtimeapi.org/api/timezone/UTC').json()['datetime']
        }
        
        with open('openrouter_models.json', 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"💾 Saved {len(filtered_models)} models to openrouter_models.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure config.yaml has a valid OpenRouter API key!")

if __name__ == "__main__":
    asyncio.run(main())