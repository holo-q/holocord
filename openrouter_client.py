#!/usr/bin/env python3
"""
OpenRouter API Client
Handles all OpenRouter API interactions
"""

import httpx
from typing import List, Dict, Optional
import logging

class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/holo-q/llmcord",
            "X-Title": "HOLO-Q Discord Multi-Model Bot"
        }
    
    async def query_chat_completion(self, model_id: str, messages: List[Dict], 
                                  system_prompt: Optional[str] = None,
                                  max_tokens: Optional[int] = None,
                                  timeout: float = 30.0) -> Optional[str]:
        """Query OpenRouter chat completion API"""
        try:
            request_data = {
                "model": model_id,
                "messages": messages.copy(),
                "stream": False,
            }
            
            # Only add max_tokens if specified
            if max_tokens is not None:
                request_data["max_tokens"] = max_tokens
            
            if system_prompt:
                request_data["messages"] = [
                    {"role": "system", "content": system_prompt}
                ] + request_data["messages"]
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=request_data
                )
                response.raise_for_status()
                
                data = response.json()
                return data['choices'][0]['message']['content']
                
        except httpx.TimeoutException:
            logging.error(f"Timeout querying {model_id}")
            return None
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error querying {model_id}: {e.response.status_code}")
            return None
        except Exception as e:
            logging.error(f"Error querying {model_id}: {e}")
            return None
    
    async def get_models(self) -> List[Dict]:
        """Get available models from OpenRouter"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/models", headers=self.headers)
                response.raise_for_status()
                
                data = response.json()
                return data.get('data', [])
                
        except Exception as e:
            logging.error(f"Error fetching models: {e}")
            return []