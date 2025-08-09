"""
Free Web Search Implementation - No API Keys Required
Provides multiple search engines without needing API keys
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional
from loguru import logger
from urllib.parse import quote
from bs4 import BeautifulSoup
import re

class FreeWebSearch:
    """
    Implements free web search using multiple providers without API keys.
    Automatically falls back between providers for reliability.
    """
    
    def __init__(self):
        self.timeout = aiohttp.ClientTimeout(total=10)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    async def search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Main search method that tries multiple providers
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Dict with search results
        """
        logger.info(f"🔍 Free web search for: {query}")
        
        # Try providers in order of reliability
        providers = [
            ('DuckDuckGo Library', self._search_duckduckgo_library),
            ('DuckDuckGo HTML', self._search_duckduckgo_html),
            ('DuckDuckGo Lite', self._search_duckduckgo_lite),
            ('SearXNG Public', self._search_searxng),
            ('Google Cache', self._search_google_cache),
        ]
        
        for provider_name, provider_func in providers:
            try:
                logger.info(f"Trying {provider_name}...")
                results = await provider_func(query, num_results)
                if results and len(results) > 0:
                    logger.info(f"✅ {provider_name} returned {len(results)} results")
                    return self._format_results(results, provider_name, query)
            except Exception as e:
                logger.warning(f"❌ {provider_name} failed: {e}")
                continue
        
        # If all fail, return error
        return {
            "error": "All search providers failed",
            "results": [],
            "ui_formatted": "Unable to perform web search at this time.",
            "voice_summary": "I couldn't search the web right now. Please try again."
        }
    
    async def _search_duckduckgo_library(self, query: str, num_results: int) -> List[Dict]:
        """
        Search using duckduckgo-search Python library (most reliable)
        """
        try:
            from duckduckgo_search import DDGS
            
            # Use sync version in async context
            ddgs = DDGS()
            results = []
            
            # Get text search results
            search_results = ddgs.text(query, max_results=num_results)
            
            for r in search_results:
                results.append({
                    'title': r.get('title', ''),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', 'No description available')
                })
                if len(results) >= num_results:
                    break
            
            return results
        except Exception as e:
            logger.debug(f"DuckDuckGo library error: {e}")
            raise
    
    async def _search_duckduckgo_html(self, query: str, num_results: int) -> List[Dict]:
        """
        Search using DuckDuckGo HTML scraping (most reliable)
        """
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            
            async with session.get(url, headers=self.headers) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")
                
                html = await resp.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                results = []
                # Find result divs
                for result_div in soup.find_all('div', class_='result'):
                    if len(results) >= num_results:
                        break
                    
                    # Extract title
                    title_elem = result_div.find('h2', class_='result__title')
                    if not title_elem:
                        continue
                    title = title_elem.get_text(strip=True)
                    
                    # Extract URL
                    link_elem = result_div.find('a', class_='result__url')
                    url = link_elem.get('href', '') if link_elem else ''
                    
                    # Extract snippet
                    snippet_elem = result_div.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    if title and (url or snippet):
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet or 'No description available'
                        })
                
                return results
    
    async def _search_duckduckgo_lite(self, query: str, num_results: int) -> List[Dict]:
        """
        Search using DuckDuckGo Lite version (mobile/simple interface)
        """
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            url = f"https://lite.duckduckgo.com/lite/?q={quote(query)}"
            
            async with session.get(url, headers=self.headers) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")
                
                html = await resp.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                results = []
                # Parse the simple table structure
                for row in soup.find_all('tr'):
                    if len(results) >= num_results:
                        break
                    
                    # Look for result links
                    link = row.find('a', class_='result-link')
                    if not link:
                        continue
                    
                    title = link.get_text(strip=True)
                    url = link.get('href', '')
                    
                    # Get snippet from next cell
                    snippet_cell = row.find('td', class_='result-snippet')
                    snippet = snippet_cell.get_text(strip=True) if snippet_cell else ''
                    
                    if title:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet or 'No description available'
                        })
                
                return results
    
    async def _search_searxng(self, query: str, num_results: int) -> List[Dict]:
        """
        Search using public SearXNG instances (privacy-focused metasearch)
        """
        # List of public SearXNG instances (these change, so we try multiple)
        instances = [
            "https://searx.be",
            "https://searx.tiekoetter.com", 
            "https://search.bus-hit.me",
            "https://searx.work"
        ]
        
        for instance in instances:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    url = f"{instance}/search"
                    params = {
                        'q': query,
                        'format': 'json',
                        'engines': 'duckduckgo,google,bing',
                        'pageno': 1
                    }
                    
                    async with session.get(url, params=params, headers=self.headers) as resp:
                        if resp.status != 200:
                            continue
                        
                        data = await resp.json()
                        results = []
                        
                        for result in data.get('results', [])[:num_results]:
                            results.append({
                                'title': result.get('title', ''),
                                'url': result.get('url', ''),
                                'snippet': result.get('content', 'No description available')
                            })
                        
                        if results:
                            return results
            except:
                continue
        
        raise Exception("All SearXNG instances failed")
    
    async def _search_google_cache(self, query: str, num_results: int) -> List[Dict]:
        """
        Search using Google's cache/text-only view (fallback option)
        """
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # Use Google's text-only search interface
            url = f"https://www.google.com/search"
            params = {
                'q': query,
                'num': num_results,
                'hl': 'en',
                'safe': 'off',
                'gbv': '1'  # Basic HTML version
            }
            
            async with session.get(url, params=params, headers=self.headers) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")
                
                html = await resp.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                results = []
                # Parse Google's basic HTML results
                for g in soup.find_all('div', class_='g'):
                    if len(results) >= num_results:
                        break
                    
                    # Extract title and URL
                    link = g.find('a')
                    if not link:
                        continue
                    
                    title = link.get_text(strip=True)
                    url = link.get('href', '')
                    
                    # Clean URL (remove Google redirect)
                    if '/url?q=' in url:
                        url = url.split('/url?q=')[1].split('&')[0]
                    
                    # Extract snippet
                    snippet_elem = g.find('span', class_='st') or g.find('div', class_='s')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    if title:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet or 'No description available'
                        })
                
                return results
    
    def _format_results(self, results: List[Dict], provider: str, query: str) -> Dict[str, Any]:
        """
        Format results for both UI and voice output
        """
        if not results:
            return {
                "results": [],
                "ui_formatted": "No results found.",
                "voice_summary": "I couldn't find any results for that search.",
                "provider": provider
            }
        
        # Format for UI with clickable links
        ui_parts = [f"**Search results for: {query}** (via {provider})\n"]
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            snippet = result.get('snippet', '')
            
            if url:
                ui_parts.append(f"{i}. [{title}]({url})")
            else:
                ui_parts.append(f"{i}. {title}")
            
            if snippet:
                ui_parts.append(f"   {snippet[:150]}...")
            ui_parts.append("")
        
        ui_formatted = "\n".join(ui_parts)
        
        # Format for voice (clean text)
        voice_parts = [f"Here are the search results for {query}:"]
        for i, result in enumerate(results[:3], 1):  # Only first 3 for voice
            title = result.get('title', 'Untitled')
            snippet = result.get('snippet', '')[:100]
            voice_parts.append(f"Result {i}: {title}. {snippet}")
        
        voice_summary = " ".join(voice_parts)
        
        return {
            "results": results,
            "ui_formatted": ui_formatted,
            "voice_summary": voice_summary,
            "result_count": len(results),
            "provider": provider
        }


# Global instance
free_web_search = FreeWebSearch()

async def search_web_free(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Main function to call for free web search
    
    Args:
        query: Search query
        num_results: Number of results to return (default: 5)
        
    Returns:
        Dict with results, ui_formatted, and voice_summary
    """
    return await free_web_search.search(query, num_results)