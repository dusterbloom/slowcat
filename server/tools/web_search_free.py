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
    
    def _detect_query_language(self, query: str) -> str:
        """Detect query language to improve search results"""
        query_lower = query.lower()
        
        # Simple language detection based on keywords
        italian_indicators = ['film', 'cinema', 'più', 'italia', 'streaming', 'scarica']
        spanish_indicators = ['películas', 'cine', 'más', 'españa', 'descargar']
        french_indicators = ['films', 'cinéma', 'plus', 'france', 'télécharger']
        
        if any(word in query_lower for word in italian_indicators):
            return 'it'
        elif any(word in query_lower for word in spanish_indicators):
            return 'es'  
        elif any(word in query_lower for word in french_indicators):
            return 'fr'
        else:
            return 'en'  # Default to English

    async def _search_duckduckgo_library(self, query: str, num_results: int) -> List[Dict]:
        """
        Search using duckduckgo-search Python library with language detection
        """
        try:
            from ddgs import DDGS
            
            # Use sync version in async context
            ddgs = DDGS()
            results = []
            
            # Detect query language
            query_lang = self._detect_query_language(query)
            
            # Set region based on detected language
            region_map = {
                'en': 'us-en',
                'it': 'it-it', 
                'es': 'es-es',
                'fr': 'fr-fr'
            }
            region = region_map.get(query_lang, 'us-en')
            
            # Try with language-specific settings first
            search_attempts = [
                # Force English language results with UK/US region
                lambda: ddgs.text(query, max_results=num_results * 3, region='uk-en', safesearch='off'),
                lambda: ddgs.text(query, max_results=num_results * 3, region='us-en', safesearch='off'), 
                lambda: ddgs.text(query, max_results=num_results * 2, region=region),
                lambda: ddgs.text(query, max_results=num_results * 2)  # Fallback without region
            ]
            
            search_results = None
            for attempt in search_attempts:
                try:
                    search_results = attempt()
                    if search_results:
                        break
                except:
                    continue
            
            if not search_results:
                raise Exception("All search attempts failed")
            
            for r in search_results:
                title = r.get('title', '')
                url = r.get('href', '')
                snippet = r.get('body', 'No description available')
                
                # Filter out obviously irrelevant results (Chinese, etc.)
                if self._is_relevant_result(title, snippet, url, query):
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet
                    })
                    if len(results) >= num_results:
                        break
            
            return results
        except Exception as e:
            logger.debug(f"DuckDuckGo library error: {e}")
            raise
    
    def _is_relevant_result(self, title: str, snippet: str, url: str, query: str = "") -> bool:
        """
        Filter out irrelevant search results
        """
        # Check for non-Latin characters (likely non-English content)
        combined_text = f"{title} {snippet}".lower()
        
        # Count non-ASCII characters
        non_ascii_count = sum(1 for char in combined_text if ord(char) > 127)
        total_chars = len(combined_text)
        
        if total_chars > 0:
            non_ascii_ratio = non_ascii_count / total_chars
            # Skip if more than 30% non-ASCII characters (likely not English)
            if non_ascii_ratio > 0.3:
                return False
        
        # Skip domains known for non-English content or irrelevant Asian news
        irrelevant_domains = [
            'scmp.com',  # South China Morning Post - shows up for London searches!
            'zhihu.com', 'baidu.com', 'weibo.com', 'qq.com', 'douban.com',
            '.cn', '.jp', '.kr', '.ru', '.br', '.hk'
        ]
        for domain in irrelevant_domains:
            if domain in url.lower():
                return False
        
        # Block content that's clearly about China/Asia when searching for other locations
        combined_text = f"{title} {snippet}".lower()
        asian_keywords = ['china', 'chinese', 'hong kong', 'beijing', 'xi jinping', 'asia']
        if any(keyword in combined_text for keyword in asian_keywords):
            # Only allow if the search query is also about Asia
            query_lower = query.lower()
            if not any(asian_word in query_lower for asian_word in ['china', 'chinese', 'hong kong', 'beijing', 'asia']):
                return False
        
        return True
    
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
        
        # Format for voice (clean text, optimized for TTS)
        voice_parts = [f"I found these results for {query}:"]
        for i, result in enumerate(results[:3], 1):  # Only first 3 for voice
            title = result.get('title', 'Untitled')
            snippet = result.get('snippet', '')
            
            # Clean title for voice (remove special characters that sound bad)
            title_clean = title.replace('…', '').replace('|', '').replace('–', '-').replace('—', '-')
            title_clean = title_clean[:60]  # Limit length for voice
            
            # Clean snippet for voice
            snippet_clean = snippet.replace('…', '').replace('\n', ' ').replace('  ', ' ')
            snippet_clean = snippet_clean[:80]  # Shorter for voice
            
            if snippet_clean:
                voice_parts.append(f"{title_clean}: {snippet_clean}.")
            else:
                voice_parts.append(f"{title_clean}.")
        
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