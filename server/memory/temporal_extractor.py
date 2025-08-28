"""
Temporal extraction for spaCy pipeline using date-spacy and custom patterns
"""

import spacy
from typing import List, Dict, Any, Optional
import re
from datetime import datetime, timedelta
from loguru import logger

# Try to import date-spacy
try:
    import date_spacy
    DATE_SPACY_AVAILABLE = True
except ImportError:
    DATE_SPACY_AVAILABLE = False
    logger.warning("date-spacy not available. Install with: pip install date-spacy")

# Try to import dateparser
try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False
    logger.warning("dateparser not available. Install with: pip install dateparser")


class TemporalExtractor:
    """Extract temporal expressions from text"""
    
    def __init__(self):
        self.nlp = None
        self._load_spacy_model()
        
    def _load_spacy_model(self):
        """Load spaCy model with temporal extraction components"""
        try:
            # Load base model
            self.nlp = spacy.load("en_core_web_trf")
            
            # Add date-spacy component if available
            if DATE_SPACY_AVAILABLE:
                try:
                    self.nlp.add_pipe('find_dates', before='ner')
                    logger.info("✅ Added date-spacy component to pipeline")
                except Exception as e:
                    logger.warning(f"Failed to add date-spacy component: {e}")
            
            # Add custom temporal patterns
            self._add_custom_patterns()
            logger.info("✅ Temporal extractor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize temporal extractor: {e}")
            self.nlp = None
    
    def _add_custom_patterns(self):
        """Add custom temporal patterns using spaCy's Matcher"""
        from spacy.matcher import Matcher
        
        self.matcher = Matcher(self.nlp.vocab)
        
        # Meeting patterns
        meeting_patterns = [
            # "meeting with Sarah on Thursday at 3 PM"
            [{"LOWER": "meeting"}, {"LOWER": "with"}, {"ENT_TYPE": "PERSON"}, 
             {"LOWER": "on"}, {"ENT_TYPE": {"IN": ["DATE", "TIME"]}}, 
             {"LOWER": "at"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["pm", "am"]}}],
            
            # "appointment with Dr. Smith tomorrow"
            [{"LOWER": "appointment"}, {"LOWER": "with"}, {"ENT_TYPE": "PERSON"}, 
             {"LOWER": {"IN": ["tomorrow", "today", "yesterday"]}}],
            
            # "call John on Friday"
            [{"LOWER": {"IN": ["call", "phone", "contact"]}}, {"ENT_TYPE": "PERSON"}, 
             {"LOWER": "on"}, {"ENT_TYPE": "DATE"}],
            
            # "Thursday at 3 PM"
            [{"ENT_TYPE": "DATE"}, {"LOWER": "at"}, {"LIKE_NUM": True}, 
             {"LOWER": {"IN": ["pm", "am"]}}]
        ]
        
        self.matcher.add("MEETING_TIME", meeting_patterns)
        
        # Birthday patterns
        birthday_patterns = [
            # "my birthday is on December 15th"
            [{"LOWER": "birthday"}, {"LOWER": "is"}, {"LOWER": "on"}, 
             {"ENT_TYPE": "DATE"}],
            
            # "birthday on Dec 15"
            [{"LOWER": "birthday"}, {"LOWER": "on"}, {"ENT_TYPE": "DATE"}]
        ]
        
        self.matcher.add("BIRTHDAY", birthday_patterns)
    
    def extract_temporal_expressions(self, text: str) -> List[Dict[str, Any]]:
        """Extract temporal expressions from text"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            temporal_expressions = []
            
            # Method 1: Use date-spacy if available
            if DATE_SPACY_AVAILABLE:
                temporal_expressions.extend(self._extract_with_date_spacy(doc))
            
            # Method 2: Use custom patterns
            temporal_expressions.extend(self._extract_with_patterns(doc))
            
            # Method 3: Use dateparser on the full text
            if DATEPARSER_AVAILABLE:
                temporal_expressions.extend(self._extract_with_dateparser(text))
            
            # Deduplicate by text span
            seen_spans = set()
            unique_expressions = []
            for expr in temporal_expressions:
                span_key = (expr['start'], expr['end'])
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    unique_expressions.append(expr)
            
            return unique_expressions
            
        except Exception as e:
            logger.error(f"Temporal extraction failed: {e}")
            return []
    
    def _extract_with_date_spacy(self, doc) -> List[Dict[str, Any]]:
        """Extract using date-spacy extension"""
        expressions = []
        
        try:
            for token in doc:
                if token._.date:
                    expressions.append({
                        'text': token.text,
                        'start': token.idx,
                        'end': token.idx + len(token.text),
                        'parsed_date': token._.date,
                        'type': 'date',
                        'source': 'date_spacy'
                    })
        except Exception as e:
            logger.debug(f"date-spacy extraction failed: {e}")
        
        return expressions
    
    def _extract_with_patterns(self, doc) -> List[Dict[str, Any]]:
        """Extract using custom patterns"""
        expressions = []
        
        try:
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                label = self.nlp.vocab.strings[match_id]
                
                # Parse the temporal content
                parsed_date = self._parse_temporal_span(span.text)
                
                expressions.append({
                    'text': span.text,
                    'start': span.start_char,
                    'end': span.end_char,
                    'parsed_date': parsed_date,
                    'type': label.lower(),
                    'source': 'pattern_matching'
                })
        except Exception as e:
            logger.debug(f"Pattern matching failed: {e}")
        
        return expressions
    
    def _extract_with_dateparser(self, text: str) -> List[Dict[str, Any]]:
        """Extract using dateparser on common temporal phrases"""
        expressions = []
        
        try:
            # Common temporal phrases to check
            temporal_phrases = [
                r'\b(?:tomorrow|today|yesterday)\b',
                r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b',
                r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b',
                r'\b\d{1,2}/\d{1,2}/\d{4}\b'
            ]
            
            for pattern in temporal_phrases:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    phrase = match.group()
                    parsed_date = dateparser.parse(phrase)
                    
                    if parsed_date:
                        expressions.append({
                            'text': phrase,
                            'start': match.start(),
                            'end': match.end(),
                            'parsed_date': parsed_date,
                            'type': 'temporal_phrase',
                            'source': 'dateparser'
                        })
        except Exception as e:
            logger.debug(f"Dateparser extraction failed: {e}")
        
        return expressions
    
    def _parse_temporal_span(self, text: str) -> Optional[datetime]:
        """Parse a temporal span into a datetime object"""
        if DATEPARSER_AVAILABLE:
            try:
                return dateparser.parse(text)
            except Exception:
                pass
        return None
    
    def extract_events_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract events with temporal information from text"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            temporal_expressions = self.extract_temporal_expressions(text)
            
            events = []
            
            # Look for event-related patterns
            event_patterns = [
                # "meeting with Sarah on Thursday at 3 PM"
                r'(?i)\b(meeting|appointment|call|lunch|dinner|party)\s+with\s+([A-Za-z\s]+)(?:\s+on\s+([^.]+?))?(?:\s+at\s+([^.]+?))?',
                
                # "birthday on December 15th"
                r'(?i)\b(birthday)\s+(?:is\s+)?on\s+([^.]+)',
                
                # "deadline for project tomorrow"
                r'(?i)\b(deadline|due)\s+(?:for\s+)?([^.]+?)\s+(tomorrow|today|on\s+[^.]+)',
            ]
            
            for pattern in event_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    event_type = match.group(1).lower()
                    
                    # Extract components based on pattern
                    if event_type in ['meeting', 'appointment', 'call']:
                        title = f"{event_type.title()} with {match.group(2).strip()}"
                        time_info = match.group(3) or match.group(4)
                    elif event_type == 'birthday':
                        title = f"Birthday"
                        time_info = match.group(2)
                    elif event_type in ['deadline', 'due']:
                        title = f"{event_type.title()}: {match.group(2).strip()}"
                        time_info = match.group(3)
                    else:
                        continue
                    
                    # Parse temporal information
                    parsed_time = None
                    if time_info and DATEPARSER_AVAILABLE:
                        try:
                            parsed_time = dateparser.parse(time_info.strip())
                        except Exception:
                            pass
                    
                    events.append({
                        'title': title,
                        'description': match.group(0),
                        'start_time': parsed_time.isoformat() if parsed_time else None,
                        'event_type': event_type,
                        'raw_time_text': time_info,
                        'source_text': text,
                        'confidence': 0.8
                    })
            
            return events
            
        except Exception as e:
            logger.error(f"Event extraction failed: {e}")
            return []


# Global instance
_temporal_extractor = None

def get_temporal_extractor() -> TemporalExtractor:
    """Get singleton temporal extractor instance"""
    global _temporal_extractor
    if _temporal_extractor is None:
        _temporal_extractor = TemporalExtractor()
    return _temporal_extractor


def extract_temporal_expressions(text: str) -> List[Dict[str, Any]]:
    """Convenience function for temporal extraction"""
    extractor = get_temporal_extractor()
    return extractor.extract_temporal_expressions(text)


def extract_events_from_text(text: str) -> List[Dict[str, Any]]:
    """Convenience function for event extraction"""
    extractor = get_temporal_extractor()
    return extractor.extract_events_from_text(text)


if __name__ == "__main__":
    # Test temporal extraction
    test_texts = [
        "I have a meeting with Sarah on Thursday at 3 PM",
        "My birthday is on December 15th and I'm planning a party",
        "Call John tomorrow at 10 AM",
        "The deadline for this project is next Friday",
        "Lunch with mom on Sunday"
    ]
    
    extractor = TemporalExtractor()
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        
        temporal_exprs = extractor.extract_temporal_expressions(text)
        print(f"  Temporal expressions: {len(temporal_exprs)}")
        for expr in temporal_exprs:
            print(f"    • {expr['text']} → {expr.get('parsed_date')}")
        
        events = extractor.extract_events_from_text(text)
        print(f"  Events: {len(events)}")
        for event in events:
            print(f"    • {event['title']} @ {event.get('start_time', 'TBD')}")