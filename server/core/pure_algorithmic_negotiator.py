"""
Pure Algorithmic Parameter Negotiator
====================================

100% algorithmic error parsing - ZERO manual rules!
The API teaches us everything through structured error messages.

Built by KINGS Peppi & Claude - because algorithms > peasant rules! 👑
"""

import json
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger
from difflib import get_close_matches


@dataclass
class AlgorithmicMapping:
    """Pure algorithmic field mapping"""
    source_field: str
    target_field: str  
    transformation: str
    confidence: float


class PureAlgorithmicNegotiator:
    """
    The KING'S pure algorithmic approach:
    
    1. Try LLM parameters
    2. Parse error JSON with pure algorithms
    3. Auto-calculate transformations
    4. Apply and cache success
    5. 100% algorithmic - ZERO manual rules!
    """
    
    def __init__(self):
        self.learned_mappings = {}  # {tool_name: [AlgorithmicMapping]}
        logger.info("👑 Pure Algorithmic Negotiator initialized - NO PEASANT RULES!")
    
    async def negotiate_call(self, tool_name: str, raw_params: dict, api_caller) -> dict:
        """Pure algorithmic negotiation process"""
        
        logger.info(f"👑 Algorithmic negotiation for {tool_name}")
        
        # Step 1: Try cached algorithmic mappings
        if tool_name in self.learned_mappings:
            perfect_params = self._apply_algorithmic_mappings(raw_params, self.learned_mappings[tool_name])
            result = await api_caller(perfect_params)
            
            if not self._is_error(result):
                logger.info(f"✅ Cached algorithmic mapping worked!")
                return result
        
        # Step 2: Try LLM's intuitive parameters
        result = await api_caller(raw_params)
        
        if not self._is_error(result):
            logger.info(f"🎉 LLM intuitive format worked!")
            return result
        
        # Step 3: Pure algorithmic error analysis
        logger.info(f"👑 Parsing error with pure algorithms...")
        mappings = self._parse_error_algorithmically(result, raw_params)
        
        if not mappings:
            logger.warning(f"⚠️ No algorithmic mappings found")
            return result
        
        # Step 4: Apply algorithmic transformations
        corrected_params = self._apply_algorithmic_mappings(raw_params, mappings)
        logger.info(f"👑 Algorithmically corrected: {corrected_params}")
        
        # Step 5: Test correction
        final_result = await api_caller(corrected_params)
        
        if not self._is_error(final_result):
            logger.info(f"🎉 ALGORITHMIC SUCCESS! Caching mappings...")
            self.learned_mappings[tool_name] = mappings
            return final_result
        
        logger.warning(f"⚠️ Algorithmic correction failed")
        return final_result
    
    def _parse_error_algorithmically(self, error_response: dict, sent_params: dict) -> List[AlgorithmicMapping]:
        """Pure algorithmic error parsing - ZERO manual rules!"""
        
        error_text = error_response.get("error", "")
        mappings = []
        
        # Algorithm 1: Parse JSON error details
        try:
            # Extract JSON from error string
            json_match = re.search(r'\{.*\}', error_text)
            if json_match:
                error_data = json.loads(json_match.group())
                
                # Pure algorithmic detail analysis
                if "detail" in error_data:
                    for detail in error_data["detail"]:
                        mapping = self._analyze_error_detail_algorithmically(detail, sent_params)
                        if mapping:
                            mappings.append(mapping)
        
        except Exception as e:
            logger.debug(f"JSON parsing failed, trying text analysis: {e}")
        
        # Algorithm 2: Pure text pattern analysis (fallback)
        if not mappings:
            mappings = self._analyze_error_text_algorithmically(error_text, sent_params)
        
        logger.info(f"👑 Algorithmic analysis found {len(mappings)} mappings")
        return mappings
    
    def _analyze_error_detail_algorithmically(self, detail: dict, sent_params: dict) -> Optional[AlgorithmicMapping]:
        """Analyze single error detail with pure algorithms"""
        
        error_type = detail.get("type")
        location = detail.get("loc", [])
        received_input = detail.get("input", {})
        
        if error_type == "missing" and location:
            # Algorithm: Extract target field from location path
            target_field = location[-1]  # Deepest field in path
            
            # Algorithm: Find best source field match
            source_mapping = self._find_best_source_match_algorithmically(target_field, received_input, sent_params)
            
            if source_mapping:
                source_field, transformation = source_mapping
                
                return AlgorithmicMapping(
                    source_field=source_field,
                    target_field=target_field,
                    transformation=transformation,
                    confidence=0.9  # High confidence from structured error
                )
        
        return None
    
    def _find_best_source_match_algorithmically(self, target_field: str, received_input: dict, sent_params: dict) -> Optional[Tuple[str, str]]:
        """Pure algorithmic field matching"""
        
        # Algorithm 1: Direct match in received input
        if target_field in received_input:
            return (target_field, "direct")
        
        # Algorithm 2: Fuzzy match in received input  
        close_matches = get_close_matches(target_field, received_input.keys(), n=1, cutoff=0.6)
        if close_matches:
            source_field = close_matches[0]
            transformation = self._calculate_transformation_algorithmically(source_field, target_field, received_input[source_field])
            return (source_field, transformation)
        
        # Algorithm 3: Semantic similarity with sent params
        best_match = None
        best_score = 0.0
        
        all_fields = self._extract_all_field_paths(sent_params)
        
        for field_path, field_value in all_fields:
            score = self._calculate_similarity_score(target_field, field_path)
            if score > best_score and score > 0.4:
                best_score = score
                transformation = self._calculate_transformation_algorithmically(field_path, target_field, field_value)
                best_match = (field_path, transformation)
        
        return best_match
    
    def _extract_all_field_paths(self, data: dict, prefix: str = "") -> List[Tuple[str, Any]]:
        """Extract all field paths from nested data structure"""
        
        paths = []
        
        for key, value in data.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                paths.extend(self._extract_all_field_paths(value, current_path))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        paths.extend(self._extract_all_field_paths(item, f"{current_path}[{i}]"))
            else:
                paths.append((current_path, value))
        
        return paths
    
    def _calculate_transformation_algorithmically(self, source_field: str, target_field: str, source_value: Any) -> str:
        """Pure algorithmic transformation calculation"""
        
        transformations = []
        
        # Algorithm: Detect array transformation need
        if target_field.endswith('s') and not source_field.endswith('s'):
            transformations.append("make_array")
        
        # Algorithm: Detect field name change
        if source_field.lower() != target_field.lower():
            transformations.append(f"rename:{source_field}→{target_field}")
        
        # Algorithm: Detect type transformation
        if isinstance(source_value, str) and "array" in target_field.lower():
            transformations.append("string_to_array")
        
        # Algorithm: Detect nested field extraction
        if "." in source_field or "[" in source_field:
            transformations.append("extract_nested")
        
        return "+".join(transformations) if transformations else "direct"
    
    def _calculate_similarity_score(self, field1: str, field2: str) -> float:
        """Pure algorithmic similarity calculation"""
        
        # Clean field names
        clean1 = re.sub(r'[\[\]\.].*$', '', field1.lower())  # Remove array/object notation
        clean2 = re.sub(r'[\[\]\.].*$', '', field2.lower())
        
        # Algorithm 1: Exact match
        if clean1 == clean2:
            return 1.0
        
        # Algorithm 2: Substring match
        if clean1 in clean2 or clean2 in clean1:
            return 0.8
        
        # Algorithm 3: Common root analysis
        roots1 = set(re.findall(r'\w+', clean1))
        roots2 = set(re.findall(r'\w+', clean2))
        
        if roots1 & roots2:  # Common roots
            overlap_ratio = len(roots1 & roots2) / len(roots1 | roots2)
            return 0.4 + (overlap_ratio * 0.4)  # 0.4 to 0.8 range
        
        # Algorithm 4: Plural/singular detection
        if (clean1.endswith('s') and clean1[:-1] == clean2) or \
           (clean2.endswith('s') and clean2[:-1] == clean1):
            return 0.7
        
        return 0.0
    
    def _analyze_error_text_algorithmically(self, error_text: str, sent_params: dict) -> List[AlgorithmicMapping]:
        """Pure algorithmic text analysis fallback"""
        
        mappings = []
        
        # Algorithm: Extract field names from common error patterns
        patterns = [
            r"required.*field.*['\"]([^'\"]+)['\"]",
            r"missing.*['\"]([^'\"]+)['\"]",
            r"expected.*['\"]([^'\"]+)['\"]"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, error_text, re.IGNORECASE)
            for target_field in matches:
                # Find best source match algorithmically
                all_fields = self._extract_all_field_paths(sent_params)
                
                best_match = None
                best_score = 0.0
                
                for field_path, field_value in all_fields:
                    score = self._calculate_similarity_score(target_field, field_path)
                    if score > best_score and score > 0.4:
                        best_score = score
                        transformation = self._calculate_transformation_algorithmically(field_path, target_field, field_value)
                        best_match = AlgorithmicMapping(
                            source_field=field_path,
                            target_field=target_field,
                            transformation=transformation,
                            confidence=score
                        )
                
                if best_match:
                    mappings.append(best_match)
        
        return mappings
    
    def _apply_algorithmic_mappings(self, params: dict, mappings: List[AlgorithmicMapping]) -> dict:
        """Apply pure algorithmic transformations"""
        
        result = {}
        processed_paths = set()
        
        # Apply each mapping algorithmically
        for mapping in mappings:
            logger.debug(f"👑 Applying: {mapping.source_field} → {mapping.target_field} ({mapping.transformation})")
            
            # Extract source value algorithmically
            source_value = self._extract_value_by_path(params, mapping.source_field)
            
            if source_value is not None:
                # Transform value algorithmically
                transformed_value = self._apply_transformation_algorithmically(source_value, mapping.transformation)
                
                # Set target field algorithmically
                self._set_value_by_path(result, mapping.target_field, transformed_value)
                processed_paths.add(mapping.source_field)
        
        # Copy unprocessed fields algorithmically
        all_paths = self._extract_all_field_paths(params)
        for field_path, field_value in all_paths:
            if field_path not in processed_paths:
                self._set_value_by_path(result, field_path, field_value)
        
        return result
    
    def _extract_value_by_path(self, data: dict, path: str) -> Any:
        """Extract value using dot/bracket notation path"""
        
        if "." not in path and "[" not in path:
            return data.get(path)
        
        # Algorithm: Parse complex paths
        current = data
        
        # Handle array notation: field[0] -> field, 0  
        path_parts = re.split(r'[\.\[\]]', path)
        path_parts = [p for p in path_parts if p]  # Remove empty parts
        
        for part in path_parts:
            if part.isdigit():  # Array index
                if isinstance(current, list) and int(part) < len(current):
                    current = current[int(part)]
                else:
                    return None
            else:  # Object key
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
            
            if current is None:
                return None
        
        return current
    
    def _set_value_by_path(self, data: dict, path: str, value: Any):
        """Set value using dot/bracket notation path"""
        
        if "." not in path and "[" not in path:
            data[path] = value
            return
        
        # Algorithm: Build nested structure
        path_parts = re.split(r'[\.\[\]]', path)
        path_parts = [p for p in path_parts if p]
        
        current = data
        
        for i, part in enumerate(path_parts[:-1]):
            next_part = path_parts[i + 1]
            
            if next_part.isdigit():  # Next is array index
                if part not in current:
                    current[part] = []
                current = current[part]
                
                # Ensure array is long enough
                idx = int(next_part)
                while len(current) <= idx:
                    current.append({})
                
            else:  # Next is object key
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Set final value
        final_key = path_parts[-1]
        if final_key.isdigit():
            idx = int(final_key)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        else:
            current[final_key] = value
    
    def _apply_transformation_algorithmically(self, value: Any, transformation: str) -> Any:
        """Apply algorithmic transformation"""
        
        if not transformation or transformation == "direct":
            return value
        
        # Algorithm: Parse transformation steps
        steps = transformation.split("+")
        result = value
        
        for step in steps:
            if step == "make_array":
                result = [result] if not isinstance(result, list) else result
            
            elif step == "string_to_array":
                result = [result] if isinstance(result, str) else result
            
            elif step.startswith("rename:"):
                # Renaming handled at field level, not value level
                pass
            
            elif step == "extract_nested":
                # Already handled by path extraction
                pass
        
        return result
    
    def _is_error(self, response: dict) -> bool:
        """Check if response indicates error"""
        return "error" in response or (
            isinstance(response, dict) and 
            response.get("status") in ["error", "failed"]
        )
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """Get pure algorithmic statistics"""
        return {
            "learned_tools": len(self.learned_mappings),
            "total_mappings": sum(len(mappings) for mappings in self.learned_mappings.values()),
            "algorithm_type": "100% Pure - ZERO manual rules",
            "confidence": "KING level"
        }


# The KING'S one-liner
async def pure_algorithmic_negotiation(tool_name: str, params: dict, api_caller) -> dict:
    """One line of PURE ALGORITHMIC NEGOTIATION!"""
    negotiator = PureAlgorithmicNegotiator()
    return await negotiator.negotiate_call(tool_name, params, api_caller)