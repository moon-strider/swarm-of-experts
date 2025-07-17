import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class JSONParseResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class JSONHandler:
    @staticmethod
    def extract_json_from_response(response: str) -> JSONParseResult:
        if not response or not response.strip():
            return JSONParseResult(success=False, error="Empty response")
        
        response = response.strip()
        
        try:
            if response.startswith('{') and response.endswith('}'):
                data = json.loads(response)
                return JSONParseResult(success=True, data=data)
        except json.JSONDecodeError as e:
            pass
        
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            r'\{(?:[^{}]|{[^{}]*})*\}',
            r'\{[\s\S]*?\}',
            r'\{[\s\S]*\}'
        ]
        
        for i, pattern in enumerate(json_patterns):
            matches = re.findall(pattern, response, re.DOTALL)
            for j, match in enumerate(matches):
                try:
                    data = json.loads(match)
                    if isinstance(data, dict):
                        return JSONParseResult(success=True, data=data)
                except json.JSONDecodeError as e:
                    continue
        
        lines = response.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            stripped = line.strip()
            if not in_json and stripped.startswith('{'):
                in_json = True
                json_lines = [line]
                brace_count = line.count('{') - line.count('}')
            elif in_json:
                json_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    try:
                        json_text = '\n'.join(json_lines)
                        data = json.loads(json_text)
                        if isinstance(data, dict):
                            return JSONParseResult(success=True, data=data)
                    except json.JSONDecodeError as e:
                        pass
                    in_json = False
                    json_lines = []
        
        return JSONParseResult(success=False, error=f"No valid JSON found in response: {repr(response)}")
    
    @staticmethod
    def validate_taskmaster_response(data: Dict[str, Any], expected_fields: List[str]) -> bool:
        if not isinstance(data, dict):
            return False
        
        return all(field in data and isinstance(data[field], str) and data[field].strip() for field in expected_fields)
    
    @staticmethod
    def get_prompt_fields(data: Dict[str, Any]) -> List[str]:
        if not isinstance(data, dict):
            return []
        
        return [key for key in data.keys() if key.startswith('prompt') and isinstance(data[key], str)]