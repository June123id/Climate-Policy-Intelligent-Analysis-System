"""
AnalysisAgent: Extracts structured entities from policy text using Kimi LLM API
"""
import json
import time
import sys
import os
from typing import Optional, Dict, Any
from openai import OpenAI, Timeout, RateLimitError, APIError

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from models.schemas import PolicyEntities


# Prompt template from get_key_benchmark_scores1.py
PROMPT_TEMPLATE = """You are an expert policy analyst specializing in climate mitigation policies. Your task is to extract structured information from a given policy text according to a predefined ontology. Output ONLY a valid JSON object with the exact schema below. Do not include any other text, explanation, or markdown.

**Output Schema**:
{
  "instrument_type": ["Regulatory", "Economic", "Informational", "Infrastructure"],
  "target_sector": ["Transport", "Energy", "Industry", "Buildings", "Cross-cutting"],
  "geographic_scope": ["National", "Regional", "Municipal", "Site-specific"],
  "temporal_scope": {
    "start_year": integer or null,
    "target_year": integer or null,
    "duration": string or null
  },
  "quantitative_targets": [
    {
      "metric": string,
      "value": number,
      "unit": string,
      "target_year": integer or null,
      "coverage": string or null
    }
  ],
  "policy_actors": {
    "issuer": string or null,
    "implementer": [string],
    "affected_parties": [string]
  },
  "implementation_mechanisms": ["Pilot", "Standard", "Subsidy", "Grid_upgrade", "PPP", "Capacity_building", "Public_awareness", "R_and_D", "Other"]
}

**Rules**:
1. Only use values from the allowed enums above. If uncertain, omit or use "Other".
2. For years, extract only 4-digit integers (e.g., 2020). Ignore months/days.
3. "quantitative_targets" must be a list; if none, return [].
4. Lists must be arrays (e.g., ["Transport"]), not strings.
5. If a field is not mentioned, use null (for single values) or [] (for lists).

**Example Input**:
"In line with the objectives of the 'Guidance on Accelerating the Construction of Electric Vehicle Charging Infrastructure' by the State Council, this guideline seeks to set out a framework through which to systematically improve the charging infrastructure of electric vehicles (EV) and to promote the healthy and rapid development of the EV industry. The guideline sets targets in accordance with projections of EV demand. Thus by 2020: More than 12,000 new centralised charging and battery replacement stations will be added. More than 4.8 million decentralised charging stations shall be added to meet the expected demand of 5 million EV. Development of public service areas such as public transport and rental services will be prioritised too. Key focus areas: Promoting charging infrastructure construction, Strengthening grid capacity, Accelerating standardised specifications, Exploring sustainable business models, Develop pilot projects."

**Example Output**:
{
  "instrument_type": ["Infrastructure"],
  "target_sector": ["Transport", "Energy"],
  "geographic_scope": ["National"],
  "temporal_scope": {
    "start_year": null,
    "target_year": 2020,
    "duration": null
  },
  "quantitative_targets": [
    {
      "metric": "centralised_charging_and_battery_replacement_stations",
      "value": 12000,
      "unit": "stations",
      "target_year": 2020,
      "coverage": "electric vehicles"
    },
    {
      "metric": "decentralised_charging_stations",
      "value": 4800000,
      "unit": "stations",
      "target_year": 2020,
      "coverage": "5 million electric vehicles"
    }
  ],
  "policy_actors": {
    "issuer": "State Council",
    "implementer": ["local governments", "power grid companies"],
    "affected_parties": ["EV users", "charging operators", "public transport operators", "rental service providers"]
  },
  "implementation_mechanisms": ["Pilot", "Standard", "Grid_upgrade", "PPP"]
}

**Now process the following policy text**:
{{POLICY_TEXT}}

Remember: ONLY output the JSON. No preamble. No apology. No markdown. If you cannot parse, output nulls and empty lists as per schema."""


class AnalysisAgentError(Exception):
    """Base exception for AnalysisAgent errors"""
    pass


class LLMTimeoutError(AnalysisAgentError):
    """Raised when LLM request times out"""
    pass


class LLMRateLimitError(AnalysisAgentError):
    """Raised when LLM rate limit is exceeded"""
    pass


class JSONParseError(AnalysisAgentError):
    """Raised when LLM returns non-JSON content"""
    pass


class SchemaValidationError(AnalysisAgentError):
    """Raised when extracted entities don't match expected schema"""
    pass


class AnalysisAgent:
    """
    Agent responsible for extracting structured entities from policy text
    using Kimi LLM API with retry logic and error handling.
    """
    
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize AnalysisAgent with LLM configuration.
        
        Args:
            api_base: Base URL for Kimi API (defaults to settings.KIMI_API_BASE)
            api_key: API key for authentication (defaults to settings.KIMI_API_KEY)
            model: Model name to use (defaults to settings.KIMI_MODEL)
            max_retries: Maximum number of retry attempts (defaults to settings.MAX_RETRIES)
            timeout: Request timeout in seconds (defaults to settings.REQUEST_TIMEOUT)
        """
        self.api_base = api_base or settings.KIMI_API_BASE
        self.api_key = api_key or settings.KIMI_API_KEY
        self.model = model or settings.KIMI_MODEL
        self.max_retries = max_retries or settings.MAX_RETRIES
        self.timeout = timeout or settings.REQUEST_TIMEOUT
        
        if not self.api_key:
            raise ValueError("KIMI_API_KEY must be set in environment or config")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
    
    def extract_entities(self, policy_text: str) -> Optional[PolicyEntities]:
        """
        Extract structured entities from policy text with retry logic.
        
        Args:
            policy_text: The policy text to analyze
            
        Returns:
            PolicyEntities object if successful, None if all retries fail
            
        Raises:
            AnalysisAgentError: If extraction fails after all retries
        """
        if not isinstance(policy_text, str) or not policy_text.strip():
            raise ValueError("policy_text must be a non-empty string")
        
        # Build prompt
        prompt = PROMPT_TEMPLATE.replace("{{POLICY_TEXT}}", policy_text.strip())
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Call LLM API
                raw_output = self._call_llm(prompt)
                
                # Parse JSON
                entities_dict = self._parse_json(raw_output, attempt)
                
                # Validate schema
                entities = self._validate_schema(entities_dict)
                
                return entities
                
            except JSONParseError as e:
                last_error = e
                print(f"  Attempt {attempt + 1}/{self.max_retries}: JSON parse error - {str(e)[:100]}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            except SchemaValidationError as e:
                last_error = e
                print(f"  Attempt {attempt + 1}/{self.max_retries}: Schema validation error - {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
                
            except LLMTimeoutError as e:
                last_error = e
                print(f"  Attempt {attempt + 1}/{self.max_retries}: Timeout error")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
                
            except LLMRateLimitError as e:
                last_error = e
                print(f"  Attempt {attempt + 1}/{self.max_retries}: Rate limit exceeded")
                if attempt < self.max_retries - 1:
                    time.sleep(5)  # Longer wait for rate limits
                continue
                
            except Exception as e:
                last_error = e
                print(f"  Attempt {attempt + 1}/{self.max_retries}: Unexpected error - {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        
        # All retries failed
        print(f"  All {self.max_retries} retries failed. Last error: {last_error}")
        return None
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call Kimi LLM API with timeout handling.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Raw text response from LLM
            
        Raises:
            LLMTimeoutError: If request times out
            LLMRateLimitError: If rate limit is exceeded
            AnalysisAgentError: For other API errors
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=self.timeout
            )
            
            raw_output = completion.choices[0].message.content.strip()
            return raw_output
            
        except Timeout:
            raise LLMTimeoutError(f"LLM request timed out after {self.timeout} seconds")
            
        except RateLimitError as e:
            raise LLMRateLimitError(f"LLM rate limit exceeded: {str(e)}")
            
        except APIError as e:
            raise AnalysisAgentError(f"LLM API error: {str(e)}")
            
        except Exception as e:
            raise AnalysisAgentError(f"Unexpected error calling LLM: {str(e)}")
    
    def _parse_json(self, raw_output: str, attempt: int) -> Dict[str, Any]:
        """
        Parse JSON from LLM output.
        
        Args:
            raw_output: Raw text from LLM
            attempt: Current attempt number (for logging)
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            JSONParseError: If JSON parsing fails
        """
        try:
            # Try to parse as-is
            return json.loads(raw_output)
            
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in raw_output:
                try:
                    json_start = raw_output.index("```json") + 7
                    json_end = raw_output.index("```", json_start)
                    json_str = raw_output[json_start:json_end].strip()
                    return json.loads(json_str)
                except (ValueError, json.JSONDecodeError):
                    pass
            
            # Try to extract JSON from curly braces
            if "{" in raw_output and "}" in raw_output:
                try:
                    json_start = raw_output.index("{")
                    json_end = raw_output.rindex("}") + 1
                    json_str = raw_output[json_start:json_end]
                    return json.loads(json_str)
                except (ValueError, json.JSONDecodeError):
                    pass
            
            raise JSONParseError(
                f"Failed to parse JSON from LLM output. "
                f"First 200 chars: {raw_output[:200]}"
            )
    
    def _validate_schema(self, entities_dict: Dict[str, Any]) -> PolicyEntities:
        """
        Validate extracted entities against expected schema.
        
        Args:
            entities_dict: Dictionary of extracted entities
            
        Returns:
            Validated PolicyEntities object
            
        Raises:
            SchemaValidationError: If validation fails
        """
        required_keys = [
            'instrument_type',
            'target_sector',
            'geographic_scope',
            'temporal_scope',
            'quantitative_targets',
            'policy_actors',
            'implementation_mechanisms'
        ]
        
        # Check for required keys
        missing_keys = [key for key in required_keys if key not in entities_dict]
        if missing_keys:
            raise SchemaValidationError(
                f"Missing required keys: {', '.join(missing_keys)}"
            )
        
        # Try to create Pydantic model (will validate types)
        try:
            entities = PolicyEntities(**entities_dict)
            return entities
            
        except Exception as e:
            raise SchemaValidationError(
                f"Failed to validate entities schema: {str(e)}"
            )
    
    def extract_entities_dict(self, policy_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract entities and return as dictionary (for backward compatibility).
        
        Args:
            policy_text: The policy text to analyze
            
        Returns:
            Dictionary of extracted entities, or None if extraction fails
        """
        entities = self.extract_entities(policy_text)
        if entities is None:
            return None
        return entities.model_dump()
