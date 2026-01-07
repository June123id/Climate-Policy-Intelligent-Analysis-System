"""
RouterAgent: Intent classification agent for routing user requests
Determines whether user input is a query or analysis request
"""
import sys
import os
from typing import Literal
from openai import OpenAI, Timeout, RateLimitError, APIError

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings


# Lightweight prompt for intent classification
INTENT_PROMPT_TEMPLATE = """Classify the user's intent as either "query" or "analysis".

- "query": User wants to search/find existing policies in the database
  Examples: "查找中国的交通政策", "Show me energy policies from 2020", "Find regulatory instruments"
  
- "analysis": User wants to analyze a new policy text they provide
  Examples: "分析这段政策：...", "Extract entities from this policy", "Analyze: [policy text]"

User input: {user_input}

Respond with ONLY one word: "query" or "analysis"."""


class RouterAgentError(Exception):
    """Base exception for RouterAgent errors"""
    pass


class IntentClassificationError(RouterAgentError):
    """Raised when intent classification fails"""
    pass


class RouterAgent:
    """
    Agent responsible for classifying user intent and routing to appropriate handler.
    Uses lightweight LLM prompt to minimize token consumption.
    """
    
    def __init__(
        self,
        api_base: str = None,
        api_key: str = None,
        model: str = None,
        timeout: int = None
    ):
        """
        Initialize RouterAgent with LLM configuration.
        
        Args:
            api_base: Base URL for Kimi API (defaults to settings.KIMI_API_BASE)
            api_key: API key for authentication (defaults to settings.KIMI_API_KEY)
            model: Model name to use (defaults to settings.KIMI_MODEL)
            timeout: Request timeout in seconds (defaults to settings.REQUEST_TIMEOUT)
        """
        self.api_base = api_base or settings.KIMI_API_BASE
        self.api_key = api_key or settings.KIMI_API_KEY
        self.model = model or settings.KIMI_MODEL
        self.timeout = timeout or settings.REQUEST_TIMEOUT
        
        if not self.api_key:
            raise ValueError("KIMI_API_KEY must be set in environment or config")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
    
    def route(self, user_input: str) -> Literal["query", "analysis"]:
        """
        Classify user intent and return routing decision.
        
        Args:
            user_input: Raw user input text
            
        Returns:
            "query" for search/retrieval requests, "analysis" for policy analysis requests
            
        Raises:
            ValueError: If user_input is empty
            RouterAgentError: If intent classification fails
        """
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string")
        
        try:
            # Call LLM for intent classification
            intent = self._classify_intent(user_input.strip())
            return intent
            
        except Exception as e:
            # Fallback logic if LLM fails
            print(f"Intent classification failed: {str(e)}, using fallback logic")
            return self._fallback_classification(user_input)
    
    def _classify_intent(self, user_input: str) -> Literal["query", "analysis"]:
        """
        Use LLM to classify user intent.
        
        Args:
            user_input: User input text
            
        Returns:
            Classified intent ("query" or "analysis")
            
        Raises:
            IntentClassificationError: If classification fails
        """
        # Build lightweight prompt
        prompt = INTENT_PROMPT_TEMPLATE.format(user_input=user_input)
        
        try:
            # Call LLM with low temperature for deterministic output
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,  # Only need one word
                timeout=self.timeout
            )
            
            raw_output = completion.choices[0].message.content.strip().lower()
            
            # Parse response
            if "query" in raw_output:
                return "query"
            elif "analysis" in raw_output:
                return "analysis"
            else:
                raise IntentClassificationError(
                    f"Unexpected LLM response: {raw_output}"
                )
                
        except Timeout:
            raise IntentClassificationError(
                f"LLM request timed out after {self.timeout} seconds"
            )
            
        except RateLimitError as e:
            raise IntentClassificationError(f"LLM rate limit exceeded: {str(e)}")
            
        except APIError as e:
            raise IntentClassificationError(f"LLM API error: {str(e)}")
            
        except Exception as e:
            raise IntentClassificationError(f"Unexpected error: {str(e)}")
    
    def _fallback_classification(self, user_input: str) -> Literal["query", "analysis"]:
        """
        Fallback heuristic-based classification when LLM fails.
        
        Uses simple keyword matching to determine intent:
        - If input contains analysis keywords or is very long, classify as "analysis"
        - Otherwise, classify as "query"
        
        Args:
            user_input: User input text
            
        Returns:
            Fallback intent classification
        """
        user_input_lower = user_input.lower()
        
        # Keywords indicating analysis intent
        analysis_keywords = [
            "分析", "analyze", "extract", "提取", "实体",
            "parse", "解析", "这段政策", "this policy",
            "以下政策", "following policy"
        ]
        
        # Keywords indicating query intent
        query_keywords = [
            "查找", "搜索", "find", "search", "show", "list",
            "get", "retrieve", "获取", "显示", "哪些"
        ]
        
        # Check for analysis keywords
        if any(keyword in user_input_lower for keyword in analysis_keywords):
            return "analysis"
        
        # Check for query keywords
        if any(keyword in user_input_lower for keyword in query_keywords):
            return "query"
        
        # If input is very long (>200 chars), likely a policy text for analysis
        if len(user_input) > 200:
            return "analysis"
        
        # Default to query for short, ambiguous inputs
        return "query"
