"""
Gemini Client Utility
Provides a centralized client for all prompting techniques
"""

import os
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
import asyncio
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GeminiClient:
    """Centralized Gemini client for advanced prompting techniques"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        """
        Initialize Gemini client
        
        Args:
            api_key: Gemini API key (if None, uses environment variable)
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file or pass it as a parameter.")
        self.model = model
        self.client = genai.Client(api_key=self.api_key)
    
    def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking_budget: int = 0
    ) -> str:
        """
        Generate a single response from Gemini
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            thinking_budget: Thinking budget for reasoning
            
        Returns:
            Generated response text
        """
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            temperature=temperature
        )
        
        if max_tokens:
            config.max_output_tokens = max_tokens
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        
        return response.candidates[0].content.parts[0].text
    
    async def generate_multiple_responses(
        self, 
        prompt: str, 
        num_responses: int = 3,
        temperature: float = 0.8,
        thinking_budget: int = 0
    ) -> List[str]:
        """
        Generate multiple responses asynchronously
        
        Args:
            prompt: Input prompt
            num_responses: Number of responses to generate
            temperature: Sampling temperature
            thinking_budget: Thinking budget for reasoning
            
        Returns:
            List of generated responses
        """
        tasks = []
        for _ in range(num_responses):
            task = asyncio.create_task(
                self._async_generate(prompt, temperature, thinking_budget)
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return responses
    
    async def _async_generate(
        self, 
        prompt: str, 
        temperature: float,
        thinking_budget: int
    ) -> str:
        """Async wrapper for generate_response"""
        return self.generate_response(prompt, temperature, thinking_budget=thinking_budget)
    
    def generate_with_system_prompt(
        self, 
        system_prompt: str, 
        user_prompt: str,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response with system and user prompts
        
        Args:
            system_prompt: System instruction
            user_prompt: User input
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        combined_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
        return self.generate_response(combined_prompt, temperature)
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def validate_response(self, response: str) -> bool:
        """
        Validate if response is valid and not empty
        
        Args:
            response: Generated response
            
        Returns:
            True if valid, False otherwise
        """
        return bool(response and response.strip() and len(response.strip()) > 10)


class PromptTemplate:
    """Template class for structured prompts"""
    
    def __init__(self, template: str, variables: List[str]):
        """
        Initialize prompt template
        
        Args:
            template: Template string with {variable} placeholders
            variables: List of variable names
        """
        self.template = template
        self.variables = variables
    
    def format(self, **kwargs) -> str:
        """
        Format template with provided variables
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Formatted prompt
        """
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing variables: {missing_vars}")
        
        return self.template.format(**kwargs)
    
    def get_variables(self) -> List[str]:
        """Get list of template variables"""
        return self.variables.copy()


# Common prompt templates
TEMPLATES = {
    "few_shot": PromptTemplate(
        template="""Here are some examples:

{examples}

Now, please {task}: {input}""",
        variables=["examples", "task", "input"]
    ),
    
    "chain_of_thought": PromptTemplate(
        template="""Let's solve this step by step.

Problem: {problem}

Step-by-step solution:""",
        variables=["problem"]
    ),
    
    "self_consistency": PromptTemplate(
        template="""Question: {question}

Please provide a clear and accurate answer. Think carefully about your response.""",
        variables=["question"]
    ),
    
    "meta_prompt": PromptTemplate(
        template="""I need to create an effective prompt for the following task: {task}

Current prompt: {current_prompt}

Please analyze this prompt and suggest improvements to make it more effective. Consider:
1. Clarity and specificity
2. Context and examples
3. Output format requirements
4. Potential edge cases

Improved prompt:""",
        variables=["task", "current_prompt"]
    )
}