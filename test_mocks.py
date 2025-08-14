"""
Mock Responses and Test Fixtures
Pre-recorded API responses for fast testing without actual API calls
"""

import json
from typing import Dict, Any, List
from unittest.mock import MagicMock
import numpy as np


class MockGeminiResponses:
    """Pre-recorded Gemini API responses for testing"""
    
    # Mock responses for different techniques
    SENTIMENT_RESPONSES = {
        "This smartphone is absolutely amazing! Best purchase ever!": "positive",
        "This product is terrible and doesn't work at all.": "negative",
        "The weather is okay today. Nothing special.": "neutral",
        "I love this new coffee shop!": "positive"
    }
    
    MATH_RESPONSES = {
        "If a pizza costs $12 and is cut into 8 slices, how much does each slice cost?": 
        "To find the cost per slice, I need to divide the total cost by the number of slices.\n\nStep 1: Identify the given information\n- Total cost of pizza: $12\n- Number of slices: 8\n\nStep 2: Calculate cost per slice\nCost per slice = Total cost ÷ Number of slices\nCost per slice = $12 ÷ 8 = $1.50\n\nTherefore, each slice costs $1.50.",
        
        "If 2 + 2 = 4, what is 3 + 3?": 
        "Let me solve this step by step.\n\nStep 1: Understand the pattern\nWe know that 2 + 2 = 4\n\nStep 2: Apply the same logic\n3 + 3 means adding 3 to itself\n3 + 3 = 6\n\nTherefore, 3 + 3 = 6.",
        
        "A car rental costs $30 per day plus $0.25 per mile. If I rent for 3 days and drive 150 miles, what's the total cost?":
        "Let me calculate the total rental cost step by step.\n\nStep 1: Calculate daily cost\nDaily rate: $30 per day\nNumber of days: 3\nDaily cost = $30 × 3 = $90\n\nStep 2: Calculate mileage cost\nMileage rate: $0.25 per mile\nMiles driven: 150\nMileage cost = $0.25 × 150 = $37.50\n\nStep 3: Calculate total cost\nTotal cost = Daily cost + Mileage cost\nTotal cost = $90 + $37.50 = $127.50\n\nTherefore, the total rental cost is $127.50."
    }
    
    LOGIC_RESPONSES = {
        "All cats are animals. Some animals are pets. Can we conclude that some cats are pets?":
        "Let me analyze this logical reasoning step by step.\n\nStep 1: Examine the premises\n- Premise 1: All cats are animals\n- Premise 2: Some animals are pets\n\nStep 2: Analyze the logical relationship\nFrom these premises alone, we cannot definitively conclude that some cats are pets.\n\nStep 3: Explain why\nWhile all cats are animals, and some animals are pets, the premises don't specify which animals are pets. It's possible that cats are among the animals that are pets, but it's also possible that only other animals (like dogs, birds, etc.) are the pets mentioned.\n\nConclusion: No, we cannot logically conclude that some cats are pets based solely on these premises. We would need additional information to make this conclusion.",
        
        "All birds can fly. Penguins are birds. But penguins cannot fly. What's wrong with this reasoning?":
        "Let me analyze this logical problem step by step.\n\nStep 1: Identify the statements\n- Statement 1: All birds can fly\n- Statement 2: Penguins are birds\n- Statement 3: Penguins cannot fly\n\nStep 2: Identify the logical issue\nThis is a classic example of a false premise leading to a logical contradiction.\n\nStep 3: Explain the problem\nThe issue is with the first statement: 'All birds can fly.' This is factually incorrect. While many birds can fly, not all birds have this ability.\n\nStep 4: Provide the correct reasoning\nThe correct premise should be: 'Most birds can fly, but some birds cannot fly.'\n\nConclusion: The reasoning contains a false universal statement. Penguins are indeed birds, but they are flightless birds, which disproves the premise that 'all birds can fly.'"
    }
    
    NER_RESPONSES = {
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.":
        "ORGANIZATION: Apple Inc. | PERSON: Steve Jobs | LOCATION: Cupertino, California",
        
        "Tim Cook is the CEO of Apple Inc. based in Cupertino, California.":
        "PERSON: Tim Cook | ORGANIZATION: Apple Inc. | LOCATION: Cupertino, California",
        
        "Elon Musk founded SpaceX in 2002 in California.":
        "PERSON: Elon Musk | ORGANIZATION: SpaceX | DATE: 2002 | LOCATION: California"
    }
    
    META_PROMPTING_RESPONSES = {
        "optimize_prompt": """Here's an improved version of the prompt:

**Optimized Prompt:**
"Analyze the sentiment of the following text and classify it as positive, negative, or neutral. Consider the overall tone, emotional indicators, and context.

Text: {text}

Please provide:
1. Classification: [positive/negative/neutral]
2. Confidence level: [high/medium/low]
3. Key indicators: [list the words/phrases that influenced your decision]

Classification:"""
    }
    
    TREE_OF_THOUGHT_RESPONSES = {
        "approach_1": "Direct analytical approach: Break down the problem into core components and analyze each systematically. This involves identifying root causes, evaluating current solutions, and developing targeted interventions.",
        
        "approach_2": "Creative problem-solving approach: Use innovative thinking and unconventional methods. This includes brainstorming alternative solutions, thinking outside traditional frameworks, and exploring novel approaches.",
        
        "approach_3": "Systematic breakdown approach: Divide the complex problem into smaller, manageable sub-problems. Address each component individually and then integrate solutions for a comprehensive approach."
    }
    
    SELF_CONSISTENCY_RESPONSES = {
        "What are the key benefits of exercise?": [
            "Exercise provides numerous benefits including improved cardiovascular health, stronger muscles and bones, better mental health, weight management, and increased energy levels.",
            "The main benefits of exercise include enhanced physical fitness, reduced risk of chronic diseases, improved mood and mental well-being, better sleep quality, and increased longevity.",
            "Key exercise benefits are: better heart health, stronger immune system, improved mental health, weight control, increased strength and endurance, and better overall quality of life."
        ]
    }


class MockGeminiClient:
    """Mock Gemini client that returns pre-recorded responses"""
    
    def __init__(self):
        self.responses = MockGeminiResponses()
        self.call_count = 0
    
    def generate_content(self, model, contents, config):
        """Mock generate_content method"""
        self.call_count += 1
        
        # Extract text from contents
        text = contents[0].parts[0].text if contents and contents[0].parts else ""
        
        # Return appropriate mock response based on text content
        mock_response = self._get_mock_response(text)
        
        # Create mock response object
        response = MagicMock()
        response.candidates = [MagicMock()]
        response.candidates[0].content = MagicMock()
        response.candidates[0].content.parts = [MagicMock()]
        response.candidates[0].content.parts[0].text = mock_response
        
        return response
    
    def _get_mock_response(self, text: str) -> str:
        """Get appropriate mock response based on input text"""
        text_lower = text.lower()
        
        # Sentiment analysis
        for key, response in self.responses.SENTIMENT_RESPONSES.items():
            if key.lower() in text_lower:
                return response
        
        # Math problems
        for key, response in self.responses.MATH_RESPONSES.items():
            if any(word in text_lower for word in ["pizza", "cost", "slice", "rental", "mile"]):
                return self.responses.MATH_RESPONSES.get(key, "The answer is 42.")
        
        # Logic problems
        for key, response in self.responses.LOGIC_RESPONSES.items():
            if any(word in text_lower for word in ["cats", "animals", "birds", "penguins", "logic"]):
                return self.responses.LOGIC_RESPONSES.get(key, "This requires logical analysis.")
        
        # Named Entity Recognition
        for key, response in self.responses.NER_RESPONSES.items():
            if any(word in text_lower for word in ["apple", "steve jobs", "tim cook", "elon musk"]):
                return self.responses.NER_RESPONSES.get(key, "PERSON: John Doe | ORGANIZATION: Company Inc.")
        
        # Meta-prompting
        if any(word in text_lower for word in ["optimize", "improve", "prompt", "better"]):
            return self.responses.META_PROMPTING_RESPONSES["optimize_prompt"]
        
        # Tree of thought
        if "approach 1" in text_lower:
            return self.responses.TREE_OF_THOUGHT_RESPONSES["approach_1"]
        elif "approach 2" in text_lower:
            return self.responses.TREE_OF_THOUGHT_RESPONSES["approach_2"]
        elif "approach 3" in text_lower:
            return self.responses.TREE_OF_THOUGHT_RESPONSES["approach_3"]
        
        # Self-consistency
        for key, responses in self.responses.SELF_CONSISTENCY_RESPONSES.items():
            if any(word in text_lower for word in key.lower().split()):
                return responses[0]  # Return first response
        
        # Default response
        return "This is a mock response for testing purposes."


class MockEmbeddingGenerator:
    """Mock embedding generator for testing"""
    
    @staticmethod
    def generate_embedding(text: str) -> np.ndarray:
        """Generate a mock embedding vector"""
        # Create deterministic embedding based on text hash
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to numbers and create 768-dimensional vector
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
            embedding.append(value)
        
        # Pad or truncate to 768 dimensions
        while len(embedding) < 768:
            embedding.extend(embedding[:768-len(embedding)])
        
        return np.array(embedding[:768])


def create_mock_gemini_service():
    """Create a mock AdvancedPromptingGemini service for testing"""
    from unittest.mock import MagicMock, patch
    
    # Create mock service
    mock_service = MagicMock()
    
    # Set up mock client
    mock_client = MockGeminiClient()
    mock_service.client = MagicMock()
    mock_service.client.models = mock_client
    
    # Set up basic properties
    mock_service.api_key = "mock_api_key_for_testing"
    mock_service.model = "gemini-2.5-flash"
    
    return mock_service, mock_client


# Test data fixtures
TEST_FIXTURES = {
    "sample_texts": [
        "This smartphone is absolutely amazing! Best purchase ever!",
        "This product is terrible and doesn't work at all.",
        "The weather is okay today. Nothing special.",
        "I love this new coffee shop!"
    ],
    
    "sample_math_problems": [
        "If a pizza costs $12 and is cut into 8 slices, how much does each slice cost?",
        "If 2 + 2 = 4, what is 3 + 3?",
        "A car rental costs $30 per day plus $0.25 per mile. If I rent for 3 days and drive 150 miles, what's the total cost?"
    ],
    
    "sample_logic_problems": [
        "All cats are animals. Some animals are pets. Can we conclude that some cats are pets?",
        "All birds can fly. Penguins are birds. But penguins cannot fly. What's wrong with this reasoning?"
    ],
    
    "sample_ner_texts": [
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "Tim Cook is the CEO of Apple Inc. based in Cupertino, California.",
        "Elon Musk founded SpaceX in 2002 in California."
    ]
}


def get_expected_response(text: str, technique: str) -> str:
    """Get expected response for a given text and technique"""
    responses = MockGeminiResponses()
    
    if technique == "sentiment":
        return responses.SENTIMENT_RESPONSES.get(text, "positive")
    elif technique == "math":
        return responses.MATH_RESPONSES.get(text, "The answer is 42.")
    elif technique == "logic":
        return responses.LOGIC_RESPONSES.get(text, "This requires logical analysis.")
    elif technique == "ner":
        return responses.NER_RESPONSES.get(text, "PERSON: John Doe")
    else:
        return "Mock response"