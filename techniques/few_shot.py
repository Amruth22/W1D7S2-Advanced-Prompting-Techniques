"""
Few-shot Learning Implementation
Enables learning from minimal examples to perform new tasks
"""

from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass
from utils.gemini_client import GeminiClient, TEMPLATES


@dataclass
class Example:
    """Represents a single few-shot example"""
    input: str
    output: str
    explanation: Optional[str] = None


class FewShotLearner:
    """Few-shot learning implementation for Gemini"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Few-shot learner
        
        Args:
            api_key: Gemini API key
        """
        self.client = GeminiClient(api_key)
        self.examples_cache = {}
    
    def add_examples(self, task_name: str, examples: List[Example]) -> None:
        """
        Add examples for a specific task
        
        Args:
            task_name: Name of the task
            examples: List of examples
        """
        self.examples_cache[task_name] = examples
    
    def format_examples(self, examples: List[Example], include_explanations: bool = False) -> str:
        """
        Format examples into a string for the prompt
        
        Args:
            examples: List of examples
            include_explanations: Whether to include explanations
            
        Returns:
            Formatted examples string
        """
        formatted = []
        for i, example in enumerate(examples, 1):
            formatted_example = f"Example {i}:\nInput: {example.input}\nOutput: {example.output}"
            
            if include_explanations and example.explanation:
                formatted_example += f"\nExplanation: {example.explanation}"
            
            formatted.append(formatted_example)
        
        return "\n\n".join(formatted)
    
    def classify_sentiment(
        self, 
        text: str, 
        examples: Optional[List[Example]] = None,
        custom_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Classify sentiment using few-shot learning
        
        Args:
            text: Text to classify
            examples: Custom examples (if None, uses default)
            custom_labels: Custom sentiment labels
            
        Returns:
            Classification result with confidence
        """
        if examples is None:
            examples = self._get_default_sentiment_examples(custom_labels)
        
        examples_str = self.format_examples(examples, include_explanations=True)
        
        prompt = TEMPLATES["few_shot"].format(
            examples=examples_str,
            task="classify the sentiment",
            input=text
        )
        
        response = self.client.generate_response(prompt, temperature=0.3)
        
        return {
            "text": text,
            "prediction": response.strip(),
            "examples_used": len(examples),
            "confidence": self._estimate_confidence(response)
        }
    
    def extract_entities(
        self, 
        text: str, 
        entity_types: List[str],
        examples: Optional[List[Example]] = None
    ) -> Dict[str, Any]:
        """
        Extract named entities using few-shot learning
        
        Args:
            text: Text to process
            entity_types: Types of entities to extract
            examples: Custom examples
            
        Returns:
            Extracted entities
        """
        if examples is None:
            examples = self._get_default_ner_examples(entity_types)
        
        examples_str = self.format_examples(examples)
        
        prompt = TEMPLATES["few_shot"].format(
            examples=examples_str,
            task=f"extract {', '.join(entity_types)} entities",
            input=text
        )
        
        response = self.client.generate_response(prompt, temperature=0.2)
        
        return {
            "text": text,
            "entities": self._parse_entities(response),
            "entity_types": entity_types,
            "examples_used": len(examples)
        }
    
    def generate_text(
        self, 
        prompt: str, 
        style: str,
        examples: Optional[List[Example]] = None
    ) -> Dict[str, Any]:
        """
        Generate text in a specific style using few-shot learning
        
        Args:
            prompt: Generation prompt
            style: Desired style
            examples: Style examples
            
        Returns:
            Generated text with metadata
        """
        if examples is None:
            examples = self._get_default_style_examples(style)
        
        examples_str = self.format_examples(examples)
        
        full_prompt = TEMPLATES["few_shot"].format(
            examples=examples_str,
            task=f"generate text in {style} style",
            input=prompt
        )
        
        response = self.client.generate_response(full_prompt, temperature=0.8)
        
        return {
            "prompt": prompt,
            "style": style,
            "generated_text": response.strip(),
            "examples_used": len(examples)
        }
    
    def solve_math_word_problems(
        self, 
        problem: str,
        examples: Optional[List[Example]] = None
    ) -> Dict[str, Any]:
        """
        Solve math word problems using few-shot learning
        
        Args:
            problem: Math word problem
            examples: Problem-solving examples
            
        Returns:
            Solution with step-by-step reasoning
        """
        if examples is None:
            examples = self._get_default_math_examples()
        
        examples_str = self.format_examples(examples, include_explanations=True)
        
        prompt = TEMPLATES["few_shot"].format(
            examples=examples_str,
            task="solve this math word problem step by step",
            input=problem
        )
        
        response = self.client.generate_response(prompt, temperature=0.3)
        
        return {
            "problem": problem,
            "solution": response.strip(),
            "examples_used": len(examples)
        }
    
    def custom_task(
        self, 
        task_description: str,
        input_text: str,
        examples: List[Example],
        temperature: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform custom task using few-shot learning
        
        Args:
            task_description: Description of the task
            input_text: Input to process
            examples: Task examples
            temperature: Generation temperature
            
        Returns:
            Task result
        """
        examples_str = self.format_examples(examples, include_explanations=True)
        
        prompt = TEMPLATES["few_shot"].format(
            examples=examples_str,
            task=task_description,
            input=input_text
        )
        
        response = self.client.generate_response(prompt, temperature=temperature)
        
        return {
            "task": task_description,
            "input": input_text,
            "output": response.strip(),
            "examples_used": len(examples)
        }
    
    def _get_default_sentiment_examples(self, custom_labels: Optional[List[str]] = None) -> List[Example]:
        """Get default sentiment classification examples"""
        labels = custom_labels or ["positive", "negative", "neutral"]
        
        examples = [
            Example(
                input="I love this product! It's amazing and works perfectly.",
                output="positive",
                explanation="The text contains positive words like 'love', 'amazing', and 'perfectly'."
            ),
            Example(
                input="This is terrible. I hate it and want my money back.",
                output="negative",
                explanation="The text contains negative words like 'terrible', 'hate', and expresses dissatisfaction."
            ),
            Example(
                input="The weather is okay today. Nothing special.",
                output="neutral",
                explanation="The text is neither particularly positive nor negative, using neutral language."
            )
        ]
        
        return examples
    
    def _get_default_ner_examples(self, entity_types: List[str]) -> List[Example]:
        """Get default named entity recognition examples"""
        return [
            Example(
                input="John Smith works at Google in Mountain View, California.",
                output="PERSON: John Smith\nORGANIZATION: Google\nLOCATION: Mountain View, California"
            ),
            Example(
                input="Apple Inc. was founded by Steve Jobs on April 1, 1976.",
                output="ORGANIZATION: Apple Inc.\nPERSON: Steve Jobs\nDATE: April 1, 1976"
            ),
            Example(
                input="The meeting is scheduled for tomorrow at 3 PM in New York.",
                output="TIME: tomorrow at 3 PM\nLOCATION: New York"
            )
        ]
    
    def _get_default_style_examples(self, style: str) -> List[Example]:
        """Get default style examples"""
        if style.lower() == "formal":
            return [
                Example(
                    input="Tell me about the weather",
                    output="I would be pleased to provide you with information regarding the current meteorological conditions."
                ),
                Example(
                    input="This is bad",
                    output="This situation presents certain challenges that require attention."
                )
            ]
        elif style.lower() == "casual":
            return [
                Example(
                    input="Please provide information about the product",
                    output="Hey! So here's the scoop on this product..."
                ),
                Example(
                    input="I require assistance",
                    output="No worries! I'm here to help you out."
                )
            ]
        else:
            return [
                Example(
                    input="Example input",
                    output=f"Example output in {style} style"
                )
            ]
    
    def _get_default_math_examples(self) -> List[Example]:
        """Get default math word problem examples"""
        return [
            Example(
                input="Sarah has 15 apples. She gives 7 to her friend. How many apples does she have left?",
                output="8 apples",
                explanation="Sarah starts with 15 apples. She gives away 7 apples. 15 - 7 = 8 apples remaining."
            ),
            Example(
                input="A rectangle has a length of 8 meters and width of 5 meters. What is its area?",
                output="40 square meters",
                explanation="Area of rectangle = length × width. Area = 8 × 5 = 40 square meters."
            )
        ]
    
    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence based on response characteristics"""
        # Simple heuristic based on response length and certainty indicators
        certainty_words = ["definitely", "clearly", "obviously", "certainly", "sure"]
        uncertainty_words = ["maybe", "perhaps", "possibly", "might", "could"]
        
        certainty_count = sum(1 for word in certainty_words if word in response.lower())
        uncertainty_count = sum(1 for word in uncertainty_words if word in response.lower())
        
        base_confidence = 0.7
        confidence_adjustment = (certainty_count - uncertainty_count) * 0.1
        
        return max(0.1, min(0.95, base_confidence + confidence_adjustment))
    
    def _parse_entities(self, response: str) -> Dict[str, List[str]]:
        """Parse entities from response"""
        entities = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                entity_type, entity_value = line.split(':', 1)
                entity_type = entity_type.strip()
                entity_value = entity_value.strip()
                
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(entity_value)
        
        return entities