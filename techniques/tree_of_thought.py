"""
Tree-of-Thought (ToT) Implementation
Explores multiple reasoning paths simultaneously and selects the best approach
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
import statistics
from utils.gemini_client import GeminiClient


class ThoughtState(Enum):
    """States of thought exploration"""
    INITIAL = "initial"
    EXPLORING = "exploring"
    EVALUATED = "evaluated"
    SELECTED = "selected"
    PRUNED = "pruned"


@dataclass
class Thought:
    """Represents a single thought in the tree"""
    id: str
    content: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    score: float = 0.0
    state: ThoughtState = ThoughtState.INITIAL
    reasoning: str = ""
    evaluation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThoughtTree:
    """Represents the complete tree of thoughts"""
    root_id: str
    thoughts: Dict[str, Thought] = field(default_factory=dict)
    best_path: List[str] = field(default_factory=list)
    exploration_stats: Dict[str, Any] = field(default_factory=dict)


class TreeOfThought:
    """Tree-of-Thought reasoning implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tree-of-Thought reasoner
        
        Args:
            api_key: Gemini API key
        """
        self.client = GeminiClient(api_key)
        self.thought_counter = 0
    
    async def solve_complex_problem(
        self,
        problem: str,
        max_depth: int = 4,
        branching_factor: int = 3,
        evaluation_criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Solve complex problems using tree-of-thought exploration
        
        Args:
            problem: Problem to solve
            max_depth: Maximum exploration depth
            branching_factor: Number of thoughts per level
            evaluation_criteria: Criteria for evaluating thoughts
            
        Returns:
            Solution with exploration tree
        """
        # Initialize the tree
        tree = await self._initialize_tree(problem)
        
        # Explore thoughts level by level
        for depth in range(1, max_depth + 1):
            await self._explore_level(tree, depth, branching_factor, evaluation_criteria)
            await self._evaluate_level(tree, depth, evaluation_criteria)
            await self._prune_level(tree, depth, branching_factor)
        
        # Select best path
        best_path = await self._select_best_path(tree)
        solution = await self._synthesize_solution(tree, best_path, problem)
        
        return {
            "problem": problem,
            "solution": solution,
            "best_path": best_path,
            "tree": tree,
            "exploration_stats": self._calculate_stats(tree),
            "methodology": "Tree-of-Thought"
        }
    
    async def creative_brainstorming(
        self,
        challenge: str,
        num_initial_ideas: int = 5,
        development_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Generate and develop creative ideas using ToT
        
        Args:
            challenge: Creative challenge
            num_initial_ideas: Number of initial ideas
            development_depth: How deep to develop each idea
            
        Returns:
            Developed creative solutions
        """
        tree = await self._initialize_creative_tree(challenge, num_initial_ideas)
        
        # Develop each initial idea
        for depth in range(1, development_depth + 1):
            await self._develop_creative_ideas(tree, depth)
            await self._evaluate_creative_ideas(tree, depth)
        
        # Select and refine best ideas
        best_ideas = await self._select_best_creative_ideas(tree)
        refined_solutions = await self._refine_creative_solutions(best_ideas, challenge)
        
        return {
            "challenge": challenge,
            "initial_ideas": num_initial_ideas,
            "developed_solutions": refined_solutions,
            "tree": tree,
            "methodology": "Creative Tree-of-Thought"
        }
    
    async def strategic_planning(
        self,
        goal: str,
        constraints: List[str],
        time_horizon: str,
        max_strategies: int = 4
    ) -> Dict[str, Any]:
        """
        Develop strategic plans using ToT exploration
        
        Args:
            goal: Strategic goal
            constraints: Planning constraints
            time_horizon: Time frame for the plan
            max_strategies: Maximum number of strategies to explore
            
        Returns:
            Strategic plans with analysis
        """
        tree = await self._initialize_strategic_tree(goal, constraints, time_horizon)
        
        # Explore strategic approaches
        await self._explore_strategic_approaches(tree, max_strategies)
        await self._evaluate_strategic_approaches(tree, constraints)
        
        # Develop detailed plans
        await self._develop_strategic_plans(tree)
        await self._evaluate_strategic_plans(tree, constraints)
        
        # Select optimal strategy
        optimal_strategy = await self._select_optimal_strategy(tree)
        implementation_plan = await self._create_implementation_plan(optimal_strategy, goal)
        
        return {
            "goal": goal,
            "constraints": constraints,
            "time_horizon": time_horizon,
            "optimal_strategy": optimal_strategy,
            "implementation_plan": implementation_plan,
            "tree": tree,
            "methodology": "Strategic Tree-of-Thought"
        }
    
    async def multi_perspective_analysis(
        self,
        topic: str,
        perspectives: List[str],
        analysis_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze topics from multiple perspectives using ToT
        
        Args:
            topic: Topic to analyze
            perspectives: Different perspectives to consider
            analysis_depth: Depth of analysis for each perspective
            
        Returns:
            Multi-perspective analysis
        """
        tree = await self._initialize_perspective_tree(topic, perspectives)
        
        # Analyze from each perspective
        for depth in range(1, analysis_depth + 1):
            await self._analyze_perspectives(tree, depth)
            await self._evaluate_perspective_insights(tree, depth)
        
        # Synthesize insights
        synthesis = await self._synthesize_perspectives(tree, topic)
        conflicts = await self._identify_perspective_conflicts(tree)
        consensus = await self._find_perspective_consensus(tree)
        
        return {
            "topic": topic,
            "perspectives": perspectives,
            "synthesis": synthesis,
            "conflicts": conflicts,
            "consensus": consensus,
            "tree": tree,
            "methodology": "Multi-Perspective Tree-of-Thought"
        }
    
    async def decision_making(
        self,
        decision: str,
        options: List[str],
        criteria: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Make complex decisions using ToT exploration
        
        Args:
            decision: Decision to make
            options: Available options
            criteria: Decision criteria
            weights: Criteria weights
            
        Returns:
            Decision analysis and recommendation
        """
        tree = await self._initialize_decision_tree(decision, options, criteria)
        
        # Explore each option
        await self._explore_decision_options(tree, criteria)
        await self._evaluate_decision_options(tree, criteria, weights)
        
        # Compare options
        comparison = await self._compare_options(tree, criteria, weights)
        recommendation = await self._make_recommendation(tree, comparison)
        
        return {
            "decision": decision,
            "options": options,
            "criteria": criteria,
            "weights": weights,
            "comparison": comparison,
            "recommendation": recommendation,
            "tree": tree,
            "methodology": "Decision Tree-of-Thought"
        }
    
    async def _initialize_tree(self, problem: str) -> ThoughtTree:
        """Initialize the thought tree with root problem"""
        root_id = self._generate_thought_id()
        
        root_thought = Thought(
            id=root_id,
            content=problem,
            depth=0,
            state=ThoughtState.INITIAL,
            reasoning="Root problem statement"
        )
        
        tree = ThoughtTree(root_id=root_id)
        tree.thoughts[root_id] = root_thought
        
        return tree
    
    async def _explore_level(
        self,
        tree: ThoughtTree,
        depth: int,
        branching_factor: int,
        evaluation_criteria: Optional[List[str]]
    ) -> None:
        """Explore thoughts at a specific depth level"""
        parent_thoughts = [
            t for t in tree.thoughts.values() 
            if t.depth == depth - 1 and t.state != ThoughtState.PRUNED
        ]
        
        for parent in parent_thoughts:
            # Generate child thoughts
            child_thoughts = await self._generate_child_thoughts(
                parent, branching_factor, evaluation_criteria
            )
            
            for child in child_thoughts:
                child.depth = depth
                child.parent_id = parent.id
                tree.thoughts[child.id] = child
                parent.children_ids.append(child.id)
    
    async def _generate_child_thoughts(
        self,
        parent: Thought,
        branching_factor: int,
        evaluation_criteria: Optional[List[str]]
    ) -> List[Thought]:
        """Generate child thoughts from a parent thought"""
        prompt = f"""Given this current thought/approach:
"{parent.content}"

Generate {branching_factor} different ways to continue or develop this thought further. Each should be:
1. A logical next step or alternative approach
2. Distinct from the others
3. Potentially valuable for solving the problem

Please provide {branching_factor} different continuations:"""
        
        response = await self.client._async_generate(prompt, temperature=0.8, thinking_budget=5000)
        
        # Parse the response into individual thoughts
        thought_texts = self._parse_multiple_thoughts(response, branching_factor)
        
        child_thoughts = []
        for i, text in enumerate(thought_texts):
            child_id = self._generate_thought_id()
            child = Thought(
                id=child_id,
                content=text.strip(),
                state=ThoughtState.EXPLORING,
                reasoning=f"Generated as continuation {i+1} of parent thought"
            )
            child_thoughts.append(child)
        
        return child_thoughts
    
    async def _evaluate_level(
        self,
        tree: ThoughtTree,
        depth: int,
        evaluation_criteria: Optional[List[str]]
    ) -> None:
        """Evaluate all thoughts at a specific depth level"""
        thoughts_to_evaluate = [
            t for t in tree.thoughts.values() 
            if t.depth == depth and t.state == ThoughtState.EXPLORING
        ]
        
        # Evaluate thoughts in parallel
        evaluation_tasks = [
            self._evaluate_thought(thought, evaluation_criteria)
            for thought in thoughts_to_evaluate
        ]
        
        evaluations = await asyncio.gather(*evaluation_tasks)
        
        # Update thoughts with evaluations
        for thought, evaluation in zip(thoughts_to_evaluate, evaluations):
            thought.score = evaluation["score"]
            thought.evaluation = evaluation
            thought.state = ThoughtState.EVALUATED
    
    async def _evaluate_thought(
        self,
        thought: Thought,
        evaluation_criteria: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Evaluate a single thought"""
        criteria = evaluation_criteria or [
            "logical soundness",
            "creativity",
            "feasibility",
            "potential effectiveness"
        ]
        
        prompt = f"""Please evaluate this thought/approach:
"{thought.content}"

Rate it on a scale of 1-10 for each criterion:
{chr(10).join(f"- {criterion}" for criterion in criteria)}

Also provide:
- Overall score (1-10)
- Strengths
- Weaknesses
- Potential for further development

Evaluation:"""
        
        response = await self.client._async_generate(prompt, temperature=0.3, thinking_budget=3000)
        
        # Parse evaluation
        evaluation = self._parse_evaluation(response, criteria)
        
        return evaluation
    
    async def _prune_level(
        self,
        tree: ThoughtTree,
        depth: int,
        branching_factor: int
    ) -> None:
        """Prune less promising thoughts at a depth level"""
        thoughts_at_depth = [
            t for t in tree.thoughts.values() 
            if t.depth == depth and t.state == ThoughtState.EVALUATED
        ]
        
        # Group by parent
        parent_groups = {}
        for thought in thoughts_at_depth:
            parent_id = thought.parent_id
            if parent_id not in parent_groups:
                parent_groups[parent_id] = []
            parent_groups[parent_id].append(thought)
        
        # Keep top thoughts for each parent
        keep_per_parent = max(1, branching_factor // 2)
        
        for parent_id, thoughts in parent_groups.items():
            # Sort by score
            thoughts.sort(key=lambda t: t.score, reverse=True)
            
            # Keep top thoughts
            for i, thought in enumerate(thoughts):
                if i < keep_per_parent:
                    thought.state = ThoughtState.SELECTED
                else:
                    thought.state = ThoughtState.PRUNED
    
    async def _select_best_path(self, tree: ThoughtTree) -> List[str]:
        """Select the best path through the tree"""
        # Find leaf nodes that weren't pruned
        leaf_nodes = [
            t for t in tree.thoughts.values()
            if not t.children_ids and t.state != ThoughtState.PRUNED
        ]
        
        if not leaf_nodes:
            return [tree.root_id]
        
        # Find the best leaf node
        best_leaf = max(leaf_nodes, key=lambda t: t.score)
        
        # Trace back to root
        path = []
        current = best_leaf
        
        while current:
            path.append(current.id)
            if current.parent_id:
                current = tree.thoughts[current.parent_id]
            else:
                break
        
        path.reverse()
        tree.best_path = path
        
        return path
    
    async def _synthesize_solution(
        self,
        tree: ThoughtTree,
        best_path: List[str],
        original_problem: str
    ) -> str:
        """Synthesize final solution from the best path"""
        path_thoughts = [tree.thoughts[thought_id].content for thought_id in best_path]
        
        prompt = f"""Original problem: {original_problem}

Best reasoning path found:
{chr(10).join(f"{i+1}. {thought}" for i, thought in enumerate(path_thoughts))}

Please synthesize these thoughts into a comprehensive, coherent solution to the original problem. The solution should:
1. Address the original problem directly
2. Incorporate insights from the reasoning path
3. Be practical and actionable
4. Be clearly explained

Final solution:"""
        
        solution = await self.client._async_generate(prompt, temperature=0.4, thinking_budget=5000)
        
        return solution.strip()
    
    async def _initialize_creative_tree(self, challenge: str, num_ideas: int) -> ThoughtTree:
        """Initialize tree for creative brainstorming"""
        tree = await self._initialize_tree(challenge)
        
        # Generate initial creative ideas
        prompt = f"""Creative challenge: {challenge}

Generate {num_ideas} diverse, creative initial ideas to address this challenge. Each idea should be:
1. Unique and innovative
2. Potentially feasible
3. Different from the others

Ideas:"""
        
        response = await self.client._async_generate(prompt, temperature=0.9, thinking_budget=3000)
        idea_texts = self._parse_multiple_thoughts(response, num_ideas)
        
        root = tree.thoughts[tree.root_id]
        
        for i, idea_text in enumerate(idea_texts):
            idea_id = self._generate_thought_id()
            idea = Thought(
                id=idea_id,
                content=idea_text.strip(),
                parent_id=root.id,
                depth=1,
                state=ThoughtState.EXPLORING,
                reasoning=f"Initial creative idea {i+1}"
            )
            tree.thoughts[idea_id] = idea
            root.children_ids.append(idea_id)
        
        return tree
    
    def _generate_thought_id(self) -> str:
        """Generate unique thought ID"""
        self.thought_counter += 1
        return f"thought_{self.thought_counter}"
    
    def _parse_multiple_thoughts(self, response: str, expected_count: int) -> List[str]:
        """Parse multiple thoughts from response"""
        lines = response.strip().split('\n')
        thoughts = []
        current_thought = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new thought (numbered or bulleted)
            if (line.startswith(tuple(f"{i}." for i in range(1, expected_count + 2))) or
                line.startswith(tuple(f"{i}:" for i in range(1, expected_count + 2))) or
                line.startswith(('-', '*', '•'))):
                
                if current_thought:
                    thoughts.append(current_thought.strip())
                    current_thought = ""
                
                # Remove numbering/bullets
                line = line.lstrip('0123456789.:-*•').strip()
            
            current_thought += " " + line
        
        if current_thought:
            thoughts.append(current_thought.strip())
        
        # Ensure we have the expected number of thoughts
        while len(thoughts) < expected_count:
            thoughts.append(f"Additional thought {len(thoughts) + 1}")
        
        return thoughts[:expected_count]
    
    def _parse_evaluation(self, response: str, criteria: List[str]) -> Dict[str, Any]:
        """Parse evaluation from response"""
        evaluation = {
            "score": 5.0,
            "criteria_scores": {},
            "strengths": [],
            "weaknesses": [],
            "potential": 5.0
        }
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip().lower()
            
            # Look for overall score
            if "overall score" in line or "total score" in line:
                score_match = self._extract_score(line)
                if score_match:
                    evaluation["score"] = score_match
            
            # Look for criteria scores
            for criterion in criteria:
                if criterion.lower() in line:
                    score_match = self._extract_score(line)
                    if score_match:
                        evaluation["criteria_scores"][criterion] = score_match
            
            # Look for strengths and weaknesses
            if "strength" in line:
                evaluation["strengths"].append(line)
            elif "weakness" in line:
                evaluation["weaknesses"].append(line)
        
        # Calculate average score if no overall score found
        if evaluation["criteria_scores"]:
            evaluation["score"] = statistics.mean(evaluation["criteria_scores"].values())
        
        return evaluation
    
    def _extract_score(self, text: str) -> Optional[float]:
        """Extract numerical score from text"""
        import re
        
        # Look for patterns like "8/10", "7.5", "8"
        patterns = [
            r'(\d+(?:\.\d+)?)/10',
            r'(\d+(?:\.\d+)?)\s*(?:out of 10|/10)?',
            r'score[:\s]*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                score = float(match.group(1))
                return min(10.0, max(1.0, score))  # Clamp between 1-10
        
        return None
    
    def _calculate_stats(self, tree: ThoughtTree) -> Dict[str, Any]:
        """Calculate exploration statistics"""
        total_thoughts = len(tree.thoughts)
        depths = [t.depth for t in tree.thoughts.values()]
        scores = [t.score for t in tree.thoughts.values() if t.score > 0]
        
        states = {}
        for thought in tree.thoughts.values():
            state = thought.state.value
            states[state] = states.get(state, 0) + 1
        
        return {
            "total_thoughts": total_thoughts,
            "max_depth": max(depths) if depths else 0,
            "average_score": statistics.mean(scores) if scores else 0,
            "state_distribution": states,
            "exploration_efficiency": len(tree.best_path) / total_thoughts if total_thoughts > 0 else 0
        }
    
    # Placeholder methods for other ToT applications
    async def _develop_creative_ideas(self, tree: ThoughtTree, depth: int) -> None:
        """Develop creative ideas further"""
        pass
    
    async def _evaluate_creative_ideas(self, tree: ThoughtTree, depth: int) -> None:
        """Evaluate creative ideas"""
        pass
    
    async def _select_best_creative_ideas(self, tree: ThoughtTree) -> List[Thought]:
        """Select best creative ideas"""
        return []
    
    async def _refine_creative_solutions(self, ideas: List[Thought], challenge: str) -> List[Dict[str, Any]]:
        """Refine creative solutions"""
        return []
    
    async def _initialize_strategic_tree(self, goal: str, constraints: List[str], time_horizon: str) -> ThoughtTree:
        """Initialize strategic planning tree"""
        return await self._initialize_tree(f"Strategic goal: {goal}")
    
    async def _explore_strategic_approaches(self, tree: ThoughtTree, max_strategies: int) -> None:
        """Explore strategic approaches"""
        pass
    
    async def _evaluate_strategic_approaches(self, tree: ThoughtTree, constraints: List[str]) -> None:
        """Evaluate strategic approaches"""
        pass
    
    async def _develop_strategic_plans(self, tree: ThoughtTree) -> None:
        """Develop detailed strategic plans"""
        pass
    
    async def _evaluate_strategic_plans(self, tree: ThoughtTree, constraints: List[str]) -> None:
        """Evaluate strategic plans"""
        pass
    
    async def _select_optimal_strategy(self, tree: ThoughtTree) -> Dict[str, Any]:
        """Select optimal strategy"""
        return {}
    
    async def _create_implementation_plan(self, strategy: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """Create implementation plan"""
        return {}
    
    async def _initialize_perspective_tree(self, topic: str, perspectives: List[str]) -> ThoughtTree:
        """Initialize multi-perspective analysis tree"""
        return await self._initialize_tree(f"Topic: {topic}")
    
    async def _analyze_perspectives(self, tree: ThoughtTree, depth: int) -> None:
        """Analyze from different perspectives"""
        pass
    
    async def _evaluate_perspective_insights(self, tree: ThoughtTree, depth: int) -> None:
        """Evaluate perspective insights"""
        pass
    
    async def _synthesize_perspectives(self, tree: ThoughtTree, topic: str) -> Dict[str, Any]:
        """Synthesize insights from multiple perspectives"""
        return {}
    
    async def _identify_perspective_conflicts(self, tree: ThoughtTree) -> List[Dict[str, Any]]:
        """Identify conflicts between perspectives"""
        return []
    
    async def _find_perspective_consensus(self, tree: ThoughtTree) -> Dict[str, Any]:
        """Find consensus across perspectives"""
        return {}
    
    async def _initialize_decision_tree(self, decision: str, options: List[str], criteria: List[str]) -> ThoughtTree:
        """Initialize decision-making tree"""
        return await self._initialize_tree(f"Decision: {decision}")
    
    async def _explore_decision_options(self, tree: ThoughtTree, criteria: List[str]) -> None:
        """Explore decision options"""
        pass
    
    async def _evaluate_decision_options(self, tree: ThoughtTree, criteria: List[str], weights: Optional[Dict[str, float]]) -> None:
        """Evaluate decision options"""
        pass
    
    async def _compare_options(self, tree: ThoughtTree, criteria: List[str], weights: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Compare decision options"""
        return {}
    
    async def _make_recommendation(self, tree: ThoughtTree, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Make final recommendation"""
        return {}