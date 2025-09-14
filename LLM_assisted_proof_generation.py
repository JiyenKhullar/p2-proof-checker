"""
LLM-Assisted Proof Generator for Łukasiewicz–Church (P2) Axiom System
Orchestrates between Gemini LLM proof generation and formal verification
"""

import os
import json
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import google.generativeai as genai

# Import from the proof verifier (assuming it's in the same directory)
from automated_proof_verifier import (
    Formula, Variable, Negation, Implication,
    parse_formula, ProofLine, ProofVerifier, 
    parse_proof_from_text, AxiomMatcher
)


class GeminiProofGenerator:
    """Gemini-assisted proof generator using Google AI API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        """
        Initialize the Gemini proof generator
        
        Args:
            api_key: Google AI API key (if None, loads from environment)
            model: Gemini model to use (flash is fastest and cheapest)
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google AI API key not found. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.verifier = ProofVerifier()
    
    def generate_proof_with_llm(self, premises: List[str], goal: str, feedback: Optional[str] = None) -> str:
        """
        Generate a proof using Gemini
        
        Args:
            premises: List of premise formulas as strings
            goal: Goal formula as string
            feedback: Optional feedback from previous verification attempt
            
        Returns:
            Proof text in verifier format
        """
        # Construct the prompt
        premise_text = ", ".join(premises) if premises else "None"
        
        system_prompt = """You are a formal logic proof generator for the Łukasiewicz–Church (P2) axiom system.

AXIOMS:
- AX1: A → (B → A)  [If A, then (if B, then A)]
- AX2: (A → (B → C)) → ((A → B) → (A → C))  [Distribution of implication]
- AX3: (¬B → ¬A) → (A → B)  [Contraposition]

INFERENCE RULE:
- Modus Ponens (MP): From φ and φ → ψ, infer ψ

IMPORTANT FORMATTING:
- Generate proofs in this EXACT format:
  1. <formula>    <justification>
  2. <formula>    <justification>
  ...

- JUSTIFICATIONS must be one of:
  * "Premise" for given premises
  * "AX1", "AX2", or "AX3" for axiom instances  
  * "MP i, j" for Modus Ponens from lines i and j (use exact format with comma and space)

- Use → for implication and ¬ for negation
- Be precise with parentheses to match axiom patterns exactly
- Each line must have exactly two or more spaces between formula and justification"""

        user_prompt = f"""Generate a formal proof:

Premises: {premise_text}
Goal: {goal}

Derive the goal from the premises using only the three axioms and Modus Ponens. 
Make sure each step follows the exact format and uses valid justifications."""

        if feedback:
            user_prompt += f"\n\nPREVIOUS ATTEMPT FAILED:\n{feedback}\n\nPlease fix the errors and generate a corrected proof."

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1500,
                    temperature=0.1  # Low temperature for consistency
                )
            )
            
            # Extract just the proof text from Gemini's response
            content = response.text.strip()
            
            # Try to extract just the numbered proof lines
            proof_lines = []
            for line in content.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.', line):  # Lines starting with number and dot
                    proof_lines.append(line)
            
            if proof_lines:
                return '\n'.join(proof_lines)
            else:
                # If no numbered lines found, return the whole content
                return content
            
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")
    
    def orchestrate_proof(self, premises: List[str], goal: str, max_attempts: int = 5) -> Tuple[bool, str]:
        """
        Orchestrate proof generation with verification feedback loop
        
        Args:
            premises: List of premise formulas as strings
            goal: Goal formula as string  
            max_attempts: Maximum number of Gemini attempts
            
        Returns:
            (success, result_message)
        """
        feedback = None
        last_proof = None
        
        # Parse goal formula for verification
        try:
            goal_formula = parse_formula(goal)
        except Exception as e:
            return False, f"Invalid goal formula '{goal}': {e}"
        
        for attempt in range(max_attempts):
            try:
                print(f"Attempt {attempt + 1}/{max_attempts}...")
                
                # Generate proof with Gemini
                proof_text = self.generate_proof_with_llm(premises, goal, feedback)
                last_proof = proof_text
                
                print("Generated proof:")
                print(proof_text)
                print()
                
                # Parse and verify the proof
                proof_lines = parse_proof_from_text(proof_text)
                success, message = self.verifier.verify_proof(proof_lines, goal_formula)
                
                if success:
                    return True, f"VALID proof found in {attempt + 1} attempts:\n\n{proof_text}"
                else:
                    feedback = f"Verifier error: {message}\n\nMake sure to:\n- Use exact justification format ('MP i, j' not 'MP i,j')\n- Match axiom patterns precisely\n- Reference correct line numbers"
                    print(f"Verification failed: {message}")
                    print()
                    
            except Exception as e:
                feedback = f"Error parsing proof: {e}\n\nPlease generate proof in the exact format:\n1. <formula>    <justification>\n2. <formula>    <justification>"
                print(f"Parsing error: {e}")
                print()
        
        return False, f"FAILED after {max_attempts} attempts. Last error: {feedback}\n\nLast proof attempt:\n{last_proof}"
    
    def generate_counterexample(self, premises: List[str], goal: str) -> str:
        """
        Generate a counterexample when no proof exists
        
        Args:
            premises: List of premise formulas as strings
            goal: Goal formula as string
            
        Returns:
            Counterexample explanation
        """
        premise_text = ", ".join(premises) if premises else "None"
        
        prompt = f"""The following cannot be proven in propositional logic:

Premises: {premise_text}
Goal: {goal}

Generate a counterexample by providing a truth value assignment that makes all premises true but the goal false. 

Format your response as:
1. Truth assignment (e.g., P=True, Q=False, R=True)
2. Verification that premises are all true under this assignment
3. Verification that goal is false under this assignment  
4. Brief explanation of why this shows the goal doesn't follow from premises"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=800,
                    temperature=0.3
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            return f"Could not generate counterexample: {e}"
    
    def orchestrate_with_counterexample(self, premises: List[str], goal: str, max_attempts: int = 5) -> Tuple[bool, str]:
        """
        Orchestrate proof generation, falling back to counterexample if proof fails
        
        Args:
            premises: List of premise formulas as strings
            goal: Goal formula as string
            max_attempts: Maximum number of Gemini attempts for proof
            
        Returns:
            (success, result_message)
        """
        success, result = self.orchestrate_proof(premises, goal, max_attempts)
        
        if success:
            return success, result
        else:
            # Generate counterexample
            print("Proof generation failed. Attempting to generate counterexample...")
            counterexample = self.generate_counterexample(premises, goal)
            return False, f"{result}\n\n--- COUNTEREXAMPLE ---\n{counterexample}"


def load_api_key_from_file(filepath: str = "config.json") -> Optional[str]:
    """
    Load API key from a JSON config file
    
    Args:
        filepath: Path to config file
        
    Returns:
        API key or None if not found
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
            return config.get('google_api_key')
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def main():
    """Example usage and test cases"""
    
    # Load API key (prioritize environment, then config file)
    api_key = os.getenv('GOOGLE_API_KEY') or load_api_key_from_file()
    
    if not api_key:
        print("ERROR: No Google AI API key found!")
        print("Get a free API key at: https://aistudio.google.com/app/apikey")
        print("Then either:")
        print("1. Set GOOGLE_API_KEY environment variable, OR")
        print('2. Create config.json with: {"google_api_key": "your-api-key-here"}')
        return
    
    # Initialize generator
    generator = GeminiProofGenerator(api_key=api_key)
    
    # Test cases
    test_cases = [
        {
            "name": "Simple Modus Ponens",
            "premises": ["P", "P → Q"],
            "goal": "Q",
            "should_succeed": True
        },
        {
            "name": "Identity via AX1",
            "premises": ["P"],
            "goal": "Q → P",
            "should_succeed": True
        },
        {
            "name": "AX1 instance (no premises)",
            "premises": [],
            "goal": "P → (Q → P)",
            "should_succeed": True
        },
        {
            "name": "Invalid inference",
            "premises": ["P"],
            "goal": "Q",
            "should_succeed": False
        },
        {
            "name": "Complex proof with AX2",
            "premises": ["P → (Q → R)", "P → Q", "P"],
            "goal": "R",
            "should_succeed": True
        }
    ]
    import json

    output_file="proof_results.json"
    
    print("=== Gemini-Assisted Proof Generator Tests ===\n")
    
    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print(f"Premises: {test_case['premises']}")
        print(f"Goal: {test_case['goal']}")
        print(f"Expected: {'SUCCESS' if test_case['should_succeed'] else 'FAILURE'}")
        print("-" * 50)
        
        try:
            success, result = generator.orchestrate_with_counterexample(
                test_case['premises'], 
                test_case['goal'], 
                max_attempts=3
            )
            
            status = "✓ SUCCESS" if success else "✗ FAILURE"
            print(f"Result: {status}")
            print(f"Details: {result}")
            
            results.append({
                "test_case": test_case['name'],
                "premises": test_case['premises'],
                "goal": test_case['goal'],
                "expected": "SUCCESS" if test_case['should_succeed'] else "FAILURE",
                "status": "SUCCESS" if success else "FAILURE",
                "details": result
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "test_case": test_case['name'],
                "premises": test_case['premises'],
                "goal": test_case['goal'],
                "expected": "SUCCESS" if test_case['should_succeed'] else "FAILURE",
                "status": "ERROR",
                "details": str(e)
            })
        
        print("\n" + "="*80 + "\n")
    
    # Save results to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()