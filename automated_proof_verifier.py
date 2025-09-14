"""
Proof Verifier for Łukasiewicz–Church (P2) Axiom System
Implements propositional logic proof verification with:
- Axioms: AX1, AX2, AX3
- Inference rule: Modus Ponens
- Formula parsing with ¬ and → connectives
"""

import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class Formula:
    """Abstract base class for propositional formulas"""
    pass


@dataclass
class Variable(Formula):
    """Propositional variable (P, Q, R, ...)"""
    name: str
    
    def __str__(self):
        return self.name


@dataclass
class Negation(Formula):
    """Negation formula (¬A)"""
    operand: Formula
    
    def __str__(self):
        return f"¬{self.operand}"


@dataclass
class Implication(Formula):
    """Implication formula (A → B)"""
    antecedent: Formula
    consequent: Formula
    
    def __str__(self):
        return f"({self.antecedent} → {self.consequent})"


class FormulaParser:
    """Parser for propositional logic formulas with ¬ and → connectives"""
    
    def __init__(self, formula_str: str):
        formula_str = formula_str.replace("->", "→")
        self.tokens = self._tokenize(formula_str)
        self.pos = 0
    
    def _tokenize(self, formula_str: str) -> List[str]:
        """Tokenize the formula string into components"""
        # Remove whitespace and split into tokens
        formula_str = re.sub(r'\s+', '', formula_str)
        tokens = []
        i = 0
        while i < len(formula_str):
            char = formula_str[i]
            if char in '()¬→':
                tokens.append(char)
            elif char.isalpha():
                # Variable name (can be multiple characters)
                var_name = ''
                while i < len(formula_str) and formula_str[i].isalnum():
                    var_name += formula_str[i]
                    i += 1
                tokens.append(var_name)
                continue
            i += 1
        return tokens
    
    def parse(self) -> Formula:
        """Parse the tokenized formula"""
        return self._parse_implication()
    
    def _parse_implication(self) -> Formula:
        """Parse implication (right-associative)"""
        left = self._parse_negation()
        
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '→':
            self.pos += 1
            right = self._parse_implication()  # Right-associative
            return Implication(left, right)
        
        return left
    
    def _parse_negation(self) -> Formula:
        """Parse negation"""
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '¬':
            self.pos += 1
            operand = self._parse_negation()  # Handle multiple negations
            return Negation(operand)
        
        return self._parse_primary()
    
    def _parse_primary(self) -> Formula:
        """Parse primary expressions (variables and parenthesized expressions)"""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of formula")
        
        token = self.tokens[self.pos]
        
        if token == '(':
            self.pos += 1
            formula = self._parse_implication()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Missing closing parenthesis")
            self.pos += 1
            return formula
        elif token.isalpha():
            self.pos += 1
            return Variable(token)
        else:
            raise ValueError(f"Unexpected token: {token}")


def parse_formula(formula_str: str) -> Formula:
    """Parse a formula string into a Formula object"""
    try:
        parser = FormulaParser(formula_str)
        return parser.parse()
    except Exception as e:
        raise ValueError(f"Failed to parse formula '{formula_str}': {e}")


class AxiomMatcher:
    """Matches formulas against axiom schemas with substitution"""
    
    @staticmethod
    def match_formula(formula: Formula, pattern: Formula, substitution: Dict[str, Formula] = None) -> Optional[Dict[str, Formula]]:
        """
        Try to match a formula against a pattern, returning substitution if successful
        """
        if substitution is None:
            substitution = {}
        
        if isinstance(pattern, Variable):
            # Pattern variable - check if we can substitute
            if pattern.name in substitution:
                # Variable already has a substitution - check if it matches
                return substitution if AxiomMatcher._formulas_equal(formula, substitution[pattern.name]) else None
            else:
                # New substitution
                substitution[pattern.name] = formula
                return substitution
        
        elif isinstance(pattern, Negation) and isinstance(formula, Negation):
            return AxiomMatcher.match_formula(formula.operand, pattern.operand, substitution)
        
        elif isinstance(pattern, Implication) and isinstance(formula, Implication):
            # Match antecedent first
            sub1 = AxiomMatcher.match_formula(formula.antecedent, pattern.antecedent, substitution.copy())
            if sub1 is None:
                return None
            # Then match consequent with the updated substitution
            return AxiomMatcher.match_formula(formula.consequent, pattern.consequent, sub1)
        
        else:
            return None
    
    @staticmethod
    def _formulas_equal(f1: Formula, f2: Formula) -> bool:
        """Check if two formulas are structurally equal"""
        if type(f1) != type(f2):
            return False
        
        if isinstance(f1, Variable):
            return f1.name == f2.name
        elif isinstance(f1, Negation):
            return AxiomMatcher._formulas_equal(f1.operand, f2.operand)
        elif isinstance(f1, Implication):
            return (AxiomMatcher._formulas_equal(f1.antecedent, f2.antecedent) and 
                   AxiomMatcher._formulas_equal(f1.consequent, f2.consequent))
        
        return False
    
    @staticmethod
    def is_axiom_instance(formula: Formula) -> Optional[str]:
        """Check if formula is an instance of AX1, AX2, or AX3"""
        # AX1: A → (B → A)
        ax1_pattern = parse_formula("A → (B → A)")
        if AxiomMatcher.match_formula(formula, ax1_pattern):
            return "AX1"
        
        # AX2: (A → (B → C)) → ((A → B) → (A → C))
        ax2_pattern = parse_formula("(A → (B → C)) → ((A → B) → (A → C))")
        if AxiomMatcher.match_formula(formula, ax2_pattern):
            return "AX2"
        
        # AX3: (¬B → ¬A) → (A → B)
        ax3_pattern = parse_formula("(¬B → ¬A) → (A → B)")
        if AxiomMatcher.match_formula(formula, ax3_pattern):
            return "AX3"
        
        return None


@dataclass
class ProofLine:
    """Represents a line in a proof"""
    line_number: int
    formula: Formula
    justification: str


class ProofVerifier:
    """Verifies proofs in the Łukasiewicz–Church (P2) system"""
    
    def __init__(self):
        self.proven_formulas: Dict[int, Formula] = {}
    
    def verify_proof(self, proof_lines: List[ProofLine], goal: Formula = None) -> Tuple[bool, str]:
        """
        Verify a proof line by line
        Returns (success, message)
        """
        self.proven_formulas.clear()
        
        for line in proof_lines:
            success, message = self._verify_line(line)
            if not success:
                return False, f"Line {line.line_number}: {message}"
            
            self.proven_formulas[line.line_number] = line.formula
        
        # If goal is specified, check if it was derived
        if goal is not None:
            last_formula = proof_lines[-1].formula if proof_lines else None
            if last_formula is None or not AxiomMatcher._formulas_equal(last_formula, goal):
                return False, "Goal formula not reached"
        
        return True, "VALID proof"
    
    def _verify_line(self, line: ProofLine) -> Tuple[bool, str]:
        """Verify a single proof line"""
        justification = line.justification.strip()
        
        # Check if it's a premise
        if justification == "Premise":
            return True, "Premise accepted"
        
        # Check if it's an axiom instance
        axiom = AxiomMatcher.is_axiom_instance(line.formula)
        if axiom and justification == axiom:
            return True, f"Valid {axiom} instance"
        
        # Check if it's Modus Ponens
        if justification.startswith("MP"):
            return self._verify_modus_ponens(line, justification)
        
        # Check for substitution (if implemented)
        if justification.startswith("Substitution"):
            return self._verify_substitution(line, justification)
        
        return False, f"Invalid justification: {justification}"
    
    def _verify_modus_ponens(self, line: ProofLine, justification: str) -> Tuple[bool, str]:
        """Verify Modus Ponens application"""
        # Parse "MP i, j" format
        try:
            # Remove "MP" and get the rest
            if not justification.startswith("MP"):
                return False, "Invalid MP format"
            
            # Extract numbers after MP
            numbers_part = justification[2:].strip()
            line_refs = numbers_part.split(',')
            if len(line_refs) != 2:
                return False, "Invalid MP format (should be 'MP i, j')"
            
            line_i = int(line_refs[0].strip())
            line_j = int(line_refs[1].strip())
            
        except ValueError:
            return False, "Invalid line numbers in MP justification"
        
        # Check if referenced lines exist and were proven
        if line_i not in self.proven_formulas:
            return False, f"Line {line_i} not found or not proven"
        
        if line_j not in self.proven_formulas:
            return False, f"Line {line_j} not found or not proven"
        
        # Get the formulas from the referenced lines
        formula_i = self.proven_formulas[line_i]
        formula_j = self.proven_formulas[line_j]
        
        # Check MP: from φ and φ → ψ, infer ψ
        # Try both orders: (i, j) and (j, i)
        if self._check_mp_application(formula_i, formula_j, line.formula):
            return True, f"Valid MP from lines {line_i}, {line_j}"
        elif self._check_mp_application(formula_j, formula_i, line.formula):
            return True, f"Valid MP from lines {line_j}, {line_i}"
        else:
            return False, f"MP does not apply to lines {line_i}, {line_j}"
    
    def _check_mp_application(self, phi: Formula, phi_implies_psi: Formula, psi: Formula) -> bool:
        """Check if MP correctly applies: from φ and φ → ψ, we get ψ"""
        if not isinstance(phi_implies_psi, Implication):
            return False
        
        # Check if the antecedent of the implication matches φ
        if not AxiomMatcher._formulas_equal(phi, phi_implies_psi.antecedent):
            return False
        
        # Check if the consequent of the implication matches ψ
        return AxiomMatcher._formulas_equal(psi, phi_implies_psi.consequent)
    
    def _verify_substitution(self, line: ProofLine, justification: str) -> Tuple[bool, str]:
        """Verify substitution (placeholder - could be extended)"""
        return False, "Substitution verification not implemented"


def parse_proof_from_text(proof_text: str) -> List[ProofLine]:
    """Parse proof from text format"""
    lines = []
    for line_text in proof_text.strip().split('\n'):
        line_text = line_text.strip()
        if not line_text:
            continue
        
        # Parse line format: "1. P → Q    Premise"
        # Use regex to properly extract parts
        match = re.match(r'(\d+)\.\s*(.+?)\s{2,}(.+)$', line_text)
        if not match:
            raise ValueError(f"Invalid line format: {line_text}")
        
        line_number = int(match.group(1))
        formula_str = match.group(2).strip()
        justification = match.group(3).strip()
        
        formula = parse_formula(formula_str)
        lines.append(ProofLine(line_number, formula, justification))
    
    return lines

def run(proof_text):
    proof_lines = parse_proof_from_text(proof_text)
    verifier = ProofVerifier()
    success, message = verifier.verify_proof(proof_lines)
    return success, message

# Example usage and test cases
def Examples():
    """Example usage of the proof verifier"""
    
    # Example 1: Simple proof using AX1
    proof_text1 = """
    1. P → (Q → P)                                    AX1
    """
    
    print("Example 1: AX1 instance")
    try:
        success, message = run(proof_text1)
        print(f"Result: {message}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Modus Ponens application
    proof_text2 = """
    1. P                                              Premise
    2. P → Q                                          Premise
    3. Q                                              MP 1, 2
    """
    
    print("\nExample 2: Modus Ponens")
    try:
        success, message = run(proof_text2)
        print(f"Result: {message}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Invalid MP application
    proof_text3 = """
    1. P                                              Premise
    2. Q → R                                          Premise
    3. R                                              MP 1, 2
    """
    
    print("\nExample 3: Invalid MP")
    try:
        success, message = run(proof_text3)
        print(f"Result: {message}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 4: AX2 instance
    proof_text4 = """
    1. (P → (Q → R)) → ((P → Q) → (P → R))           AX2
    """
    
    print("\nExample 4: AX2 instance")
    try:
        success, message = run(proof_text4)
        print(f"Result: {message}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 5: AX3 instance  
    proof_text5 = """
    1. (¬Q → ¬P) → (P → Q)                           AX3
    """
    
    print("\nExample 5: AX3 instance")
    try:
        success, message = run(proof_text5)
        print(f"Result: {message}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 6: More complex proof
    proof_text6 = """
    1. P                                              Premise
    2. P → (Q → P)                                    AX1
    3. Q → P                                          MP 1, 2
    """
    
    print("\nExample 6: Complex proof with AX1 and MP")
    try:
        success, message = run(proof_text6)
        print(f"Result: {message}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 7: More complex proof with ->
    proof_text7 = """
    1. P                                              Premise
    2. P -> (Q → P)                                    AX1
    3. Q -> P                                          MP 1, 2
    """
    
    print("\nExample 7: Complex proof with AX1 and MP using ->")
    try:
        success, message = run(proof_text6)
        print(f"Result: {message}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    Examples()