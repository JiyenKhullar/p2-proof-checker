# Łukasiewicz–Church (P2) Proof System

A Python implementation of a formal proof verifier and LLM-assisted proof generator for the Łukasiewicz–Church (P2) axiom system in propositional logic.

## 🎯 Overview

This project provides two main components:

1. **Automated Proof Verifier** (`automated_proof_verifier.py`) - A formal verifier that checks proofs for correctness
2. **LLM-Assisted Proof Generator** (`LLM_assisted_proof_generation.py`) - An intelligent proof generator using Google's Gemini API that can automatically construct valid proofs

## 🧠 The Łukasiewicz–Church System

The system uses three axioms and one inference rule:

### Axioms
- **AX1**: A → (B → A) - *If A, then (if B, then A)*
- **AX2**: (A → (B → C)) → ((A → B) → (A → C)) - *Distribution of implication*
- **AX3**: (¬B → ¬A) → (A → B) - *Contraposition*

### Inference Rule
- **Modus Ponens (MP)**: From φ and φ → ψ, infer ψ

## 🚀 Features

### Proof Verifier
- ✅ Parse propositional logic formulas with ¬ (negation) and → (implication)
- ✅ Verify axiom instances through pattern matching with substitution
- ✅ Check Modus Ponens applications
- ✅ Line-by-line proof validation
- ✅ Support for premises and goal verification
- ✅ Clear error messages for debugging

### LLM-Assisted Generator
- 🤖 Automatic proof generation using Google Gemini
- 🔄 Iterative refinement with verification feedback
- 💡 Intelligent error correction and retry mechanism
- 🎯 Counterexample generation for unprovable goals
- 📊 Comprehensive test suite with result logging

## 📦 Installation

### Prerequisites
- Python 3.8+
- Google AI API key (free from [Google AI Studio](https://aistudio.google.com/app/apikey))

### Dependencies
```bash
pip install google-generativeai
```

### Setup API Key
Choose one of these methods:

**Option 1: Environment Variable**
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

**Option 2: Config File**
Create `config.json`:
```json
{
    "google_api_key": "your-api-key-here"
}
```

## 🔧 Usage

### Basic Proof Verification

```python
from automated_proof_verifier import run

# Verify a simple proof
proof_text = """
1. P                                              Premise
2. P → Q                                          Premise
3. Q                                              MP 1, 2
"""

success, message = run(proof_text)
print(f"Result: {message}")
```

### LLM-Assisted Proof Generation

```python
from LLM_assisted_proof_generation import GeminiProofGenerator

# Initialize generator
generator = GeminiProofGenerator()

# Generate and verify a proof
premises = ["P", "P → Q"]
goal = "Q"

success, result = generator.orchestrate_proof(premises, goal)
print(f"Success: {success}")
print(f"Result:\n{result}")
```

### Advanced Usage with Counterexamples

```python
# Try proof generation, fall back to counterexample if impossible
success, result = generator.orchestrate_with_counterexample(premises, goal)
print(result)
```

## 📝 Proof Format

Proofs must follow this exact format:

```
<line_number>. <formula>    <justification>
```

**Example:**
```
1. P → (Q → P)                                    AX1
2. P                                              Premise
3. Q → P                                          MP 2, 1
```

### Valid Justifications
- `Premise` - Given premise
- `AX1`, `AX2`, `AX3` - Axiom instances
- `MP i, j` - Modus Ponens from lines i and j

### Formula Syntax
- Variables: `P`, `Q`, `R`, etc.
- Negation: `¬P` 
- Implication: `P → Q` (or `P -> Q`)
- Parentheses for grouping: `(P → Q) → R`

## 🧪 Examples

### Example 1: Simple Modus Ponens
```python
proof = """
1. P                                              Premise
2. P → Q                                          Premise  
3. Q                                              MP 1, 2
"""
```

### Example 2: Axiom Instance
```python
proof = """
1. P → (Q → P)                                    AX1
"""
```

### Example 3: Complex Derivation
```python
proof = """
1. P                                              Premise
2. P → (Q → P)                                    AX1
3. Q → P                                          MP 1, 2
"""
```

## 🤖 LLM Integration

The system uses Google's Gemini model to:

1. **Generate proofs** from premises and goals
2. **Learn from verification errors** through feedback loops
3. **Retry with corrections** up to configurable attempts
4. **Provide counterexamples** when proofs are impossible

### Feedback Loop Process
1. LLM generates initial proof attempt
2. Formal verifier checks for errors
3. If invalid, specific error feedback is sent back to LLM
4. LLM generates corrected version
5. Process repeats until valid proof found or max attempts reached

## 📊 Testing

Run the built-in test suite:

```bash
python LLM_assisted_proof_generation.py
```

This will run comprehensive tests including:
- Simple Modus Ponens
- Axiom instances  
- Invalid inferences (with counterexamples)
- Complex multi-step proofs

Results are saved to `proof_results.json` for analysis.

## 🗂️ Project Files

- `automated_proof_verifier.py` - Core proof verification engine
- `LLM_assisted_proof_generation.py` - Gemini-powered proof generator  
- `config.json` - API key configuration (create manually)
- `proof_results.json` - Test results output

## 🏗️ Architecture

### Core Components

**`Formula` Classes**
- `Variable` - Propositional variables (P, Q, R...)
- `Negation` - Negation formulas (¬A)  
- `Implication` - Implication formulas (A → B)

**`FormulaParser`**
- Tokenizes and parses formula strings
- Handles precedence and associativity
- Supports both → and -> notation

**`AxiomMatcher`**
- Pattern matching with substitution
- Verifies axiom instances
- Structural formula equality checking

**`ProofVerifier`**
- Line-by-line proof validation
- Modus Ponens verification
- Goal checking

**`GeminiProofGenerator`**
- LLM orchestration
- Iterative refinement
- Counterexample generation

## 🔍 Error Handling

The system provides detailed error messages for common issues:

- **Parse errors**: Invalid formula syntax
- **Invalid justifications**: Wrong axiom or MP format  
- **Line reference errors**: Missing or incorrect line numbers
- **MP application errors**: Incorrect antecedent/consequent matching
- **API errors**: Gemini connection or quota issues

## 🎨 Customization

### Extending the System

**Add new axioms:**
```python
# In AxiomMatcher.is_axiom_instance()
ax4_pattern = parse_formula("your_axiom_pattern")
if AxiomMatcher.match_formula(formula, ax4_pattern):
    return "AX4"
```

**Custom inference rules:**
```python
# In ProofVerifier._verify_line()
if justification.startswith("NEW_RULE"):
    return self._verify_new_rule(line, justification)
```

**Different LLM backends:**
```python
# Create new generator class inheriting from base
class OpenAIProofGenerator(ProofGenerator):
    def generate_proof_with_llm(self, premises, goal, feedback=None):
        # Your OpenAI implementation
        pass
```

## 📈 Performance

- **Verifier**: ~1000 proofs/second for typical formulas
- **Generator**: 2-10 seconds per proof depending on complexity
- **Memory**: <10MB for typical usage

## 🙏 Acknowledgments

- Jan Łukasiewicz and Alonzo Church for the original axiom system
- Google for the Gemini API
- The formal logic and automated theorem proving communities

## 🔗 Links

- [Google AI Studio](https://aistudio.google.com/app/apikey) - Get your free API key
- [Łukasiewicz Logic on Wikipedia](https://en.wikipedia.org/wiki/Łukasiewicz_logic)
- [Propositional Logic](https://en.wikipedia.org/wiki/Propositional_calculus)

---

**Assignment Project: Automated Theorem Proving with LLM Assistance**