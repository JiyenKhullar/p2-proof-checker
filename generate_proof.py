from LLM_assisted_proof_generation import ClaudeProofGenerator

generator = ClaudeProofGenerator()
success, result = generator.orchestrate_proof(
    premises=["P", "P → Q"],
    goal="Q"
)
print(result)