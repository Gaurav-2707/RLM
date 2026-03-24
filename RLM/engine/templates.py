"""
Prompt templates for the RLMEngine's 3-phase reasoning process.
"""

STEP_0_DECOMPOSITION = """
# MISSION: REASONING ENGINE STEP 0 (DECOMPOSITION)
# ANCHOR: ORIGINAL PROBLEM
{problem}

TASK:
Decompose the original problem into atomic logical sub-units and sub-categories. 
Identify the core components and potential data points needed to solve the problem.

FORMAT:
- Decomposition: [Detailed list of logical blocks]
- Sub-problems: [List of specific questions to answer]
- Reasoning Path: [High-level strategy for solving the problem]

<summary>
[Provide a ~150-token structured summary of this decomposition for the next stage.]
</summary>

REMINDER: Your primary goal is to solve the ORIGINAL PROBLEM. Ensure every sub-unit is directly relevant.
"""

STEP_1_REFINEMENT = """
# MISSION: REASONING ENGINE STEP 1 (REFINEMENT & ASSUMPTIONS)
# ANCHOR: ORIGINAL PROBLEM
{problem}

# CONTEXT: PREVIOUS STEP SUMMARY (DECOMPOSITION)
{prev_summary}

TASK:
1. Identify hidden assumptions and constraints in the previous decomposition.
2. Refine the reasoning path and core hypothesis.
3. CRITIQUE: Analyze Step 0's output. Point out any errors, omissions, or misinterpretations of the original problem. Correct them here.

FORMAT:
- Critique of Step 0: [Detailed self-correction]
- Identified Assumptions: [List of implicit or explicit assumptions]
- Refined Hypothesis: [Updated strategy based on constraints]

<summary>
[Provide a ~150-token structured summary of this refinement for the next stage.]
</summary>

REMINDER: Stay anchored to the ORIGINAL PROBLEM. Do not let the refinement phase drift away from the core objective.
"""

STEP_2_SYNTHESIS = """
# MISSION: REASONING ENGINE STEP 2 (SYNTHESIS & ACTION)
# ANCHOR: ORIGINAL PROBLEM
{problem}

# CONTEXT: PREVIOUS STEP SUMMARY (REFINEMENT)
{prev_summary}

TASK:
1. Synthesize the final comprehensive answer based on the refined hypothesis and logic.
2. Propose the immediate next action or decision.
3. CRITIQUE: Analyze Step 1's output. Point out any flaws in the assumptions or hypothesis and fix them in your final synthesis.

FORMAT:
- Critique of Step 1: [Final self-correction]
- Final Answer: [The comprehensive response to the original problem]
- Proposed Action: [The specific next step or decision]

REMINDER: The final answer must be the most accurate and logically sound resolution to the ORIGINAL PROBLEM.
"""
