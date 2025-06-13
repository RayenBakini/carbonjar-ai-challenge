# Question 7 – Embedding Domain Knowledge into AI 

## Problem
AI predictions like emissions need to follow real-world rules. For example, if emissions exceed a legal cap, the system must respond.

## What I Did
- I wrote a post-processing function that:
  - Checks if the predicted emissions exceed a regulatory limit.
  - Flags it as a breach and caps it to the limit.
- This prevents the AI from suggesting risky or even illegal predictions.

## Output
A structured dictionary with:
- Final value (either original or capped)
- breach_flag (True/False)
- original_prediction (always shown)

## Other Ways to Enforce Constraints
- Feature engineering: add regulatory limits as inputs.
- Hybrid models: combine ML with rule-based systems.
- Pros: Safe, human-readable; Cons: More engineering work

## Conclusion
This method keeps the model flexible but adds real-world control. It's simple but effective.
