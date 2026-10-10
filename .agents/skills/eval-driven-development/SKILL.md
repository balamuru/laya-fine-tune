---
name: eval-driven-development
description: >-
  Enforces Eval-Driven Development (EDD) methodology. Triggers on new projects or roadmap requests to mandate phased plans, Functional Requirements (FRs), and strict, executable Evaluations before declaring work complete.
---

# Eval-Driven Development (EDD) Methodology

## Overview
This skill enforces a rigorous, self-verifying workflow for complex coding tasks. It ensures that you (the agent) never declare a task "done" without first defining explicit completion criteria and physically executing tests to prove those criteria are met.

## Dependencies
None.

## Quick Start
Trigger this workflow automatically whenever the user asks to "build a project", "create a plan", or execute a multi-step coding task. 

## Workflow

### 1. Plan Generation
- Before writing application code, create a persistent Markdown artifact (e.g., `plan.md`).
- Break the user's request into distinct Phases.
- For each Phase, explicitly define 1-3 **Functional Requirements (FRs)**.
- For each FR, explicitly define an **Executable Eval** (a specific terminal command, script, or log check that will unambiguously prove the FR is met).

### 2. Phase Execution
- Write the code, configurations, or infrastructure required for the current phase.

### 3. Mandatory Verification (The Eval)
- You **MUST NOT** tell the user the phase is complete yet.
- You **MUST** execute the Evals defined in your plan using terminal commands (e.g., `pytest`, `docker logs`, `curl`, `cat`).
- Carefully observe the output of your Evals.

### 4. Self-Correction
- If an Eval fails or produces unexpected output, you must silently fix the code and re-run the Eval. Do not bother the user with trivial failures; fix them yourself.

### 5. Reporting & State Tracking
- Once all Evals for the phase pass successfully, update the Markdown plan artifact (e.g., check off `[x]` for the phase).
- Report the success to the user, briefly mentioning the Evals you ran to verify it.
- Ask the user for permission to proceed to the next phase.

## Common Mistakes
- **Assuming success without testing:** Declaring a phase "done" just because the code was written. You *must* run the Eval.
- **Vague Evals:** Writing an Eval like "Check if it works." Evals must be specific, executable commands (e.g., "Run `docker logs --tail 10` and grep for 'success'").
- **Skipping the Artifact:** Failing to write the plan down in an artifact, making it hard for the user to track progress.
