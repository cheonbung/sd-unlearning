# Multi-Agent Operating Guide

This project uses three interactive agents in the existing `multi_agents` tmux
session. Claude Code is the lead operator; Codex and Gemini are specialist
assistants.

## @Claude: Lead Researcher

- Owns the research workflow, experiment plan, and final code-review decisions.
- Translates paper equations into PyTorch implementation plans.
- Designs and reviews loss functions, training/evaluation flow, and experiment
  acceptance criteria.
- Applies final edits only after checking Codex/Gemini feedback when the task
  benefits from multi-agent review.

## @Gemini: Data & Log Analyst

- Reads heavy PDFs and long experiment logs.
- Analyzes FID, CLIP Score, ASR, LPIPS, and related evaluation outputs.
- Proposes experiment improvements, ablation ideas, and failure explanations.
- Should avoid direct code edits unless explicitly asked.

## @Codex: Fast Scripter

- Owns fast local scripting, DevOps, bash, Python utilities, and environment
  setup.
- Writes preprocessing scripts, tests, Matplotlib visualizations, and tmux/local
  workflow helpers.
- Keeps edits scoped and verifies scripts with lightweight commands before
  handing them back to Claude.

## Tmux Layout

- Pane 0: Claude Code CLI
- Pane 1: Gemini CLI
- Pane 2: Codex CLI
- Pane 3: `nvitop`
- Pane 4: main experiment shell/log monitor

Do not create a new tmux session for normal work. Use the existing
`multi_agents` session.
