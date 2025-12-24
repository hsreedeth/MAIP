<div align = "center>
   <img height = "auto", width = "auto" src = "https://i.postimg.cc/pXk4FKnJ/Main-Repo-Cover.jpg" alt = "MAIP Repo Cover"/>
</div>

# MAIP: Multimorbidity-Aware ICU Phenotyping

This repository contains the full, auditable pipeline we use to derive interpretable ICU phenotypes from multimodal data, translate surrogate decision-tree rules into clinician-facing “rulecards,” and validate those translations end to end. The code is designed for repeatable analysis, deterministic prompt construction, and automated QC of every generated artifact.

## Key capabilities
- **Data preparation and clustering**: Build cleaned, multi-view feature sets and run Similarity Network Fusion (SNF) clustering to obtain multimorbidity strata.
- **Surrogate decision trees**: Fit shallow trees that approximate the SNF phenotypes, export JSON rulesets, and keep portable model bundles for audit.
- **RAG-backed rule translation**: Assemble retrieval-augmented prompts (style guide, phenotype summaries, and variable dictionary snippets) and call an LLM to produce markdown rulecards and ASCII flowcharts.
- **Deterministic validation**: Check textual rulecards against the JSON rules (feature/threshold coverage), enforce feature naming alignment with the variable dictionary, and optionally test synthetic profiles against the trained surrogate model.
- **Scriptable CLI**: A single command (`python -m src.cli build-rulecards`) rebuilds prompts, runs the remote LLM, and executes QC for selected strata.

## Repository layout (selected)
- `src/`: Pipeline code (preprocessing, clustering, surrogate tree fitting, prompt assembly, LLM invocation, validation).
- `rag_corpus/`: Style guide, phenotype summaries (`phenotype_*.md`), and `variable_dictionary.json` used for retrieval.
- `reports/`: Expected location for surrogate tree artifacts (`surrogate_<stratum>/`) and generated rulecards/validation outputs (`rulecards_rag/`, `rulecards_final/`).
- `data/`: User-provided processed feature tables (not included in the repository).
- `requirements.txt`: Python dependencies.

## Quick start
1) Create an environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2) Prepare inputs: place your processed feature tables under `data/01_processed/` and SNF assignment tables under `reports/snf_<stratum>/tables/snf_assignments.csv` (for strata such as `high`, `mid`, `low`).
3) Build rulecards (end-to-end) for all strata:
   ```bash
   python -m src.cli build-rulecards --strata High_MM Mid_MM Low_MM --out-root reports/rulecards_final
   ```
   This will:
   - Fit surrogate trees if `rule_ruleset.json` is missing for a stratum (uses `src.surrogate_tree`).
   - Assemble RAG prompts with `src.rag_translate_rules`.
   - Call the remote LLM via `src.run_rulecards_remote_canonical` (set `OPENAI_API_KEY` in your environment).
   - Run QC with `src.rulecard_validate`, writing CSVs alongside the rulecards.

## Core scripts
- `src.surrogate_tree`: Trains a decision-tree surrogate for a given stratum and exports `rule_ruleset.json` plus a `surrogate_tree.joblib` bundle.
- `src.rag_translate_rules`: Builds one prompt JSON per phenotype by merging rules, style guide text, phenotype summaries, and variable dictionary snippets.
- `src.run_rulecards_remote_canonical`: Calls a chat-completion model (default `gpt-4.1-mini`, temperature 0) to generate canonical rulecards and ASCII flowcharts with strict headings.
- `src.rulecard_validate`: Verifies coverage of every feature/threshold, alignment with the variable dictionary, and (optionally) agreement with the surrogate model using synthetic profiles.
- `src.cli`: Orchestrates the end-to-end rulecard build and QC workflow.

## Data expectations
- Processed feature matrices live under `data/01_processed/` (C/P/S views as referenced in `src.surrogate_tree`).
- SNF cluster assignments per stratum are expected under `reports/snf_<stratum>/tables/`.
- RAG context lives in `rag_corpus/` and can be edited to adjust style, summaries, or variable definitions.
No raw patient-level data is included in this repository; users must supply their own processed inputs.

## Notes on reproducibility
- LLM calls are run at near-zero temperature and with fixed token budgets to minimise variance.
- All intermediate artifacts (JSON rulesets, prompt payloads, rulecards, QC CSVs) are written as plain text for audit.
- Deterministic fallbacks (`src/cli_rulecard.py` and `src/flowchart_fallback.py`) can generate rulecards/flowcharts directly from JSON rules when running without an LLM.

## Contributing
Issues and pull requests are welcome. Please keep added dependencies minimal, maintain deterministic defaults, and document any changes to the RAG corpus or validation logic.
