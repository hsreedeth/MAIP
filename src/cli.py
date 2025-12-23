# src/cli.py
from pathlib import Path
import subprocess
import typer
from src.make_views import run_preprocessing
from src.clean_data import clean_file
from src.run_mmsp_phase1_pam import run as run_mmsp
from src.external_validation import main as run_validate

app = typer.Typer()
ROOT = Path(__file__).resolve().parents[1]

def _run(cmd: list[str]) -> None:
    """Run a subprocess, echoing the command for transparency."""
    typer.echo(f"[CLI] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

@app.command("build-rulecards")
def build_rulecards(
    strata: list[str] = typer.Option(
        ["High_MM", "Mid_MM", "Low_MM"],
        "--strata",
        "-s",
        help="Which SNF strata to process (e.g. High_MM Mid_MM Low_MM).",
    ),
    out_root: Path = typer.Option(
        Path("reports/rulecards_final"),
        "--out-root",
        help="Root directory for canonical rulecard outputs and QC tables.",
    ),
):
    """
    End-to-end pipeline for MAIP rulecards:

    1. Ensure surrogate trees exist for the requested strata.
    2. Build RAG prompts (JSON + variable dictionary + phenotype summaries + style guide).
    3. Call the remote LLM to produce canonical rulecards.
    4. Run QC checks on coverage, thresholds and feature naming.
    """

    project_root = Path(__file__).resolve().parents[1]  # repo root (.. from src/cli.py)

    # --- 1. Ensure surrogate trees exist for each stratum --------------------
    for label in strata:
        # Expect labels like "High_MM", "Mid_MM", "Low_MM"
        base = label.split("_")[0].lower()  # "High_MM" -> "high"
        surrogate_dir = project_root / "reports" / f"surrogate_{base}"
        rule_json = surrogate_dir / "tables" / "rule_ruleset.json"

        if rule_json.exists():
            typer.echo(f"[CLI] Surrogate rules already present for {label} ({rule_json}).")
        else:
            typer.echo(f"[CLI] Surrogate rules missing for {label}; fitting surrogate tree.")
            snf_assign = project_root / "reports" / f"snf_{base}" / "tables" / "snf_assignments.csv"

            _run(
                [
                    "python",
                    "-m",
                    "src.surrogate_tree",
                    "--cview",
                    "data/01_processed/C_view.csv",
                    "--pview",
                    "data/01_processed/P_view_scaled.csv",
                    "--sview",
                    "data/01_processed/S_view.csv",
                    "--snf-assign",
                    str(snf_assign),
                    "--stratum",
                    label,
                    "--out-dir",
                    str(surrogate_dir),
                ]
            )

        if not rule_json.exists():
            raise RuntimeError(f"Expected rule_ruleset.json not found for {label}: {rule_json}")

    # --- 2. Rebuild RAG prompts for each stratum ----------------------------
    prompt_root = project_root / "reports" / "rulecards_rag"
    variable_dict = project_root / "rag_corpus" / "variable_dictionary.json"
    style_guide = project_root / "rag_corpus" / "style_guide.md"
    phenotype_dir = project_root / "rag_corpus"

    for label in strata:
        base = label.split("_")[0].lower()
        surrogate_dir = project_root / "reports" / f"surrogate_{base}"
        rule_json = surrogate_dir / "tables" / "rule_ruleset.json"
        out_dir = prompt_root / base

        typer.echo(f"[CLI] Building RAG prompts for {label} → {out_dir}")
        _run(
            [
                "python",
                "-m",
                "src.rag_translate_rules",
                "--rules-json",
                str(rule_json),
                "--variable-dict",
                str(variable_dict),
                "--style-guide",
                str(style_guide),
                "--phenotype-dir",
                str(phenotype_dir),
                "--out-dir",
                str(out_dir),
            ]
        )

    # 3. Canonical remote rulecard generation 
    # This is built for `src.run_rulecards_remote_canonical` and accepts:
    #   --in-root reports/rulecards_rag
    #   --out-root reports/rulecards_final
    # and writes rulecards into e.g. <out-root>/<high|mid|low>/rulecard_*.md
    typer.echo("[CLI] Invoking remote LLM to generate canonical rulecards.")
    _run(
        [
            "python",
            "-m",
            "src.run_rulecards_remote_canonical",
            "--in-root",
            str(prompt_root),
            "--out-root",
            str(out_root),
        ]
    )

    # --- 4. QC: coverage, thresholds, feature alignment, synthetic checks ----
    for label in strata:
            base = label.split("_")[0].lower()
            surrogate_dir  = project_root / "reports" / f"surrogate_{base}"
            rules_json     = surrogate_dir / "tables" / "rule_ruleset.json"
            model_bundle   = surrogate_dir / "models" / "surrogate_tree.joblib"
            rulecards_dir  = out_root / base
            out_dir_stratum = out_root / base  # write CSVs alongside rulecards

            out_dir_stratum.mkdir(parents=True, exist_ok=True)

            typer.echo(f"[CLI] Running rulecard validation for {label}.")

            # This matches the CLI signature in src/rulecard_validate.py
            cmd = [
                "python",
                "-m",
                "src.rulecard_validate",
                "--stratum",
                label,
                "--rules-json",
                str(rules_json),
                "--rulecards-dir",
                str(rulecards_dir),
                "--var-dict",
                str(variable_dict),
                "--out-dir",
                str(out_dir_stratum),
            ]

            # Attach model bundle if it exists (enables synthetic profile checks)
            if model_bundle.exists():
                cmd.extend(["--model-bundle", str(model_bundle)])

            _run(cmd)

    typer.echo("[CLI] build-rulecards completed successfully.")

@app.command()
def preprocess(no_save: bool = False, strict_row_drop: bool = False):
    """Build support_preprocessed + views (+ scaler/imputer)."""
    run_preprocessing(save_output=not no_save, strict_row_drop=strict_row_drop)

@app.command()
def curate(no_update_views: bool = False, reimpute_alb: bool = False, refit_scaler: bool = False):
    """Light cleaning on preprocessed -> support_preprocessed_clean."""
    clean_file(
        update_views=not no_update_views,
        reimpute_alb=reimpute_alb,
        refit_scaler=refit_scaler
    )

@app.command()
def mmsp():
    """Run MMSP Phase-1 PAM and save clusters."""
    run_mmsp()

@app.command()
def validate():
    """External validation: KM, Cox, Kruskal; figures+tables."""
    run_validate()

if __name__ == "__main__":
    app()

# sriptable/auditable CLI pipeline. (one-liner pipeline)

#   python -m src.cli build-rulecards \
#   --strata High_MM Mid_MM Low_MM \
#   --out-root reports/rulecards_final

# test only one stratum with:
#   python -m src.cli build-rulecards \
#   --strata High_MM \
#   --out-root reports/rulecards_final_high_only


# Under the hood, this command can simply:
    # Call surrogate_tree.py if the surrogate artifacts are missing.
    # Call rag_translate_rules.py for each stratum.
    # Call run_rulecards_remote.py for each stratum
    # Call rulecard_validate.py and write QC tables.