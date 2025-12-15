# ---- Paths ----
RAW = data/00_raw/support2.csv
PROC = data/01_processed
CLUS = data/02_clusters
MODELS = models/transformers

# ---- Default ----
.PHONY: all
all: clusters validate

# ---- Step 1: Preprocess (single source of truth) ----
$(PROC)/support_preprocessed.csv: $(RAW) src/make_views.py support_schema.yml
	python -m src.cli preprocess

# Views depend on preprocessed
$(PROC)/C_view.csv $(PROC)/P_view.csv $(PROC)/S_view.csv $(PROC)/P_view_scaled.csv: $(PROC)/support_preprocessed.csv
	@# created by preprocess target above

# Optional curated copy (does not rewrite base)
$(PROC)/support_preprocessed_clean.csv: $(PROC)/support_preprocessed.csv src/clean_data.py $(MODELS)/scaler_P.joblib
	python -m src.cli curate

# Y for validation (written once inside preprocess)
$(PROC)/Y_validation.csv: $(PROC)/support_preprocessed.csv
	@# produced in preprocess; nothing to do

# ---- Step 2: MMSP ----
$(CLUS)/mmsp_clusters.csv: $(PROC)/P_view_scaled.csv src/run_mmsp_phase1_pam.py
	python -m src.cli mmsp

# ---- Step 3: Validation ----
.PHONY: validate
validate: $(CLUS)/mmsp_clusters.csv $(PROC)/Y_validation.csv
	python -m src.cli validate

# ---- Notebooks read-only ----
.PHONY: notebooks
notebooks: $(PROC)/support_preprocessed.csv
	@echo "Notebooks should import data only; no writes."

# ---- Hygiene ----
.PHONY: clean
clean:
	rm -rf $(CLUS)/* reports/figures/* reports/tables/*
