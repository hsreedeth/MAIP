suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(purrr)
  library(stringr)
  library(survival)
  library(broom)
  library(ggplot2)
  library(forcats)
})

#----------------------------
# 1) Load + merge views
#----------------------------
load_and_merge_views <- function(PROC, CLUS,
                                 c_file = "C_view.csv",
                                 p_file = "P_view_scaled.csv",
                                 s_file = "S_view.csv",
                                 y_file = "Y_validation.csv",
                                 l_file = "snf_clusters_all.csv") {
  
  C <- readr::read_csv(file.path(PROC, c_file), show_col_types = FALSE)
  P <- readr::read_csv(file.path(PROC, p_file), show_col_types = FALSE)
  S <- readr::read_csv(file.path(PROC, s_file), show_col_types = FALSE)
  Y <- readr::read_csv(file.path(PROC, y_file), show_col_types = FALSE)
  L <- readr::read_csv(file.path(CLUS, l_file), show_col_types = FALSE)
  
  dfs <- list(C = C, P = P, S = S, Y = Y, L = L)
  for (nm in names(dfs)) {
    if (!("eid" %in% names(dfs[[nm]]))) stop(sprintf("eid missing in %s", nm))
  }
  
  # inner-join C,P,S,Y on eid (like pandas merge default)
  df <- reduce(list(C, P, S, Y), ~ inner_join(.x, .y, by = "eid"))
  
  # inner-join with L subset
  if (!all(c("stratum", "label") %in% names(L))) {
    stop("mmsp_clusters.csv must contain columns: stratum, label (and eid)")
  }
  
  df <- inner_join(df, L %>% select(eid, stratum, label), by = "eid")
  df
}

km_fitx <- survfit(Surv(d.time, death) ~ stratum, data = df)

# 3. View the Median Survival Times
print(km_fitx)

#----------------------------
# 2) Cox per stratum (+ HR CSV, zph CSV, forest plot)
#----------------------------
run_cox_by_stratum <- function(df,
                               TAB,
                               severity_priority = c("aps", "sps", "scoma"),
                               extra_covars = c("age", "num.co", "sfdm2"),
                               condition_covars = character(),   # e.g. c("cond_copd", "cond_ckd")
                               min_events = 5,
                               reference_label = NULL,           # e.g. "0" or "1" or "Cluster1"
                               forest_terms = c("label", "all")) {
  
  forest_terms <- match.arg(forest_terms)
  
  dir.create(TAB, showWarnings = FALSE, recursive = TRUE)
  
  required_base <- c("d.time", "death", "label", "stratum", "eid")
  missing_base <- setdiff(required_base, names(df))
  if (length(missing_base) > 0) stop("Missing required columns: ", paste(missing_base, collapse = ", "))
  
  # helper: sanitize file-safe stratum name
  safe_name <- function(x) str_replace_all(as.character(x), "[^A-Za-z0-9_\\-]+", "_")
  
  out <- list()
  
  for (s in sort(unique(df$stratum))) {
    g <- df %>% filter(stratum == s)
    
    # choose severity covar (aps > sps > scoma)
    severity <- severity_priority[severity_priority %in% names(g)]
    severity <- if (length(severity) > 0) severity[1] else NA_character_
    
    # build covariate list: chosen severity + extras + conditions (keep those present)
    cand_covars <- c(severity, extra_covars, condition_covars)
    cand_covars <- cand_covars[!is.na(cand_covars)]
    covars_present <- cand_covars[cand_covars %in% names(g)]
    covars_present <- unique(covars_present)
    
    if (is.na(severity)) {
      out[[length(out) + 1]] <- tibble(stratum = s, ok = FALSE, note = "no severity covariate found (aps/sps/scoma)")
      next
    }
    
    # ensure we at least have severity + survival cols + label
    needed <- unique(c("d.time", "death", "label", covars_present))
    dat <- g %>% select(all_of(needed))
    
    # coerce types
    dat <- dat %>%
      mutate(
        d.time = as.numeric(d.time),
        death  = as.integer(death)
      )
    
    # treat label as factor => "one-hot" handled automatically (reference is baseline level)
    dat <- dat %>% mutate(label = as.factor(label))
    if (!is.null(reference_label) && reference_label %in% levels(dat$label)) {
      dat <- dat %>% mutate(label = relevel(label, ref = reference_label))
    }
    
    # drop NA rows
    dat <- tidyr::drop_na(dat)
    
    n_events <- sum(dat$death == 1, na.rm = TRUE)
    if (n_events < min_events) {
      out[[length(out) + 1]] <- tibble(stratum = s, ok = FALSE, note = "too few events", events = n_events)
      next
    }
    
    # formula: Surv ~ label + covars (excluding label if user accidentally put it in covars)
    covars_for_formula <- setdiff(covars_present, "label")
    rhs <- paste(c("label", covars_for_formula), collapse = " + ")
    fml <- as.formula(paste0("survival::Surv(d.time, death) ~ ", rhs))
    
    # fit Cox model (robust SE if supported by your survival version)
    fit <- tryCatch(
      survival::coxph(fml, data = dat, robust = TRUE),
      error = function(e) e
    )
    
    if (inherits(fit, "error")) {
      out[[length(out) + 1]] <- tibble(stratum = s, ok = FALSE, error = fit$message)
      next
    }
    
    # HR table
    hr <- broom::tidy(fit, exponentiate = TRUE, conf.int = TRUE) %>%
      rename(
        term = term,
        HR = estimate,
        CI_low = conf.low,
        CI_high = conf.high,
        p = p.value
      ) %>%
      mutate(
        stratum = s,
        n = nrow(dat),
        events = n_events
      ) %>%
      select(stratum, n, events, term, HR, CI_low, CI_high, p)
    
    hr_path <- file.path(TAB, paste0("cox_", safe_name(s), ".csv"))
    readr::write_csv(hr, hr_path)
    
    # Schoenfeld residual test
    zph <- tryCatch(survival::cox.zph(fit), error = function(e) e)
    if (!inherits(zph, "error")) {
      ztab <- as.data.frame(zph$table)
      ztab$term <- rownames(ztab)
      rownames(ztab) <- NULL
      ztab <- ztab %>% relocate(term)
      
      zph_path <- file.path(TAB, paste0("cox_zph_", safe_name(s), ".csv"))
      readr::write_csv(ztab, zph_path)
    }
    
    # Forest plot
    plot_df <- hr
    if (forest_terms == "label") {
      plot_df <- plot_df %>% filter(str_starts(term, "label"))
    }
    
    if (nrow(plot_df) > 0) {
      plot_df <- plot_df %>%
        mutate(term = fct_reorder(term, HR))  # order by HR for readability
      
      p_forest <- ggplot(plot_df, aes(x = HR, y = term)) +
        geom_vline(xintercept = 1, linetype = 2) +
        geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.2) +
        geom_point() +
        scale_x_log10() +
        labs(
          title = paste0("Cox HRs (stratum = ", s, ")"),
          x = "Hazard Ratio (log scale)",
          y = NULL
        ) +
        theme_minimal(base_size = 12)
      
      plot_path <- file.path(TAB, paste0("forest_", safe_name(s), ".png"))
      ggsave(plot_path, p_forest, width = 9, height = max(4, 0.25 * nrow(plot_df) + 2))
    }
    
    out[[length(out) + 1]] <- tibble(stratum = s, ok = TRUE, n = nrow(dat), events = n_events)
  }
  
  overview <- bind_rows(out)
  readr::write_csv(overview, file.path(TAB, "cox_overview.csv"))
  invisible(overview)
}

#----------------------------
# 3) Example usage
#----------------------------
# Set paths
CLUS <- "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/data/02_clusters"
PROC <- "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/data/01_processed"
TAB  <- "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/Rscripts_tests/Out_mmsp"

df <- load_and_merge_views(PROC, CLUS)

# Add covariates you want:
# - age, num.co, sfdm2 are already in extra_covars by default
# - add condition flags if you have them as columns in df
condition_covars <- c("dzgroup_mosf_malig", "dzgroup_arf_mosf", "dzgroup_cirrhosis")  # <-- EDIT to your real column names

run_cox_by_stratum(
  df = df,
  TAB = TAB,
  severity_priority = c("aps", "sps", "scoma"),
  extra_covars = c("age", "num.co", "sfdm2"),  # plus severity covar chosen per stratum
  condition_covars = condition_covars,
  min_events = 5,
  reference_label = NULL,       # or set e.g. "0" / "1" / "ClusterA"
  forest_terms = "label"        # "label" (clusters only) or "all"
)
