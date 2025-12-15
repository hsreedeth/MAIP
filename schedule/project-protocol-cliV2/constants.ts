import { Week } from './types';

export const PROJECT_PLAN: Week[] = [
  {
    id: "week-1",
    title: "Week 1",
    subtitle: "Analyses & Methods consolidation",
    days: [
      {
        id: "w1-d1-2",
        title: "Day 1–2: Outcome model decision + rerun",
        tasks: [
          {
            id: "w1-d1-task1",
            title: "Decide primary survival analysis path",
            subtasks: [
              { id: "w1-d1-t1-s1", text: "Make R Cox models primary (cleaner story)" },
              { id: "w1-d1-t1-s2", text: "Keep Python external_validation (KM/logrank, LOS/cost KW, profiles)" }
            ]
          },
          {
            id: "w1-d1-task2",
            title: "Actions: R & Python Execution",
            subtasks: [
              { id: "w1-d1-t2-s1", text: "Finalise adjustments in R (age, sex, severity)" },
              { id: "w1-d1-t2-s2", text: "Re-run R Cox script per stratum" },
              { id: "w1-d1-t2-s3", text: "Export HR tables by cluster" },
              { id: "w1-d1-t2-s4", text: "Export Global tests (LR/Wald)" },
              { id: "w1-d1-t2-s5", text: "Re-run Python external_validation.py" },
              { id: "w1-d1-t2-s6", text: "Regenerate KM logrank summary" },
              { id: "w1-d1-t2-s7", text: "Regenerate LOS/cost Kruskal–Wallis" },
              { id: "w1-d1-t2-s8", text: "Regenerate cluster profiles" }
            ]
          }
        ]
      },
      {
        id: "w1-d3-4",
        title: "Day 3–4: Methods section clean-up",
        tasks: [
          {
            id: "w1-d3-task1",
            title: "Phenotyping + surrogate model",
            subtasks: [
              { id: "w1-d3-t1-s1", text: "Tighten MMSP/SNF-lite descriptions" },
              { id: "w1-d3-t1-s2", text: "Clarify surrogate tree training metrics" }
            ]
          },
          {
            id: "w1-d3-task2",
            title: "Outcome models",
            subtasks: [
              { id: "w1-d3-t2-s1", text: "Describe unadjusted KM/logrank per stratum" },
              { id: "w1-d3-t2-s2", text: "Describe adjusted Cox models in R" },
              { id: "w1-d3-t2-s3", text: "Clarify Kruskal–Wallis usage for LOS/cost" }
            ]
          },
          {
            id: "w1-d3-task3",
            title: "RAG section alignment",
            subtasks: [
              { id: "w1-d3-t3-s1", text: "Update ref: remote GPT-4.1-mini via API" },
              { id: "w1-d3-t3-s2", text: "Emphasise retrieval corpus (JSON/Dictionary)" },
              { id: "w1-d3-t3-s3", text: "Confirm prompts enforce no new logic" },
              { id: "w1-d3-t3-s4", text: "Document QC checks (coverage, alignment)" }
            ]
          }
        ]
      },
      {
        id: "w1-d5-7",
        title: "Day 5–7: Internal QC & documentation",
        tasks: [
          {
            id: "w1-d5-task1",
            title: "QC for analyses",
            subtasks: [
              { id: "w1-d5-t1-s1", text: "Confirm all tables present in reports/tables" },
              { id: "w1-d5-t1-s2", text: "Sanity-check HR magnitudes and p-values" },
              { id: "w1-d5-t1-s3", text: "Create analyst notes markdown" }
            ]
          },
          {
            id: "w1-d5-task2",
            title: "Methods text cross-check",
            subtasks: [
              { id: "w1-d5-t2-s1", text: "Verify every method has a script" },
              { id: "w1-d5-t2-s2", text: "Remove legacy protocol items not in code" },
              { id: "w1-d5-t2-s3", text: "Add missing code descriptions to Methods" }
            ]
          }
        ]
      }
    ]
  },
  {
    id: "week-2",
    title: "Week 2",
    subtitle: "Results, figures and phenotype stories",
    days: [
      {
        id: "w2-d8-10",
        title: "Day 8–10: Phenotype characterisation",
        tasks: [
          {
            id: "w2-d8-task1",
            title: "Narrative Generation",
            subtasks: [
              { id: "w2-d8-t1-s1", text: "Write chronic disease profile narrative" },
              { id: "w2-d8-t1-s2", text: "Write acute physiology narrative" },
              { id: "w2-d8-t1-s3", text: "Write socio-context narrative" }
            ]
          },
          {
            id: "w2-d8-task2",
            title: "Mapping to Outputs",
            subtasks: [
              { id: "w2-d8-t2-s1", text: "Update Manuscript Results" },
              { id: "w2-d8-t2-s2", text: "Update Portfolio 'What phenotypes look like'" }
            ]
          }
        ]
      },
      {
        id: "w2-d11-12",
        title: "Day 11–12: Outcome sections",
        tasks: [
          {
            id: "w2-d11-task1",
            title: "Survival Analysis Writing",
            subtasks: [
              { id: "w2-d11-t1-s1", text: "Summarise KM results per stratum" },
              { id: "w2-d11-t1-s2", text: "Summarise Cox results per stratum" }
            ]
          },
          {
            id: "w2-d11-task2",
            title: "LOS and Cost Writing",
            subtasks: [
              { id: "w2-d11-t2-s1", text: "Report significant KW differences" },
              { id: "w2-d11-t2-s2", text: "Align with survival patterns" }
            ]
          },
          {
            id: "w2-d11-task3",
            title: "Wiring to Documents",
            subtasks: [
              { id: "w2-d11-t3-s1", text: "Update Manuscript Results (concise)" },
              { id: "w2-d11-t3-s2", text: "Update Portfolio 'Outcome patterns'" }
            ]
          }
        ]
      },
      {
        id: "w2-d13-14",
        title: "Day 13–14: Figures & tables",
        tasks: [
          {
            id: "w2-d13-task1",
            title: "Finalise Visuals",
            subtasks: [
              { id: "w2-d13-t1-s1", text: "KM curves per stratum" },
              { id: "w2-d13-t1-s2", text: "Bar/dot plot of median LOS/cost" },
              { id: "w2-d13-t1-s3", text: "Compact HR table for Supplement" }
            ]
          },
          {
            id: "w2-d13-task2",
            title: "Lockdown",
            subtasks: [
              { id: "w2-d13-t2-s1", text: "Lock filenames and references" }
            ]
          }
        ]
      }
    ]
  },
  {
    id: "week-3",
    title: "Week 3",
    subtitle: "Repo, portfolio and webapp MVP",
    days: [
      {
        id: "w3-d15-17",
        title: "Day 15–17: Codebase hardening",
        tasks: [
          {
            id: "w3-d15-task1",
            title: "Repo Structure",
            subtasks: [
              { id: "w3-d15-t1-s1", text: "Organize src/ (Python modules + CLI)" },
              { id: "w3-d15-t1-s2", text: "Organize R/ scripts" },
              { id: "w3-d15-t1-s3", text: "Document data/ expectations" },
              { id: "w3-d15-t1-s4", text: "Clean reports/ folder" }
            ]
          },
          {
            id: "w3-d15-task2",
            title: "Documentation",
            subtasks: [
              { id: "w3-d15-t2-s1", text: "Define pipeline run in README" },
              { id: "w3-d15-t2-s2", text: "Add environment.yml / requirements.txt" },
              { id: "w3-d15-t2-s3", text: "Note R package versions" }
            ]
          }
        ]
      },
      {
        id: "w3-d18-19",
        title: "Day 18–19: Portfolio page completion",
        tasks: [
          {
            id: "w3-d18-task1",
            title: "Content Fill",
            subtasks: [
              { id: "w3-d18-t1-s1", text: "Surrogate tree & rulecard link" },
              { id: "w3-d18-t1-s2", text: "Phenotype characteristics grid" },
              { id: "w3-d18-t1-s3", text: "Outcome associations text" },
              { id: "w3-d18-t1-s4", text: "RAG & QC explanation (prose)" }
            ]
          }
        ]
      },
      {
        id: "w3-d20-21",
        title: "Day 20–21: Webapp MVP scaffold",
        tasks: [
          {
            id: "w3-d20-task1",
            title: "Scaffold Build",
            subtasks: [
              { id: "w3-d20-t1-s1", text: "Create documented JSON schema" },
              { id: "w3-d20-t1-s2", text: "Build read-only FE (load JSON)" },
              { id: "w3-d20-t1-s3", text: "Implement 2D scatter plot" },
              { id: "w3-d20-t1-s4", text: "Implement tooltip/panel on click" },
              { id: "w3-d20-t1-s5", text: "Update portfolio with 'In Progress' note" }
            ]
          }
        ]
      }
    ]
  },
  {
    id: "week-4",
    title: "Week 4",
    subtitle: "Preprint packaging & final coherence",
    days: [
      {
        id: "w4-d22-24",
        title: "Day 22–24: Manuscript polish",
        tasks: [
          {
            id: "w4-d22-task1",
            title: "Line Editing",
            subtasks: [
              { id: "w4-d22-t1-s1", text: "Check terminology consistency" },
              { id: "w4-d22-t1-s2", text: "Harmonise RAG description" },
              { id: "w4-d22-t1-s3", text: "Clarify limitations statement" }
            ]
          },
          {
            id: "w4-d22-task2",
            title: "Figure Selection",
            subtasks: [
              { id: "w4-d22-t2-s1", text: "Select Main text figures" },
              { id: "w4-d22-t2-s2", text: "Select Supplement figures" }
            ]
          }
        ]
      },
      {
        id: "w4-d25-26",
        title: "Day 25–26: Reproducibility and release",
        tasks: [
          {
            id: "w4-d25-task1",
            title: "Release Management",
            subtasks: [
              { id: "w4-d25-t1-s1", text: "Tag GitHub release (v0.9-preprint)" },
              { id: "w4-d25-t1-s2", text: "Write release notes" },
              { id: "w4-d25-t1-s3", text: "Update README with links (Portfolio, Preprint)" }
            ]
          }
        ]
      },
      {
        id: "w4-d27-28",
        title: "Day 27–28: medRxiv submission",
        tasks: [
          {
            id: "w4-d27-task1",
            title: "Submission Prep",
            subtasks: [
              { id: "w4-d27-t1-s1", text: "Prepare Manuscript PDF + Source" },
              { id: "w4-d27-t1-s2", text: "Prepare disclosures/ethics statement" },
              { id: "w4-d27-t1-s3", text: "Separate figure files if needed" },
              { id: "w4-d27-t1-s4", text: "Final check and Submit" }
            ]
          }
        ]
      }
    ]
  }
];