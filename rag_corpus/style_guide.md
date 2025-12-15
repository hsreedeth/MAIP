# MAIP Rulecard & Flowchart Style Guide

This document defines how ICU phenotyping rules should be translated into clinician-facing text.  
It applies to:

- Phenotype **narratives** (short descriptions of each phenotype).
- **Rulecards** (explicit IF/THEN rules).
- **ASCII flowcharts** (tree-like visual skeletons of the rules).

The rules are derived from a surrogate decision tree trained to approximate data-driven SNF phenotypes.  
The goal is to make the logic readable **without changing it**.

---

## 1. General principles

1. **Preserve logic exactly**

   - Do **not** introduce new conditions, thresholds, or phenotypes.
   - Do **not** delete or merge rules unless explicitly instructed.
   - Every logical element in the JSON (feature, operator, threshold) must appear in an equivalent IF/THEN rule.
   - If you cannot fit everything, prioritise complete rulecard and ASCII flowchart over repetition of the style guide itself.
   - Do **not** include raw JSON or code blocks in the final answer. Only output the narrative, rulecard and ASCII flowchart.

2. **Use only trusted sources**

   - Use information from:
     - The JSON rules for that phenotype.
     - The variable dictionary (names, units, scales, notes).
     - The phenotype summary for that label.
     - This style guide.
   - Do **not** invent clinical facts or mechanisms beyond those sources.

3. **Describe phenotypes, not prescribe care**

   - Describe what kind of patients tend to fall into each phenotype.
   - Do **not** give treatment advice or management recommendations.
   - Avoid language that sounds prescriptive (e.g. “should be treated with…”).

4. **Uncertainty and frequency**

   - Use formulations like “patients in this phenotype tend to…” rather than “always”.
   - If support/purity are given, you may say “the rule mainly selects patients with…” but do not over-interpret small differences.

---

## 2. Structure of the output

For each phenotype, aim for three clearly separated sections with fixed headings:

1. `Key idea` – one or two sentences.
2. `Rulecard` – bullet list of IF/THEN rules.
3. `ASCII flowchart` – a tree skeleton representing the same rules.

Example structure:

```text
Phenotype High_MM_0 – Multi-organ failure with diabetes and cancer

Key idea:
Patients with high multimorbidity who present with multi-organ failure, often on a background of diabetes and cancer, and with markedly abnormal acute physiology.

Rulecard:
- IF ...
- IF ...
- ...

ASCII flowchart:
[Start]
  |
  |-- ...
