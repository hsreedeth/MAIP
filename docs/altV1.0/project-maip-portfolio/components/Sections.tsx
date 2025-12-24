import React from 'react';

// --- Shared Helper Components ---

const SectionHeading = ({ children }: { children: React.ReactNode }) => (
  <h2 className="text-[32px] font-medium mt-20 mb-6 tracking-[0.08em] uppercase text-[#111111]">
    {children}
  </h2>
);

const SectionLead = ({ children }: { children: React.ReactNode }) => (
  <div className="max-w-[720px] text-[15px] leading-[1.6] text-[#8f8a82] mb-4">
    {children}
  </div>
);

const Divider = () => (
  <hr className="border-0 border-t border-[#e3d7c6] my-10" />
);

// Improved Sidenote Component
// This fits into the text flow on mobile, and floats on desktop if space permits
const Sidenote = ({ number, children, link, linkText }: { number: string, children: React.ReactNode, link?: string, linkText?: string }) => {
    return (
        <span className="relative group inline-block align-baseline">
             {/* The reference number in text */}
            <sup className="text-xs text-[#777] font-bold cursor-pointer ml-0.5">{number}</sup>
            
            {/* The note content - rendered as a block on mobile, positioned on desktop */}
            <span className="block my-2 p-3 bg-stone-100 border-l-2 border-stone-300 text-[13px] leading-relaxed text-[#6b6b6b] 
                           md:absolute md:w-[220px] md:bg-transparent md:border-l-0 md:p-0 md:my-0 md:top-0
                           md:left-[calc(100%+2rem)] lg:left-[calc(100%+3rem)] z-10">
                <span className="font-bold mr-1 hidden md:inline">{number}</span>
                {children}
                {link && (
                    <>
                        {' '}
                        <a href={link} target="_blank" rel="noopener noreferrer" className="underline decoration-1 underline-offset-2 text-inherit hover:text-black">
                            {linkText || 'Link'}
                        </a>.
                    </>
                )}
            </span>
        </span>
    );
};


// --- Specific Sections ---

export const IntroSection: React.FC = () => {
  return (
    <section>
      <div className="text-[40px] font-medium tracking-[0.06em] uppercase mb-6 mt-[120px] text-[#111111]">
        Introduction
      </div>
      <Divider />
      
      {/* Flattened to single column as requested */}
      <div className="max-w-3xl relative">
        <div className="text-[15px] leading-[1.6] space-y-4 text-justify">
          <p>
            Critically ill patients arriving in intensive care rarely fit neat templates. Even within a single ICU, patients differ widely in age, comorbidity burden, physiology and social context. This heterogeneity makes it hard to reason about prognosis, compare like with like, or decide which treatment strategies are most appropriate.
          </p>
          <p className="relative">
            Unsupervised machine learning is an obvious tool for this problem, and a large body of ICU phenotyping work already exists. However, in practice many clustering pipelines are dominated by the blunt signal of multimorbidity: patients with more chronic disease, and who are older, simply get grouped together. The resulting “phenotypes” often mirror gradients in age and comorbidity count rather than revealing distinct, acute physiological patterns that could plausibly change clinical decisions.
             <Sidenote 
                number="1" 
                link="https://github.com/hsreedeth/ProjectMAIP" 
                linkText="Project-MAIP"
             >
                 See GitHub repository:
             </Sidenote>
          </p>
          <p className="relative">
            The Multimorbidity-Anchored ICU Phenotypes (MAIP) project
            <Sidenote 
                number="2" 
                link="https://hbiostat.org/data/" 
                linkText="Support-II"
             >
                 Vanderbilt University Department of Biostatistics, Professor Frank Harrell 2022
             </Sidenote>
             {' '} is my attempt to tackle that limitation in a principled, end-to-end way using the SUPPORT-II cohort (over 9,000-patient sample from the Study to Understand Prognoses, Preferences, Outcomes and Risks of Treatment). I designed MAIP as a full pipeline that (i) explicitly separates chronic disease burden from acute physiology and (ii) carries the resulting clusters all the way through to a human-readable bedside tool.
          </p>
          <p>
            Methodologically, MAIP combines two complementary unsupervised pathways: a multimorbidity-stratified phenotyping route (MMSP) and a simplified multi-view Similarity Network Fusion implementation (SNF-lite). Once stable clusters are identified, I train a sparse decision-tree surrogate model to approximate the cluster assignments. The tree’s branch structure is exported as structured JSON rules, which are then translated into a clinician-facing rulecard and ASCII flowchart via a retrieval-augmented generation (RAG) pipeline running on remotely through chatgpt-4.1-mini model (with 7B parameters). The RAG layer is grounded in a curated corpus of variable definitions, phenotype summaries and style guidance, and is coupled with programmatic checks to ensure that the textual rulecard remains logically faithful to the underlying JSON rules.
          </p>
          <p>
            The project therefore has two concrete goals:
          </p>
          <ul className="list-none pl-0">
             <li>• <strong>Derive</strong> ICU phenotypes that are not just “more sick vs less sick” along a comorbidity axis</li>
             <li>• <strong>Demonstrate</strong> a robust, versioned , maintainable & auditable pathway from unsupervised clustering to a deployable, human-readable decision tool.</li>
          </ul>
        </div>
      </div>
    </section>
  );
};

export const OverviewSection: React.FC = () => {
    const services = [
        {
            title: "Study Design & Data Source",
            text: "Retrospective cohort analysis of ~9,000 critically ill adults from the multicentre SUPPORT-II dataset. Includes demographics, comorbidities, physiological measurements at ICU admission, severity scores and longitudinal outcomes. Only features required for clustering are used; outcome variables (mortality, length of stay) are held out for independent validation."
        },
        {
            title: "Feature Engineering & Views",
            text: "Constructed a feature matrix organised into three clinical views: (1) chronic disease burden (C-View), (2) acute physiology (P-View), and (3) socio-contextual characteristics (S-View). Categorical variables are encoded interpretably and continuous variables are transformed where needed and standardised. Missing data are imputed using an outcome-free recipe, with optional missingness indicators. Outcomes live in a separate Y-matrix."
        },
        {
            title: "Phenotyping: MMSP & SNF-lite",
            text: "Implemented two complementary unsupervised pipelines. MMSP stratifies patients by multimorbidity level, runs PCA on the z-scaled physiology view within each stratum, and applies PAM k-medoids on Euclidean distances in the PCA space. SNF-lite constructs view-specific similarity matrices (Gower / RBF), fuses them with a lightweight Similarity Network Fusion routine, and uses spectral clustering on the fused graph. Candidate cluster numbers (K) are chosen using bootstrap stability metrics, standard internal indices and clinical interpretability."
        },
        {
            title: "Surrogate Decision Tree",
            text: "Trained a shallow, constrained decision-tree surrogate on a curated set of interpretable variables to approximate final cluster assignments. Tuned depth and node-size to balance fidelity and simplicity. Evaluated performance using accuracy, macro-averaged F1 and confusion matrices to identify phenotypes that are intrinsically hard to separate with simple rules. Exported the final tree into a strict JSON rule schema capturing features, thresholds, operators and phenotype labels."
        },
        {
            title: "RAG-Assisted Translation & QC",
            text: "Used a retrieval-augmented pipeline with remote GPT-4.1-mini to convert JSON surrogate-tree rules into clinician-facing rulecards and ASCII flowcharts. For each phenotype, the prompt combines the canonical JSON rules with a curated variable dictionary, analytically derived phenotype summaries and a style guide. Prompts enforce “no new logic” and a one-to-one mapping from JSON to text. Programmatic checks compare JSON rules and textual rules on synthetic patient profiles and verify consistency of feature names, units and phenotype labels. All JSON, prompts and model outputs are versioned for auditability."
        }
    ];

    return (
        <section>
            <SectionHeading>Project Overview</SectionHeading>
            <SectionLead>
                I treat each project as a combination of research and translation. Below is how that played out in MAIP similar to my other related work.
            </SectionLead>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 text-sm leading-[1.7] mt-8">
                {services.map((s, idx) => (
                    <div key={idx} className="service-block">
                        <div className="font-semibold mb-1.5 text-[15px]">{s.title}</div>
                        <p className="m-0 text-[#2c2c2c]">{s.text}</p>
                    </div>
                ))}
            </div>

            <div className="mt-12">
                 <img
                    src="/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/PortfolioMain/Overview Schematic.svg"
                    alt="Overview schematic"
                    className="block mx-auto max-w-full h-auto"
                />
            </div>
        </section>
    );
};

export const VariableSummary: React.FC = () => {
  return (
    <section>
      <SectionHeading>Data source, Feature engineering and View construction</SectionHeading>
      <div className="max-w-3xl text-[15px] leading-[1.6] text-justify mb-8">
         <p>
            I worked with the SUPPORT-II ICU cohort (~9,000 critically ill adults from multiple North American centres in the late 1980s–1990s), which includes rich baseline demographics, comorbidities, physiology at ICU admission, prognostic scores and follow-up outcomes (mortality, length of stay, resource use). For phenotyping, I built a clean feature matrix X and a held-out outcome matrix Y: X feeds the unsupervised models and surrogate trees, while Y (in-hospital and 5-year mortality, time-to-event and utilisation) is used only for external validation, never for clustering. X is organised into three clinically meaningful views: a comorbidity view (num.co, disease indicators, grouped diagnoses), a physiology view (vitals, labs, blood gases, severity scores., transformed and z-scaled), and a socio-contextual view (age, sex, SES proxies, treatment context). Categorical variables are encoded to preserve interpretability, and missing data are handled by an outcome-blind imputation recipe with documented seeds and predictors. This design gives me a reproducible, view-structured feature space for discovering phenotypes and a separate outcome layer for testing whether those phenotypes actually improve prognostic performance.
         </p>
      </div>

      <h3 className="text-xl font-medium mt-12 mb-4">Variable Summary</h3>
      <SectionLead>
         Key baseline characteristics of the SUPPORT-II cohort. Continuous variables are summarised as mean ± SD and median (IQR); categorical variables as N (%).
      </SectionLead>

      <div className="overflow-x-auto mt-6">
        <table className="w-full border-collapse text-sm text-left">
            <thead>
                <tr className="border-b border-[#e3d7c6]">
                    <th className="py-2.5 px-1 pl-0 font-medium text-[13px] uppercase tracking-wider text-[#8f8a82]">Characteristic</th>
                    <th className="py-2.5 px-1 font-medium text-[13px] uppercase tracking-wider text-[#8f8a82]">N / Mean ± SD</th>
                    <th className="py-2.5 px-1 font-medium text-[13px] uppercase tracking-wider text-[#8f8a82]">Median (IQR)</th>
                    <th className="py-2.5 px-1 font-medium text-[13px] uppercase tracking-wider text-[#8f8a82]">Dtype</th>
                </tr>
            </thead>
            <tbody>
                {/* Demographics */}
                <tr className="bg-[#f2efe9]/50"><td colSpan={4} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">Demographics</td></tr>
                {[
                    ["Age (years)", "62.65 ± 15.59", "64.86 (52.80, 73.99)", "Continuous (slightly skewed)"],
                    ["Male sex", "5123 (56.27%)", "—", "Categorical"],
                    ["Race: White", "7190 (78.97%)", "—", "Categorical"],
                    ["Race: Black", "1391 (15.28%)", "—", "Categorical"],
                    ["Income (Level 1–4)", "1.94 ± 0.89", "2.0 (1.0, 2.0)", "Ordinal (skewed)"]
                ].map((row, i) => (
                    <tr key={i} className="border-b border-[#e3d7c6]">
                        {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                    </tr>
                ))}

                 {/* Multimorbidity */}
                 <tr className="bg-[#f2efe9]/50"><td colSpan={4} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">Multimorbidity & Chronic Conditions</td></tr>
                 {[
                    ["Number of comorbidities (num.co)", "1.87 ± 1.34", "2.0 (1.0, 3.0)", "Count (skewed)"],
                    ["Cancer (ca)", "0.55 ± 0.81", "0.0 (0.0, 1.0)", "Index / binary"],
                    ["Diabetes mellitus", "1789 (19.65%)", "—", "Binary"],
                    ["Chronic heart failure (dzgroup_chf)", "1387 (15.23%)", "—", "Binary"],
                    ["Chronic obstructive pulmonary disease (dzgroup_copd)", "967 (10.62%)", "—", "Binary"]
                 ].map((row, i) => (
                    <tr key={i} className="border-b border-[#e3d7c6]">
                        {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                    </tr>
                ))}

                {/* Severity */}
                <tr className="bg-[#f2efe9]/50"><td colSpan={4} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">Severity & Acute Status</td></tr>
                {[
                    ["Acute Physiology Score (aps)", "37.60 ± 19.90", "34.0 (23.0, 49.0)", "Continuous (skewed)"],
                    ["Mean blood pressure (meanbp, mmHg)", "84.55 ± 27.69", "77.0 (63.0, 107.0)", "Continuous (skewed)"],
                    ["PaO2/FiO2 ratio (pafi)", "263.39 ± 102.67", "276.19 (180.00, 333.30)", "Continuous"],
                    ["Albumin (alb, g/dL)", "3.15 ± 0.68", "3.5 (2.7, 3.5)", "Continuous (low average)"],
                    ["Creatinine (crea, mg/dL)", "1.77 ± 1.68", "1.20 (0.90, 1.90)", "Continuous (highly skewed)"]
                ].map((row, i) => (
                    <tr key={i} className="border-b border-[#e3d7c6]">
                        {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                    </tr>
                ))}

                {/* Outcomes */}
                <tr className="bg-[#f2efe9]/50"><td colSpan={4} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">Outcomes & Resource Use</td></tr>
                {[
                    ["Death (any)", "6201 (68.10%)", "—", "Binary"],
                    ["Hospital death (hospdead)", "2359 (25.91%)", "—", "Binary"],
                    ["Length of stay (slos, days)", "17.86 ± 22.01", "11.0 (6.0, 20.0)", "Count (highly skewed)"],
                    ["Total medical cost (totmcst, $)", "28,839.15 ± 43,608.61", "13,228.93 (5,179.89, 34,263.85)", "Cost (missing ≈ 38%)"]
                ].map((row, i) => (
                    <tr key={i} className="border-b border-[#e3d7c6]">
                        {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                    </tr>
                ))}
            </tbody>
        </table>
      </div>
    </section>
  );
};

export const PhenotypingStrategy: React.FC = () => {
    return (
        <section>
            <SectionHeading>Phenotyping Strategy</SectionHeading>
            <SectionLead>
                Two complementary unsupervised pathways: one conditions on multimorbidity and looks for physiology-driven patterns; the other fuses comorbidity, physiology and context into a single set of ICU phenotypes.
            </SectionLead>
            
            <div className="max-w-3xl">
                <p className="font-bold mb-2">MMSP: Physiology within Burden Strata</p>
                <p className="mb-6 text-[15px] leading-relaxed text-justify">
                    Patients are first split into Low, Mid and High multimorbidity strata using the comorbidity count. Within each stratum I cluster only on the z-scaled physiology view (vitals, labs, severity scores). PCA reduces this to a low-dimensional acute-state space, Euclidean distances are computed, and PAM (k-medoids) is used to derive clusters. Candidate K values are compared using bootstrap stability (Adjusted Rand Index) and internal indices, then reviewed for clinical interpretability. The result is a set of physiology-driven phenotypes defined separately within each level of multimorbidity.
                </p>
                <div className="my-8">
                     <img
                        src="/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/PortfolioMain/MMSP.svg"
                        alt="MMSP"
                        className="block mx-auto max-w-full h-auto"
                    />
                    <p className="text-sm text-[#7c7165] mt-2 font-medium">Figure 1: Multimorbidity-Stratified Phenotyping (MMSP) workflow.</p>
                </div>
                {/* Inline styled note */}
                <div className="bg-white/50 border-l-4 border-[#e3d7c6] p-4 text-sm text-[#6b6b6b] mb-8">
                    <span className="font-bold mr-2">Figure 1</span>
                    Multimorbidity-Stratified Phenotyping (MMSP) workflow. Patients are first partitioned into low, mid and high multimorbidity strata by comorbidity count. Within each stratum, the day-3 physiology view (vitals, labs and severity scores) is reduced with PCA, Euclidean distances are computed, and PAM (k-medoids) clustering is applied to yield separate high-, mid- and low-burden ICU phenotypes.
                </div>
            </div>
        </section>
    );
};

export const VisualStrip: React.FC<{ 
  title: string; 
  description: string; 
  images: { src: string; caption: string }[];
  note?: string; 
}> = ({ title, description, images, note }) => {
  return (
    <section className="py-16 relative">
        <p className="font-bold text-lg mb-2">{title}</p>
        <SectionLead>{description}</SectionLead>

        {/* The Grid - using standard container rules to avoid scrollbar */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
            {images.map((img, i) => (
                <figure key={i} className="m-0 text-center">
                    <img src={img.src} alt={img.caption} className="w-full h-auto block shadow-sm" />
                    <figcaption className="mt-4 text-xs font-semibold tracking-wider uppercase text-[#7c7165]">
                        {img.caption}
                    </figcaption>
                </figure>
            ))}
        </div>
        
        {note && (
            <div className="bg-white/50 border-l-4 border-[#e3d7c6] p-4 text-sm text-[#6b6b6b] mt-8 max-w-4xl mx-auto">
                 {note}
            </div>
        )}
    </section>
  );
};

export const InternalValidation: React.FC = () => {
    return (
        <section>
            <h3 className="text-[32px] font-medium mt-20 mb-6 tracking-[0.08em] uppercase text-[#111111]">
                Internal Validation of MMSP and <br />SNF-lite Clusters
            </h3>
            <div className="max-w-3xl text-[15px] leading-[1.6] text-justify mb-8">
                <p>
                    Internal clustering performance of the multimorbidity-stratified phenotyping (MMSP) solution and the Similarity Network Fusion (SNF-lite) solution across high, mid and low multimorbidity strata. For each stratum, MMSP metrics are reported at the rule-selected K from phase 1, and SNF-lite metrics are reported at the K chosen by the same stability-first selection rule (search range K = 3–8). Higher stability (ARI), silhouette and Calinski–Harabasz indices, and lower Davies–Bouldin indices, indicate better cluster separation. Across all strata, the SNF-lite solution with K = 3 shows consistently superior internal validity compared with MMSP; combined with prognostic and parsimony analyses, this motivated choosing SNF-lite as the primary phenotyping system.
                </p>
            </div>
            
            <div className="overflow-x-auto mt-6">
                 <table className="w-full border-collapse text-sm text-left">
                    <thead>
                        <tr className="border-b border-[#e3d7c6]">
                            {["Stratum", "Method", "Clusters (K)", "Stability (ARI)", "Silhouette", "Calinski–Harabasz", "Davies–Bouldin"].map(h => (
                                <th key={h} className="py-2.5 px-1 font-medium text-[13px] uppercase tracking-wider text-[#8f8a82]">{h}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {/* High MM */}
                         <tr className="bg-[#f2efe9]/50"><td colSpan={7} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">High multimorbidity (4+ chronic conditions)</td></tr>
                         {[
                             ["High MM", "MMSP", "5", "0.221", "0.115", "91.16", "2.48"],
                             ["High MM", "SNF-lite", "3", "0.477", "0.273", "406.91", "1.25"]
                         ].map((row, i) => (
                             <tr key={i} className="border-b border-[#e3d7c6]">
                                 {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                             </tr>
                         ))}

                        {/* Mid MM */}
                         <tr className="bg-[#f2efe9]/50"><td colSpan={7} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">Mid multimorbidity (2–3 chronic conditions)</td></tr>
                         {[
                             ["Mid MM", "MMSP", "3", "0.321", "0.110", "367.23", "2.80"],
                             ["Mid MM", "SNF-lite", "3", "0.670", "0.319", "1566.30", "1.13"]
                         ].map((row, i) => (
                             <tr key={i} className="border-b border-[#e3d7c6]">
                                 {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                             </tr>
                         ))}

                         {/* Low MM */}
                         <tr className="bg-[#f2efe9]/50"><td colSpan={7} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">Low multimorbidity (0–1 chronic condition)</td></tr>
                         {[
                             ["Low MM", "MMSP", "4", "0.267", "0.114", "370.92", "2.61"],
                             ["Low MM", "SNF-lite", "3", "0.607", "0.388", "1961.89", "1.05"]
                         ].map((row, i) => (
                             <tr key={i} className="border-b border-[#e3d7c6]">
                                 {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                             </tr>
                         ))}
                    </tbody>
                 </table>
            </div>
        </section>
    );
};

export const PrognosticValue: React.FC = () => {
    return (
        <section>
             <h3 className="text-[32px] font-medium mt-20 mb-6 tracking-[0.08em] uppercase text-[#111111]">
                Cross-Validated Prognostic Value <br />of Phenotypes
            </h3>
            <div className="max-w-3xl text-[15px] leading-[1.6] text-justify mb-8">
                <p>
                    Prognostic performance was evaluated using 5-fold cross-validated Harrell’s C-index for 6-month mortality within each multimorbidity stratum. The base model included age, sex, number of comorbidities (num.co) and APS, and was compared against versions augmented with MMSP or SNF-lite phenotypes. Across low (N = 4,183; 2,765 deaths), mid (N = 3,822; 2,648 deaths) and high multimorbidity (N = 1,100; 788 deaths), SNF-lite clusters consistently delivered larger gains in cross-validated C-index than MMSP clusters, with the strongest uplift in the low- and high-burden groups.
                </p>
            </div>

            <div className="my-8">
                <img
                    src="/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/PortfolioMain/deltaC.png"
                    alt="Delta C Graph"
                    className="block mx-auto max-w-full h-auto"
                />
                 <p className="text-center text-sm font-medium mt-4">Figure 3: ΔC vs base (methods: MMSP, SNF-lite)</p>
            </div>

            <div className="bg-white/50 border-l-4 border-[#e3d7c6] p-4 text-sm text-[#6b6b6b] mb-8">
                <span className="font-bold mr-2">Figure 3</span>
                Cross-validated gain in Harrell’s C-index over the base clinical model, comparing MMSP and SNF-lite phenotypes within each multimorbidity stratum.
            </div>

            <div className="max-w-3xl text-[15px] leading-[1.6] mb-8">
                 <p>
                    Five-fold cross-validated Harrell’s C-index for 6-month mortality by multimorbidity stratum. The base model includes age, sex, number of comorbidities (num.co) and the Acute Physiology Score (APS). MMSP and SNF-lite phenotypes are added as categorical predictors; we report mean C-index and mean ΔC versus the base model across folds.
                 </p>
            </div>

             <div className="overflow-x-auto mt-6">
                 <table className="w-full border-collapse text-sm text-left">
                    <thead>
                        <tr className="border-b border-[#e3d7c6]">
                            {["Stratum", "Model", "Mean C-index", "Mean ΔC vs base"].map(h => (
                                <th key={h} className="py-2.5 px-1 font-medium text-[13px] uppercase tracking-wider text-[#8f8a82]">{h}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                         {/* High MM */}
                         <tr className="bg-[#f2efe9]/50"><td colSpan={4} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">High multimorbidity (4+ chronic conditions, N = 1,100; 788 deaths)</td></tr>
                         {[
                             ["High MM", "Base clinical model", "0.659", "—"],
                             ["High MM", "Base + MMSP phenotypes", "0.662", "+0.003"],
                             ["High MM", "Base + SNF-lite phenotypes", "0.674", "+0.015"]
                         ].map((row, i) => (
                             <tr key={i} className="border-b border-[#e3d7c6]">
                                 {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                             </tr>
                         ))}

                          {/* Mid MM */}
                         <tr className="bg-[#f2efe9]/50"><td colSpan={4} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">Mid multimorbidity (2–3 chronic conditions, N = 3,822; 2,648 deaths)</td></tr>
                         {[
                             ["Mid MM", "Base clinical model", "0.642", "—"],
                             ["Mid MM", "Base + MMSP phenotypes", "0.650", "+0.007"],
                             ["Mid MM", "Base + SNF-lite phenotypes", "0.655", "+0.012"]
                         ].map((row, i) => (
                             <tr key={i} className="border-b border-[#e3d7c6]">
                                 {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                             </tr>
                         ))}

                          {/* Low MM */}
                         <tr className="bg-[#f2efe9]/50"><td colSpan={4} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">Low multimorbidity (0–1 chronic condition, N = 4,183; 2,765 deaths)</td></tr>
                         {[
                             ["Low MM", "Base clinical model", "0.640", "—"],
                             ["Low MM", "Base + MMSP phenotypes", "0.646", "+0.006"],
                             ["Low MM", "Base + SNF-lite phenotypes", "0.682", "+0.042"]
                         ].map((row, i) => (
                             <tr key={i} className="border-b border-[#e3d7c6]">
                                 {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                             </tr>
                         ))}
                    </tbody>
                 </table>
             </div>
             <p className="mt-4 text-[15px] leading-[1.6] text-[#8f8a82]">
                Across all strata, SNF-lite phenotypes yielded larger cross-validated gains in discrimination than MMSP phenotypes, supporting SNF-lite as the primary multi-view phenotyping system in MAIP.
             </p>
        </section>
    );
};

export const Parsimony: React.FC = () => {
    return (
        <section>
            <h3 className="text-[32px] font-medium mt-20 mb-6 tracking-[0.08em] uppercase text-[#111111]">
                Parsimony & Interpretability of <br />Phenotyping Systems
            </h3>
             <div className="max-w-3xl text-[15px] leading-[1.6] text-justify mb-8">
                <p>
                    MMSP and SNF-lite differ not only in how they discover structure, but also in how compact and interpretable the resulting phenotypes are. Below, I summarise the number and balance of clusters and the complexity of the surrogate decision trees in each multimorbidity stratum.
                </p>
             </div>

             <div className="overflow-x-auto mt-6">
                 <table className="w-full border-collapse text-sm text-left">
                    <thead>
                        <tr className="border-b border-[#e3d7c6]">
                            {["Stratum", "Method", "Clusters (K)", "Smallest cluster (%)", "Largest cluster (%)", "Rules", "Median conditions", "Range"].map(h => (
                                <th key={h} className="py-2.5 px-1 font-medium text-[13px] uppercase tracking-wider text-[#8f8a82]">{h}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                         {/* High MM */}
                         <tr className="bg-[#f2efe9]/50"><td colSpan={8} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">High multimorbidity (4+ chronic conditions, N = 1,100)</td></tr>
                         {[
                             ["High MM", "MMSP", "5", "1.6%", "46.6%", "14", "4", "3–4"],
                             ["High MM", "SNF-lite", "3", "31.0%", "37.5%", "14", "4", "3–4"]
                         ].map((row, i) => (
                             <tr key={i} className="border-b border-[#e3d7c6]">
                                 {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                             </tr>
                         ))}
                          {/* Mid MM */}
                         <tr className="bg-[#f2efe9]/50"><td colSpan={8} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">Mid multimorbidity (2–3 chronic conditions, N = 3,822)</td></tr>
                         {[
                             ["Mid MM", "MMSP", "3", "21.7%", "49.8%", "16", "4", "4–4"],
                             ["Mid MM", "SNF-lite", "3", "26.2%", "44.2%", "13", "4", "2–4"]
                         ].map((row, i) => (
                             <tr key={i} className="border-b border-[#e3d7c6]">
                                 {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                             </tr>
                         ))}
                          {/* Low MM */}
                         <tr className="bg-[#f2efe9]/50"><td colSpan={8} className="py-2 px-1 font-semibold text-[#8f8a82] border-b border-[#e3d7c6]">Low multimorbidity (0–1 chronic condition, N = 4,183)</td></tr>
                         {[
                             ["Low MM", "MMSP", "4", "8.0%", "43.8%", "15", "4", "3–4"],
                             ["Low MM", "SNF-lite", "3", "11.4%", "58.6%", "11", "4", "2–4"]
                         ].map((row, i) => (
                             <tr key={i} className="border-b border-[#e3d7c6]">
                                 {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                             </tr>
                         ))}
                    </tbody>
                 </table>
             </div>
             
             <p className="mt-4 mb-8 text-[15px] leading-[1.6] text-[#8f8a82]">
                Both systems admit shallow surrogate trees with a median of four conditions per rule, but SNF-lite achieves this with fewer, generally better-balanced clusters, which simplifies explanation and naming of phenotypes for clinicians.
             </p>

            <div className="my-8">
                <img
                    src="/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/PortfolioMain/Cluster Sizes.svg"
                    alt="Cluster Sizes"
                    className="block mx-auto max-w-full h-auto"
                />
            </div>
            
             <SectionLead>
                Cluster balance in SNF-lite versus MMSP phenotypes. Each bubble represents the smallest and largest clusters in the most imbalanced stratum for each method. SNF-lite’s biggest imbalance occurs in the low multimorbidity stratum (11.4% vs 58.6%), whereas MMSP shows a much more extreme tail in the high multimorbidity stratum, with a tiny 1.6% cluster alongside a dominant 46.6% cluster.
             </SectionLead>
        </section>
    );
};

export const Concordance: React.FC = () => {
    return (
        <section>
             <h3 className="text-[32px] font-medium mt-20 mb-6 tracking-[0.08em] uppercase text-[#111111]">
                Concordance & Complementarity
            </h3>
            <SectionLead>
                 <p className="mb-4">
                    We next compared MMSP and SNF-lite labelings within each multimorbidity stratum. Cross-tabulations and pairwise Venn diagrams showed that no SNF-lite cluster mapped cleanly onto a single MMSP cluster or vice versa; most clusters shared only partial overlap.
                 </p>
            </SectionLead>

            <p className="font-bold mb-2">Concordance between MMSP and SNF-lite phenotypes</p>
            <SectionLead>
                 Agreement between MMSP and SNF-lite labels was quantified within each multimorbidity stratum using Adjusted Rand Index (ARI) and Normalised Mutual Information (NMI). Values near 0 indicate little more than random overlap; values near 1 indicate near-identical partitions.
            </SectionLead>

             <div className="overflow-x-auto mt-6 mb-8">
                 <table className="w-full border-collapse text-sm text-left">
                    <thead>
                        <tr className="border-b border-[#e3d7c6]">
                            {["Stratum", "N", "MMSP (K)", "SNF (K)", "ARI", "NMI", "Summary"].map(h => (
                                <th key={h} className="py-2.5 px-1 font-medium text-[13px] uppercase tracking-wider text-[#8f8a82]">{h}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                         {[
                             ["Low multimorbidity", "4,183", "4", "3", "0.078", "0.119", "Low concordance; large SNF clusters mix several MMSP phenotypes."],
                             ["Mid multimorbidity", "3,822", "3", "3", "0.141", "0.152", "Highest overlap, but still partial nesting and many-to-many mapping."],
                             ["High multimorbidity", "1,100", "5", "3", "0.098", "0.129", "Low concordance; each SNF cluster draws from multiple MMSP clusters."]
                         ].map((row, i) => (
                             <tr key={i} className="border-b border-[#e3d7c6]">
                                 {row.map((cell, j) => <td key={j} className="py-2.5 px-1">{cell}</td>)}
                             </tr>
                         ))}
                    </tbody>
                 </table>
             </div>

             <div className="max-w-3xl text-[15px] leading-[1.6] text-justify mb-8">
                <p className="mb-4">
                    Concordance metrics were modest <strong>(ARI 0.08–0.14; NMI 0.12–0.15)</strong>, indicating that the two methods capture related but distinct structure in the patient space. Together with the cross-validated prognostic analysis—where both labelings improved discrimination over a clinical base model but SNF-lite yielded the largest ΔC—this suggests that MMSP and SNF-lite provide <strong>complementary phenotypic views</strong> rather than redundant partitions.
                </p>
                <p>
                    We therefore treat SNF-lite phenotypes as the primary clustering solution, with MMSP phenotypes providing a physiology-focused sensitivity and interpretability check within comorbidity strata.
                </p>
             </div>

             {/* Sankey Grid */}
             <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8 mb-20">
                {[
                    { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/reports/figures/sankey_Low_MM.png", caption: "Low multimorbidity (ARI = 0.08)" },
                    { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/reports/figures/sankey_Mid_MM.png", caption: "Mid multimorbidity (ARI = 0.14)" },
                    { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/reports/figures/sankey_High_MM.png", caption: "High multimorbidity (ARI = 0.10)" }
                ].map((img, i) => (
                    <figure key={i} className="m-0 text-center">
                        <img src={img.src} alt={img.caption} className="w-full h-auto block shadow-sm" />
                        <div className="mt-4 text-xs font-semibold tracking-wider uppercase text-[#7c7165]">
                            {img.caption}
                        </div>
                    </figure>
                ))}
             </div>
        </section>
    );
};
