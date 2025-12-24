import React, { ReactNode } from 'react';
import { 
  IntroSection, 
  OverviewSection, 
  VariableSummary, 
  PhenotypingStrategy, 
  VisualStrip, 
  InternalValidation, 
  PrognosticValue, 
  Parsimony, 
  Concordance 
} from './components/Sections';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#f7f1e8] text-[#2c2c2c] pb-32">
      <main className="max-w-[1120px] mx-auto pt-[72px] px-6 md:px-8">
        {/* Top Bar */}
        <header className="flex flex-col md:flex-row justify-between items-baseline mb-20 gap-4">
          <h1 className="text-3xl md:text-[40px] leading-tight tracking-[0.14em] uppercase font-medium text-[#111111]">
            Multimorbidity-Anchored ICU Phenotypes <br />(MAIP)
          </h1>
          <div className="text-sm uppercase tracking-[0.16em] text-[#8f8a82] text-right max-w-[360px] leading-relaxed self-end md:self-auto">
            A Dual-Approach Unsupervised Learning Strategy with <br />
            RAG-Assisted Translation of Surrogate Rules
          </div>
        </header>

        {/* Profile Header */}
        <section className="mb-12">
          <h2 className="text-2xl font-normal mb-2 text-[#111111]">Hari S. Sreedeth</h2>
          <div className="text-base text-[#8f8a82] mb-6">
            Health Data Scientist &nbsp;·&nbsp; Statistical Machine Learning
          </div>

          <h3 className="text-base font-semibold mb-1.5 text-[#111111]">Context</h3>
          <div className="text-[15px] leading-relaxed text-[#2c2c2c] max-w-3xl">
            <p className="mb-4">
              Problem: ML models are accurate but opaque. Doctors don't trust them.
            </p>
            <p>
              Solution: I built a pipeline that uses high-complexity ML to find patterns, but translates them into transparent, rule-based logic (Rulecards) that a human can audit, complete with a QC layer that mathematically guarantees the explanation matches the model.
            </p>
          </div>
        </section>

        {/* Components for Sections */}
        <IntroSection />
        <OverviewSection />
        <VariableSummary />
        <PhenotypingStrategy />
        
        {/* Visual Strips */}
        <VisualStrip 
          title="Physiology Space"
          description="PCA of the acute physiology view (P-view) within each multimorbidity stratum. Points are coloured by MMSP cluster; arrows indicate the main contributing physiological variables."
          images={[
            { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/data/02_clusters/pca_biplot_2d_Low_MM.png", caption: "Low multimorbidity (N = 4,183)" },
            { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/data/02_clusters/pca_biplot_2d_Mid_MM.png", caption: "Mid multimorbidity (N = 3,822)" },
            { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/data/02_clusters/pca_biplot_2d_High_MM.png", caption: "High multimorbidity (N = 1,100)" },
          ]}
        />

        <VisualStrip 
          title="MMSP cluster profiles"
          description="Heatmaps of MMSP clusters within each multimorbidity stratum. Colours show cluster-level z-scores for selected comorbidities and key physiology variables; warmer cells indicate relative enrichment within a cluster."
          images={[
            { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/reports/figures/heatmap_mmsp_Low_MM.png", caption: "Low multimorbidity (N = 4,183)" },
            { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/reports/figures/heatmap_mmsp_Mid_MM.png", caption: "Mid multimorbidity (N = 3,822)" },
            { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/reports/figures/heatmap_mmsp_High_MM.png", caption: "High multimorbidity (N = 1,100)" },
          ]}
          note="Figure 2: MMSP cluster profiles by multimorbidity stratum. For each low, mid and high multimorbidity group, the heatmaps display z-scored cluster averages of selected comorbidities and age/APS, highlighting relative enrichment and depletion across MMSP phenotypes."
        />

        {/* SNF-Lite Text Section */}
        <section className="mb-16">
           <div className="max-w-3xl">
            <h3 className="font-bold text-lg mb-2">SNF-lite: Multi-View Fusion Phenotypes</h3>
            <p className="mb-6 leading-relaxed">
              In parallel, I build a multi-view similarity network using all three views: 
              comorbidity (C), physiology (P) and socio-contextual features (S). For C and 
              S I compute Gower similarities; for the scaled P view I use an RBF kernel. 
              A lightweight Similarity Network Fusion routine iteratively combines these 
              into a single fused affinity matrix that emphasises similarities supported 
              by multiple views. Spectral clustering on this fused graph yields a global 
              set of ICU phenotypes. K is chosen using the eigengap heuristic plus the 
              same stability and internal metrics as MMSP, and competing solutions are 
              compared on prognostic performance and clinical interpretability.
            </p>
            <div className="my-8">
              <img
                src="/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/PortfolioMain/SNF-Lite.svg"
                alt="SNF-Lite"
                className="block mx-auto max-w-full h-auto"
              />
              <p className="text-sm text-[#7c7165] mt-2 font-medium">Figure 2: Similarity Fusion Network-Lite (SNF-Lite) workflow.</p>
            </div>
            <div className="bg-white/50 border-l-4 border-[#e3d7c6] p-4 text-sm text-[#6b6b6b] mb-8">
              <span className="font-bold mr-2">Figure 2</span>
              SNF-lite workflow: multimorbidity strata feed C, P and S views into a lightweight similarity network fusion step, which produces a fused affinity matrix and spectral-clustering–derived phenotypes across low, mid and high burden groups.
            </div>
          </div>
        </section>

        <VisualStrip 
          title="SNF-lite patient similarity networks"
          description="k-nearest-neighbour graphs derived from the fused multi-view similarity matrix within each multimorbidity stratum. Each node is a patient, edges connect highly similar patients, and colours indicate SNF-lite cluster membership."
          images={[
            { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/reports/figures/snf_network_Low_MM.png", caption: "Low multimorbidity (N = 4,183)" },
            { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/reports/figures/snf_network_Mid_MM.png", caption: "Mid multimorbidity (N = 3,822)" },
            { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/reports/figures/snf_network_High_MM.png", caption: "High multimorbidity (N = 1,100)" },
          ]}
          note="Figure 3: SNF-lite patient similarity networks by multimorbidity stratum. Each panel shows the k-nearest-neighbour graph constructed from the fused multi-view similarity matrix; nodes are ICU admissions and edges represent strong pairwise similarities. Node colours denote SNF-lite clusters, which form visually distinct communities within the fused patient graph."
        />

        <VisualStrip 
          title="SNF cluster profiles"
          description="Heatmaps of SNF clusters within each multimorbidity stratum. Colours show cluster-level z-scores for selected comorbidities and key physiology variables; warmer cells indicate relative enrichment within a cluster."
          images={[
            { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/reports/figures/heatmap_snf_Low_MM.png", caption: "Low multimorbidity (N = 4,183)" },
            { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/reports/figures/heatmap_snf_Mid_MM.png", caption: "Mid multimorbidity (N = 3,822)" },
            { src: "/Users/harisreedeth/Desktop/D/personal/ProjectMAIP/reports/figures/heatmap_snf_High_MM.png", caption: "High multimorbidity (N = 1,100)" },
          ]}
        />

        <InternalValidation />
        <PrognosticValue />
        <Parsimony />
        <Concordance />

      </main>
    </div>
  );
};

export default App;
