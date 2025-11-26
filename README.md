## SAXSier Suite

**To view everything at a glance**

An integrated, user-friendly suite forreduction, analysis, and visualization of SAXS and SEC-SAXS data.

The idea is to get a maximum of information through various plots (Guinier, Nomalized Kratky, Volume of Corelation) at a glance.

For documentation proposed all plots are automatically save as image and txt files in dedicated folder. This (at least for me) helps to keep an insightful trace of the analysis.

Stand alone version will be avalaible soon for Windows MacOs and Linux.


[A small **Wiki** is here](https://github.com/JMB-Scripts/SAXSier/wiki)

📖 **Overview**

SAXSier is a software toolkit designed to streamline the Small-Angle X-ray Scattering (SAXS) analysis pipeline. It facilitates buffer subtraction for SEC-SAXS, automated Guinier assessment, molecular weight estimation, and Pair-Distance Distribution $P(r)$ calculations, bridging the gap between raw beamline data and biophysical interpretation.

The suite is built with Python 3.10, PySide6, and Matplotlib, ensuring a modern, responsive interface for both Windows MacOS and Linux

🛠️ **Included Tools**

SAXSier v4.x includes four specialized modules accessible via a central launcher:

**1. Ragtime (v5.x) - SEC-SAXS Analysis**

Designed for Size-Exclusion Chromatography coupled with SAXS.

Unified Visualization: with I(0)vsRg and I(0)vs MW but also for individual frame form Factor, Guinier fit, Kratky plot, and Volume of Correlation analysis in a single window, enabling immediate data quality assessment at a glance.

**2. Sexier (v7.x) - Detailed Structural Analysis**

A comprehensive tool for analyzing single scattering profiles.

Guinier & Kratky: Instant visualization of linearity and folding state.

P(r) Distribution: Real-space analysis using BIFT (Bayesian Indirect Fourier Transform) for $Dmax$.

Molecular Weight: Estimates MW using the Volume of Correlation ($V_c$) and Porod Invariant.

**3. SAXSting (v3.x) - Comparison & Averaging**

Curve Superimposition: Compare multiple datasets visually.

Normalization: View raw or normalized data ($I/I_0$).

it also facilitates the visualization of all key parameters ($R_g$, MW, Kratky) to judge the differences between several scattering curves in a single glance.

Averaging: Statistical averaging of selected curves with error propagation.

**4. SubMe (v3.x) - Buffer Subtraction**

Baseline Correction: Advanced subtraction using linear baselines (drift correction).

Averaging: Standard average buffer subtraction.

Visualization: Real-time preview of the subtracted signal.

⚙️ **Installation**

Clone or download this repository.

Create the environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate saxsier_env
```

Put all scripts in the same folder then run 

Run the launcher:

```bash
python SAXSier-v4.x.py
```

🧪 **Methodology & References**

If you use SAXSier in your research, please cite the original works used within the software:

Guinier Analysis ($R_g$ & $I_0$)

Guinier, A. (1939). La diffraction des rayons X aux très petits angles: application à l'étude de phénomènes ultramicroscopiques. Annales de Physique, 11(12), 161-237.

Molecular Weight (Volume of Correlation)

Rambo, R. P., & Tainer, J. A. (2013). Accurate assessment of mass, models and resolution by small-angle scattering. Nature, 496(7446), 477-481.

P(r) Distribution (BIFT)

The $P(r)$ calculation in Sexier utilizes the Bayesian Indirect Fourier Transform.

Hansen, S. (2000). Bayesian estimation of the pair distance distribution function from small-angle scattering data. Journal of Applied Crystallography, 33(6), 1415-1421.

Glatter, O. (1977). A new method for the evaluation of small-angle scattering data. Journal of Applied Crystallography, 10(5), 415-421.

Hopkins, J. B.  (2024). BioXTAS RAW 2: new developments for a free open-source program for small-angle scattering data reduction and analysis. . Journal of Applied Crystallography (2024), 57, 194-208.

**Special Thanks**: 

Acknowledgment to Jesse B. Hopkins with Raw, by making everything available it helps me a lot .
(for instence, clearly, I had  no idea how to code bift).

📄 **License*** (because github ask for one)

Basically be aware it has been coded by a stupid biochemist (for this matter me), so might be some problem...
And do whatever you want to improve it. 
