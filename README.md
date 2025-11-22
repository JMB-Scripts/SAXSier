SAXSier Suite v4.x

An integrated, user-friendly suite forreduction, analysis, and visualization of SAXS and SEC-SAXS data.
The idea is to get a maximum of information at a glance.

📖 Overview

SAXSier is a unified software toolkit designed to streamline the Small-Angle X-ray Scattering (SAXS) analysis pipeline. It facilitates precise buffer subtraction, automated Guinier assessment, molecular weight estimation, and Pair-Distance Distribution $P(r)$ calculations, bridging the gap between raw beamline data and biophysical interpretation.

The suite is built with Python 3.10, PySide6, and Matplotlib, ensuring a modern, responsive interface for both Windows and macOS.

🛠️ Included Tools

SAXSier v4.x includes four specialized modules accessible via a central launcher:

1. Ragtime (v5.x) - SEC-SAXS Analysis

Designed for Size-Exclusion Chromatography coupled with SAXS.

Unified Visualization: Automatically generates and displays the Form Factor, Guinier fit, Kratky plot, and Volume of Correlation analysis in a single window, enabling immediate data quality assessment at a glance.

Automated Guinier Analysis: Scans thousands of frames to find linear regions.

Quality Control: Plots $R_g$, $I(0)$, and Molecular Weight (MW) vs. Frame.

Peak Analysis: Automatically detects peaks and valid ranges.

2. Sexier (v7.x) - Detailed Structural Analysis

A comprehensive tool for analyzing single scattering profiles.

Guinier & Kratky: Instant visualization of linearity and folding state.

P(r) Distribution: Real-space analysis using BIFT (Bayesian Indirect Fourier Transform).

Molecular Weight: Estimates MW using the Volume of Correlation ($V_c$) and Porod Invariant.

3. SAXSting (v3.x) - Comparison & Averaging

Facilitates the visualization of all key parameters ($R_g$, MW, Kratky) to judge data quality in a single glance.

Curve Superimposition: Compare multiple datasets visually.

Normalization: View raw or normalized data ($I/I_0$).

Averaging: Statistical averaging of selected curves with error propagation.

4. SubMe (v3.x) - Buffer Subtraction

Baseline Correction: Advanced subtraction using linear baselines (drift correction).

Averaging: Standard average buffer subtraction.

Visualization: Real-time preview of the subtracted signal.

⚙️ Installation

Prerequisites

Miniforge

Setup

Clone or download this repository.

Create the environment using the provided environment.yml file:

conda env create -f environment.yml


Activate the environment:

conda activate saxsier_env


Run the launcher:

python SAXSier-v4.py


🧪 Methodology & References

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

Special Thanks: 

Acknowledgment to Jesse B. Hopkins with Raw, for the incredible work with raw that helps a lot the small angle communauty  (clearly I had  no idea how to code that).
The implementation of BIFT within the Sexier module allows for the objective determination of the maximum dimension ($D_{max}$) and the smoothing parameter ($\alpha$) without manual intervention. This robust mathematical framework is essential for generating reliable Pair-Distance Distribution functions ($P(r)$) from noisy experimental data.

📄 License (because they ask fo one)

Basically be aware it has been code by a stupid biochemist for this matter me...

SAXSier is distributed under the MIT License.


