# MSA-Theory: Canonical Correlation Analysis for MSA

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Research](https://img.shields.io/badge/Focus-Research%20%26%20Theory-purple)
![Method](https://img.shields.io/badge/Core-Canonical%20Correlation%20Analysis-orange)

Research-oriented repository for studying the mathematical foundations of MSA, with a strong emphasis on Canonical Correlation Analysis (CCA), geometric interpretation, and derivation-driven reasoning.

## Overview

`MSA-theory` is the conceptual and analytical track of the project.  
The repository is used to:

- consolidate CCA theory from papers and classical multivariate statistics,
- derive and validate the covariance/Cholesky/SVD formulation used in implementation,
- document geometric intuition for canonical variables,
- maintain notes, report chapters, and exploratory prototypes.

## Research Focus

- CCA as a core bridge between paired multivariate views.
- Covariance-matrix formulation and numerical stability considerations.
- Canonical weights, loadings, variates, and interpretability.
- Geometry of correlation (`r = cos(phi)`) in variable-space diagrams.
- Hypothesis building for downstream MSA design decisions.

## CCA in MSA

Within this repository, CCA is treated as both:

1. **A mathematical object**: derive canonical directions from covariance structure and matrix factorizations.
2. **A design tool**: reason about which transformed components should be retained for practical MSA pipelines.

The current code and notes focus on:

- paired datasets `X1` and `X2`,
- canonical correlation spectrum across components,
- comparison between theoretical expectations and computed outputs,
- geometry-based explanation artifacts for communication and teaching.

## Repository Structure

```text
MSA-theory/
├── core.py                         # CCA implementation via covariance + Cholesky + SVD
├── main.py                         # CLI workflow for running and inspecting CCA
├── utils.py                        # Data I/O, summary statistics, and preprocessing helpers
├── demo_app.py                     # Interactive Streamlit demo for theory exploration
├── visualize_explain_streamlit.py  # Additional explanatory visual flows
├── geometry_cca/
│   ├── __init__.py
│   └── geometry.py                 # Geometric CCA visualizations and angle interpretation
├── report/
│   ├── main.tex                    # Main report document
│   ├── Trangbia.tex
│   ├── chapter_part_A/             # Theory chapters and derivations
│   ├── chapter_part_B/             # Problem framing/application context
│   └── chapter_part_D/             # Appendix materials
├── example-test/notebooks/
│   └── CCA.ipynb                   # Notebook-style exploration
├── AQ_X1.csv                       # Sample paired dataset view X1
├── AQ_X2.csv                       # Sample paired dataset view X2
├── requirements.txt
└── README.md
```

## Key Questions

- Which derivation of CCA is most transparent for MSA communication and reproducibility?
- How do canonical correlations and loadings change under standardization choices?
- When does geometric interpretation improve model/debug intuition in practice?
- What assumptions in theory are most likely to break in noisy real-world paired data?

## Methodology

1. **Literature pass**: collect core CCA references and implementation variants.
2. **Derivation pass**: map equations into matrix operations used in `core.py`.
3. **Validation pass**: compare derived quantities vs. computed canonical outputs.
4. **Interpretation pass**: connect algebraic results to geometry and domain meaning.
5. **Transfer pass**: export actionable insights to the implementation repository.

## References

This repository is intended to track references from:

- classical multivariate analysis texts (CCA foundations),
- paper notes and internal report chapters in `report/`,
- implementation-oriented CCA resources linked to numerical linear algebra.

If you contribute, please include bibliographic metadata (title, authors, year, venue, link/DOI) in your notes or PR description.

## Roadmap

- [ ] Expand literature review matrix (assumptions, methods, limitations).
- [ ] Formalize derivation notes into reproducible theorem-to-code mappings.
- [ ] Add synthetic data experiments for edge-case validation.
- [ ] Improve consistency between report notation and source code variables.
- [ ] Add research issue templates (question, hypothesis, expected evidence).

## Contributing

Contributions are welcome from researchers, students, and collaborators.

- Open an issue with one of: **theory clarification**, **derivation check**, **reference addition**, **visual explanation**.
- Prefer small, reviewable pull requests with clear rationale.
- Keep notation consistent and explicitly define symbols when introducing equations.
- For code changes, include a short note on how the change supports research interpretation.
