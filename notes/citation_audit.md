# Citation audit of PROPOSAL.md (2026-11-24)

Checked against Crossref and full text (Europe PMC / bioRxiv).

| # | Status | Corrected reference | Issue |
|---|---|---|---|
| 1 | Minor error | International Brain Laboratory et al. (2025). A brain-wide map of neural activity during complex behaviour. *Nature*, 645(8079), 177–191. https://doi.org/10.1038/s41586-025-09235-0 | "~240 regions" → 279 areas covered; 201 regions in the canonical analysis set (62,857 neurons). 139 mice, 12 labs, 459 sessions, 699 insertions, 621,733 units, 75,708 good units: correct. |
| 2 | Minor error | Findling, C., Hubert, F., International Brain Laboratory, … Pouget, A. (2025). Brain-wide representations of prior information in mouse decision-making. *Nature*, 645(8079), 192–200. https://doi.org/10.1038/s41586-025-09226-1 | Title incomplete. Null is a **pseudosession** test (200 pseudosessions), not "session permutation". Fisher combination; BH 1% FDR only on combined ephys+WFI map. |
| 3 | Minor error | Ashwood, Z. C., Roy, N. A., Stone, I. R., International Brain Laboratory, Urai, A. E., Churchland, A. K., Pouget, A., & Pillow, J. W. (2022). Mice alternate between discrete strategies during perceptual decision-making. *Nature Neuroscience*, 25(2), 201–212. https://doi.org/10.1038/s41593-021-01007-z | Pages 201–212. IBL fits used only the **first 90 unbiased trials** of each session. Median dwell 24/13/12 trials (engaged/left/right). |
| 4 | Minor error | Steinmetz, N. A., Zatka-Haas, P., Carandini, M., & Harris, K. D. (2019). Distributed coding of choice, action and engagement across the mouse brain. *Nature*, 576(7786), 266–273. https://doi.org/10.1038/s41586-019-1787-x | Engagement defined by task vs passive (engagement index), separating Go vs Miss; late-session miss runs were the *exclusion* rule. Figshare 9598406 is the paper's own deposit (CC BY 4.0, not CC0); 9974357 is the ONE-format copy. 10 mice, 39 sessions, 29,134 neurons (22,458 in 42 regions). |
| 5 | Correct | Stringer, C., Pachitariu, M., Steinmetz, N., Reddy, C. B., Carandini, M., & Harris, K. D. (2019). Spontaneous behaviors drive multidimensional, brainwide activity. *Science*, 364(6437), eaav7893. https://doi.org/10.1126/science.aav7893 | — |
| 6 | Incomplete | Yin, C., Melin, M. D., Rojas-Bowe, G., Sun, X. R., Couto, J., Gluf, S., Kostiuk, A., Musall, S., & Churchland, A. K. (2025). Spontaneous movements and their relationship to neural activity fluctuate with latent engagement states. *Neuron*, 113(18), 3048–3063.e5. https://doi.org/10.1016/j.neuron.2025.06.001 | Title missing. Auditory click task (not IBL), widefield dorsal cortex. |
| 7 | Not peer-reviewed | Nadeem, A. (2026). ibl-glmhmm-individuality [GitHub repository]. https://github.com/anasnadeem30/ibl-glmhmm-individuality | Informal; cite at most as an observation. Use Mohammadi et al. 2025 (Nat Commun, 10.1038/s41467-025-66738-0) as the peer-reviewed source on block in GLM-HMM. |

## Tools / data facts
- ONE-api ≥3.6: the shared public login was removed; public users register a free Open Alyx account (openalyx.internationalbrainlab.org/signup). Old shared password: untested.
- AWS bucket `ibl-brain-wide-map-public`: correct (anonymous listing works).
- `ssm`: install from GitHub (PyPI version stale). `dynamax` 1.0.3 active. `iblatlas`: Beryl = 308 regions; Swanson plotting available.
- DANDI 000017: NWB, 39 assets, 14.7 GB, CC BY 4.0.
