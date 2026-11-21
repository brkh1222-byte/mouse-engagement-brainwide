# Research Proposal

## Where in the Brain Is "Zoning Out"? Brain-Wide Pre-Stimulus Signatures of Latent Engagement States in the IBL Brain-Wide Map, with Replication in Steinmetz et al. (2019)

**Author:** Berkh (independent student researcher)
**Adult mentor / co-author:** [to be confirmed]
**Field:** Systems neuroscience
**Planned duration:** 4 weeks
**Compute:** CPU is sufficient; an H100 speeds up the many decoders
**Status:** Proposal, Nov 2026

---

## 1. Summary

Mice performing a visual decision task are not always engaged. In some stretches of trials they ignore the stimulus and guess or follow a bias. Ashwood et al. (2022, *Nat Neurosci*) showed these hidden **engagement states** can be inferred from behaviour alone with a **GLM-HMM** (a hidden Markov model whose states each have their own choice model).

The **International Brain Laboratory (IBL) Brain-Wide Map** (2025, *Nature*) recorded ~75,000 good neurons in ~240 brain regions of 139 mice doing this task. It is open (CC BY 4.0). The engagement state has not been mapped onto this dataset region by region.

This project:
1. infers each trial's engagement state from behaviour;
2. asks which brain regions' **pre-stimulus** activity predicts engagement, controlling for the block prior and movement;
3. tests whether the region map **replicates** in an independent dataset from a different lab (Steinmetz et al. 2019, ~30,000 neurons, 42 regions).

## 2. Background

- **Engagement and arousal** strongly modulate brain-wide activity (Steinmetz 2019 found engagement signals before the stimulus across many regions; Stringer et al. 2019 linked brain-wide activity to movement).
- **GLM-HMM:** Ashwood et al. 2022 found "engaged" and "disengaged/biased" states lasting tens of trials in IBL mice.
- **Confound:** the IBL task has 20/80 stimulus-probability blocks. An independent reanalysis (github.com/anasnadeem30/ibl-glmhmm-individuality) suggests the "biased" states partly track block structure. Any neural analysis must separate *engagement* from *the block prior*, which the IBL companion paper showed is itself encoded brain-wide.

## 3. Prior work and the gap

| Work | What it did | Gap |
|---|---|---|
| IBL Brain-Wide Map 2025, *Nature*, doi:10.1038/s41586-025-09235-0 | Brain-wide encoding of stimulus, choice, feedback and movement | No latent engagement states |
| IBL prior paper 2025, *Nature*, doi:10.1038/s41586-025-09226-1 | Brain-wide encoding of the block prior | Not engagement |
| Ashwood et al. 2022, *Nat Neurosci* | GLM-HMM states from IBL behaviour | No neural data |
| Yin, Melin et al. 2025, *Neuron* | Engagement states in widefield imaging (dorsal cortex) | No Neuropixels; no subcortical regions |
| Steinmetz et al. 2019, *Nature* | Brain-wide engagement signals (42 regions) | Engagement defined by task epoch, not GLM-HMM; never compared with IBL |

**Gap:** No region-by-region Neuropixels map of GLM-HMM engagement state controls for block prior and movement, and none has been tested for cross-lab replication.

**Competition note:** the IBL server lists a protected-dataset tag "2026_Q2_Melin_et_al", which hints at an upcoming IBL paper. Task T1.1 checks bioRxiv for it before work starts.

## 4. Hypotheses (pre-registered before decoding)

- **H1:** Engagement state can be decoded above shuffled-session nulls from pre-stimulus activity in a subset of regions after controlling for block prior and movement.
- **H2:** Decodability will be highest in midbrain/arousal-related and frontal/motor regions (e.g., MRN, SCm, MOs) and lowest in primary sensory cortex. This prediction is directional but exploratory.
- **H3:** The region ranking in IBL correlates positively with the region ranking in Steinmetz 2019, across the regions both datasets share.

## 5. Data

| Dataset | Access | Notes |
|---|---|---|
| **IBL Brain-Wide Map** (2025 release) | ONE-api with the public credentials; AWS `ibl-brain-wide-map-public` (no account) | 459 sessions / 699 insertions (verified live); 139 mice; 75,708 good units; CC BY 4.0 |
| **Steinmetz et al. 2019** | Figshare 9974357 (8.25 GB, CC0) or DANDI 000017 (NWB) | 10 mice, ~30k neurons, 42 regions |

## 6. Method

1. **Behavioural states:** fit a 3-state GLM-HMM (ssm or dynamax) to all IBL sessions, with regressors for stimulus, previous choice, previous reward and bias. Label trials with posterior > 0.8. Do the same for Steinmetz (or define disengagement as late-session miss runs, as a sensitivity check).
2. **Neural features:** pre-stimulus spike counts per region (e.g., −400 to −100 ms before stimulus onset) per trial, via ONE / brainbox.
3. **Covariates:** block side (prior), wheel speed, whisker motion energy, trial number.
4. **Decoding:** per region × session, run L2-logistic decoding of engagement state from population activity, with nested cross-validation and covariates regressed out or included as nuisance. Use the IBL "session-permutation" null approach, which gives correct significance despite slow drifts.
5. **Aggregation:** combine across sessions per region (Fisher/Stouffer plus FDR). Make a brain-region map.
6. **Replication:** Spearman correlation of region effect sizes between IBL and Steinmetz, over the regions both datasets share.

## 7. Timeline

| Week | Tasks |
|---|---|
| 1 | Competition check; ONE setup; download behaviour; fit the GLM-HMM; **commit the pre-registration** |
| 2 | Pull pre-stimulus spike counts and covariates; decoding pipeline on 10 sessions |
| 3 | Full IBL run (GPU optional); Steinmetz replication |
| 4 | Figures; write-up; bioRxiv |

## 8. Venues

- eNeuro, PLOS ONE, Scientific Reports, or Royal Society Open Science (replication angle).
- JEI as the student alternative (needs an adult mentor).
- bioRxiv preprint.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Few disengaged trials in some mice | Restrict to sessions with ≥N disengaged trials (pre-registered); report the counts |
| Block-prior confound | Covariates plus within-block analyses |
| Slow drifts inflate decoding | Session-permutation nulls (IBL method) |
| Upcoming IBL paper overlaps | Check first; differentiate on cross-lab replication |

## 10. Key references

- International Brain Laboratory et al. (2025). A brain-wide map of neural activity during complex behaviour. *Nature*. doi:10.1038/s41586-025-09235-0
- International Brain Laboratory et al. (2025). Brain-wide representations of prior information. *Nature*. doi:10.1038/s41586-025-09226-1
- Ashwood, Z. C. et al. (2022). Mice alternate between discrete strategies during perceptual decision-making. *Nat Neurosci* 25, 201
- Steinmetz, N. A. et al. (2019). Distributed coding of choice, action and engagement across the mouse brain. *Nature* 576, 266
- Stringer, C. et al. (2019). Spontaneous behaviors drive multidimensional, brainwide activity. *Science* 364
- Yin, C., Melin, M. et al. (2025). *Neuron*

> ⚠️ These references were compiled by an AI research assistant. Verify every citation before using it.
