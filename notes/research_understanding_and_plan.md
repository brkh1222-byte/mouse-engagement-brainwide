# mouse-engagement-brainwide — Research Understanding & Plan

Working copy of the Claude Doc (https://claude.ai/artifact/PYBb94penovduagu6xcd1o). Sections are added as they are written. As of 2026-11-24.

## A.1 Core concepts, in plain words

The project asks one thing: when a mouse is about to "zone out" on a task, which parts of its brain already look different before anything appears on the screen?

**The task.** In the International Brain Laboratory (IBL) task, a head-fixed mouse sees a striped patch (a grating) appear on the left or right screen. It turns a small wheel to bring the patch to the centre and gets a drop of water if it is right. Some gratings are bright and easy; some are faint; on 0% contrast trials there is nothing to see, so the mouse must guess.

**The block prior.** The IBL task hides a pattern. For stretches of trials, the grating appears on one side 80% of the time, then the bias flips. Mice learn this and lean towards the likely side on hard trials. Scientists call this belief the "prior". It matters here because a mouse leaning on the prior can look, from the outside, like a mouse that has stopped paying attention.

**Engagement.** Animals, like students in a long lesson, drift in and out of focus. An engaged mouse's choices follow the stimulus closely. A disengaged mouse ignores the stimulus and guesses, keeps picking one side, or stops responding.

**Latent states and the GLM-HMM.** We never see engagement directly; it is "latent" (hidden). A GLM-HMM infers it from choices alone. Think of it as two parts:

- The **GLM** (generalised linear model) is a small formula for how the mouse chooses: how much it weighs the stimulus, its last choice, its last reward, and a side bias.
- The **HMM** (hidden Markov model) says the mouse switches between a few such formulas, and that each one tends to persist for many trials before switching.

The model gives each trial a probability of being in each state, for example 0.95 "engaged". We label a trial only when that probability is high.

**Pre-stimulus activity.** We count how often each neuron fires in a short window *before* the grating appears. That activity cannot be a response to the stimulus, so it reflects the brain's ongoing state: alertness, expectations and readiness to move.

**Decoding.** A decoder is a simple classifier (here, logistic regression) that tries to guess the engagement label from the firing counts of the neurons recorded in one brain region. If it guesses better than chance, that region carries information about engagement. We test on trials the decoder never saw (cross-validation).

**Confounds and controls.** A confound is something else that could explain the result. Three matter most here:

1. **Block prior.** Already encoded across the brain, and partly mixed up with "biased" states.
2. **Movement and arousal.** Fidgeting, whisking and pupil size change activity almost everywhere. A disengaged mouse may simply move differently.
3. **Time in session.** Mice get less thirsty and less engaged late in a session, while recordings also drift slowly. Both change with time, so they can look related even if they are not.

**Null distributions.** To know what "chance" looks like, we repeat the analysis with labels that keep the same slow, sticky structure but cannot be truly linked to this session's neurons, for example the state sequence from a different session. Shuffling single trials is not enough: it breaks the slow structure and makes almost anything look significant.

**Replication.** A result found in one lab's data is stronger if it reappears in data from another lab. Steinmetz et al. (2019) recorded a similar task with the same probe type (Neuropixels) in a different lab, so it is a natural test.
