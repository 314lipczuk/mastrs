
#let PAGE-MARGIN = 2.5cm
#let TEXT-WIDTH = 16.0cm // = 21.0cm (A4) − 2 × 2.5cm.  Must match the figures.
#let WIDE-WIDTH = 24.0cm // landscape float standard (W_WIDE = 9.40 in). Rare.

#let BODY-FONT = ("Libertinus Serif", "Linux Libertine", "Georgia", "Times New Roman")
#let SANS-FONT = ("Inter", "Helvetica Neue", "Helvetica", "Arial")
#let MONO-FONT = ("JetBrains Mono", "Menlo", "DejaVu Sans Mono")

#set document(title: "Deep model-predictive control of ERK signalling in single cells")

#set page(
  paper: "a4",
  margin: PAGE-MARGIN, // ← see the warning at the top of this file
  number-align: center,
  numbering: "1",
)

#set text(font: BODY-FONT, size: 11pt, lang: "en")
#set par(justify: true, leading: 0.72em, first-line-indent: 1.2em)

#set heading(numbering: "1.1")
#show heading: set text(font: SANS-FONT, weight: "semibold")
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  block(above: 1.6em, below: 1.0em, text(size: 17pt, it))
}
#show heading.where(level: 2): it => block(above: 1.4em, below: 0.7em, text(size: 13pt, it))
#show heading.where(level: 3): it => block(above: 1.1em, below: 0.5em, text(size: 11.5pt, it))

#set figure(gap: 0.9em)
#show figure.caption: it => block(
  width: 100%,
  align(left, text(font: SANS-FONT, size: 9pt, {
    text(weight: "semibold", [#it.supplement #context it.counter.display(it.numbering)])
    [ · ]
    it.body
  })),
)

#show raw: set text(font: MONO-FONT, size: 9.5pt)
#set table(stroke: 0.5pt + luma(70%))

// ── Figure helpers ───────────────────────────────────────────────────────────
//
// reference figures by LABEL (`@fig-heterogeneity`), never by number.
// The files in figures/ are mid-renumbering — fig14/15/16 each name two
// different figures, and 02–06 and 12 do not exist. Labels make that
// renumbering free and keep filenames out of the prose.

// Which rendering of each figure to place. Every figure exists in figures/ as
// both .pdf (vector) and .png (300 dpi), and BOTH are exactly 16.0 cm wide, so
// this is a free swap — one line changes the whole document.
//
//   ".png"  works on every Typst version. 300 dpi, i.e. print quality.
//   ".pdf"  vector, but needs Typst >= 0.14 (0.13 errors "unknown image
//           format"). Installed here is 0.13.1; `brew upgrade typst` gets 0.15,
//           after which flipping this line is the only change needed.
#let FIG-EXT = ".png"

/// Place a figure at the text width (1:1, no scaling).
/// `path` is relative to figures/, with or without an extension.
#let thesisfig(path, caption, label-name) = {
  let p = if path.contains(".") { path } else { path + FIG-EXT }
  [#figure(
      image("figures/" + p, width: 100%),
      caption: caption,
      kind: image,
    ) #label(label-name)]
}

/// Landscape float for the rare 24 cm figure. Rotated so the page turns.
#let thesisfig-wide(path, caption, label-name) = {
  let p = if path.contains(".") { path } else { path + FIG-EXT }
  page(flipped: true, [#figure(
      image("figures/" + p, width: 100%),
      caption: caption,
      kind: image,
    ) #label(label-name)])
}

// ── Drafting aids ────────────────────────────────────────────────────────────

/// Inline note to self. Visible on purpose; grep for TODO before submitting.
#let todo(body) = text(fill: rgb("#b3261e"), weight: "semibold", [[TODO: #body]])

/// A figure that does not exist yet. Reserves the slot, the label and the
/// caption so surrounding prose can already reference it.
#let figure-placeholder(caption, label-name, height: 4cm) = {
  [#figure(
      block(
        width: 100%,
        height: height,
        fill: luma(96%),
        stroke: (paint: rgb("#b3261e"), thickness: 0.8pt, dash: "dashed"),
        inset: 10pt,
        align(center + horizon, text(
          font: SANS-FONT,
          size: 10pt,
          fill: rgb("#b3261e"),
          [FIGURE NOT YET MADE],
        )),
      ),
      caption: caption,
      kind: image,
    ) #label(label-name)]
}

// ── Title page ───────────────────────────────────────────────────────────────

#page(numbering: none, {
  set align(center)
  v(5cm)
  text(font: SANS-FONT, size: 20pt, weight: "bold")[
    Deep model-predictive control of ERK signalling in single cells
  ]
  v(1.2cm)
  text(size: 13pt)[Przemysław Pilipczuk]
  v(0.4cm)
  text(size: 11pt, style: "italic")[Master's thesis]
  v(0.3cm)
  text(size: 11pt)[Institute of Cell Biology, University of Bern]
  v(1.2cm)
  text(size: 10.5pt)[Supervised by: Dr. Maciej Dobrzyński, Prof. Olivier Pertz]
  v(1fr)
  text(size: 10.5pt)[#todo[submission date]]
})

#counter(page).update(1)
#outline(depth: 2, indent: auto)

// ═════════════════════════════════════════════════════════════════════════════
//  ABSTRACT
// ═════════════════════════════════════════════════════════════════════════════

//   deep learning for prediction task on single-cell trajectories; We then try
//   to use it for closed loop control in Live experiments. Then investigate
//   whether a learned controller distinguishes distinct stimulation strategies
//   based on its internal representation. We investigate the limits of
//   controllability of such system.
#heading(numbering: none, outlined: true)[Abstract]

In this work, I aim to create a practical method of controlling ERK signalling dynamics 
on a single-cell level in a live experiment using optogenetic stimulation. 
Deep learning model is used for forecasting future ERK levels, 
utilising MDN head for communicating uncertainty. Predicive model is then integrated into a 
Model Predictive Control (MPC) controller, and used to steer experiments in real time. 
We demonstrate ... #todo[what? ]


#todo[write the abstract last]

// ═════════════════════════════════════════════════════════════════════════════
//  INTRODUCTION
// ═════════════════════════════════════════════════════════════════════════════

= Introduction

//   - why we care about ERK dynamics; cell fate; knowing the components but not
//     how they are connected
//   - single-cell heterogeneity — the locality argument: why predicting average
//     behaviour is not enough to understand the system
//   - ODE models: expert construction, hard to scale, good at averages, costly
//     to fit per cell, parameter non-identifiability, no good way to model
//     CHANGES in how a cell responds — which we would rather learn from data
//   - closed-loop control as a way of interacting with the system more
//     meaningfully (Khammash; FARO)
//   - what a closed loop needs: a predictor, a controller mapping prediction to
//     action, and a per-cell objective

== ERK dynamics and cell fate

Dynamics of ERK have been shown to influence cell fate.

#todo[continue — per the writing strategy, first sentence of each paragraph
first, then fill the paragraphs out]

== Why the population average is not enough

A standard problem in studying a biological system is that while a given
quantity might be easily modelled at the level of a summary statistic of a
population, the same does not hold when talking about an individual.

// brief: the signalling events, fates, symmetry breaking, etc. are
// individual-level events.

#thesisfig(
  "heterogeneity",
  [*The population average is not a cell.* 7,141 cells receiving the same pulse
   train, sorted by response. The response is continuously distributed over more
   than a two-fold range rather than falling into classes, and 16% stay within
   10% of their own baseline. Data: experiment #raw("bo_v8"), 78 fields, pulses
   every five minutes. Because that experiment swept per-pulse exposure between
   fields (280--1753 ms), the dose-controlled number is the spread measured
   inside a single field, p90#sym.minus#h(0.1em)p10 #sym.eq 1.73, against 2.00
   pooled across fields.],
  "fig-heterogeneity",
)

#todo[this figure is a premise, not a result: it motivates why a per-cell model
is needed at all. It returns to Results only if E2 runs, where the same
statistic measured under closed-loop control becomes the comparison. Indicative
and cross-plate: the closed-loop within-field spread in v21 is 1.02 against this
1.73, i.e. roughly 40% narrower.]

== Mechanistic models

Using ODE or PDE models of a signalling cascade together with standard
statistical machine-learning techniques, we can fit a model and use it to
generate predictions about future states of a system.

// brief: ODE model problems — expert knowledge, costly fitting, modelling
// changes in how a cell responds, explicit need for multiple levels of
// phenomena encoded, parameter non-identifiability.

== A data-driven alternative

While the mechanistic approach can test our understanding of the theory behind a
biological system, another approach is to use data-driven techniques to predict
the system's behaviour, and to learn to control it.

// brief: not explaining mechanisms, but uncovering more complex behaviour and
// learning through interactions.

== Closed-loop control as an instrument

#todo[closed-loop control literature — Khammash, FARO, the preceding grant]

== What a model-predictive loop requires

A closed loop of this kind needs three things: a predictor, a controller that
turns a prediction into an action, and an objective function defined at the
level of the single cell.


= Materials and methods

// Prose below is ported verbatim from "Master thesis.md" (Obsidian). Where that
// document carries an outline bullet rather than text, the bullet is kept as a
// comment so the brief travels with the section.

== OptoEGFR cell line and culture

A previously established NIH3T3 mouse fibroblasts cell line (ATCC CRL-1658) stably
expressing optoEGFR-mCitrine together with ERK-KTR-mScarlet3 and H2B-miRFP670nano3 was
used for all experiments. OptoEGFR and the downstream biosensors were expressed under
CAG promoters. Cells were grown and maintained in Dulbecco's Modified Eagle's Medium —
high glucose (Sigma-Aldrich \#D5671), supplemented with 10% (v/v) fetal bovine serum,
2% L-Glutamine (stable, 200 mM) and 1% penicillin/streptomycin at 37 °C and 5% CO#sub[2].
Mycoplasma contamination was routinely assessed by PCR.

#figure-placeholder(
  [The biological system. Blue light drives optoEGFR; ERK activity is read out from the
   ERK-KTR translocation reporter as the cytoplasm-to-nucleus ratio.],
  "fig-wetware",
  height: 5cm,
)

// Outline: lack of the external part of the natural receptor — what that gives us
// (a synthetic system / poking device for the rest of the pathway); spatial effects
// are less relevant here but the method extends to problems with a spatial component
// (this may belong in the introduction — it is argumentative).

== Live-cell microscopy

For imaging, cells were cultured in 96-well glass-bottom plates (Cellvis, \#P96-1.5H-N)
and starved overnight in FluoroBrite-based starvation medium containing 0.5% fetal calf
serum, 2% L-glutamine, 0.5% BSA and 1% penicillin/streptomycin. Live-cell imaging was
then performed at 37 °C and 5.2% CO#sub[2].

Model training data was acquired using a Nikon Eclipse Ti inverted microscope equipped
with a Lumencor SPECTRA X LED light engine, an Andor Zyla 4.2 sCMOS camera (2×2 binning)
and a Nikon Plan Apo 20×/0.75 NA objective. Images were acquired at 16-bit depth with a
temporal resolution of one frame per minute. Fluorescence imaging on this setup was
performed using the following excitation/emission configurations: *H2B-miRFP670nano3*
640 nm LED, Lumencor 645/30x excitation filter, Chroma 89100bs dichroic mirror and
Chroma ET705/72m emission filter; *ERK-KTR-mScarlet3* 555 nm LED, Lumencor 575/25x
excitation filter, Chroma 89100bs dichroic mirror and Chroma ET632/60m emission filter;
and *optoEGFR-mCitrine* 508 nm LED, Lumencor ET500/20x excitation filter, Chroma 69008bs
dichroic mirror and Chroma ET535/36m emission filter. Activation of the optogenetic
construct optoEGFR was done using either a 470 nm LED with a 470/10x excitation filter
(most experiments were performed at 10% power, corresponding to approximately 340 µW at
sample) or a 470/24x excitation filter (3% LED power corresponding to approximately
1310 µW at sample), both through a Chroma 86100bs dichroic mirror.

The targeted stimulation experiment was done on a Nikon Eclipse Ti2 inverted microscope,
equipped with a Lumencor SPECTRA X LED light engine for stimulation and imaging
mScarlet3 and miRFP670nano3, a Lumencor CELESTA Laser light source for imaging mCitrine,
both captured using a Teledyne Kinetix (2×2 binning) at 16-bit depth attached to a Crest
CICERO in widefield mode. The objective used was a CFI Plan Apochromat lambda 20×/0.8 NA.
Fluorescence imaging was performed using the following excitation/emission
configurations: *H2B-miRFP670nano3* 640 nm LED, Lumencor 645/30x excitation filter, two
consecutive Semrock FF421/491/567/659/776-DI01 dichroic mirrors and FF01-441/511/593/684/817-25
emission filter; *ERK-KTR-mScarlet3* 555 nm LED, Lumencor 575/25x excitation filter, two
consecutive Semrock FF421/491/567/659/776-DI01 dichroic mirrors and Chroma 59022m
emission filter; and *optoEGFR-mCitrine* 477 nm Laser, Semrock FF01-391/477/549/639/741
excitation filter, two consecutive Semrock FF421/491/567/659/776-DI01 dichroic mirrors
and Semrock FF01-511/20-25 emission filter.

Targeted optogenetic stimulation of optoEGFR was done using the 470 nm LED of the
Spectra X LED Light Engine with a 470/40 excitation filter (at 10% power, corresponding
to 3499 µW at sample) through a Mightex Polygon 1000 DMD and one Semrock
FF421/491/567/659/776-DI01 dichroic mirror.

== Automated acquisition and stimulation

Image acquisition and optogenetic stimulation were controlled using the FARO software
framework. Blue-light stimulation was controlled by illumination intensity, exposure
duration and timing relative to image acquisition. For the targeted experiments, the DMD
was used to apply the cell-defined spatial and temporal optoEGFR stimulation pattern
while ERK-KTR and H2B fluorescence were recorded.

== Image analysis

Cell nuclei were segmented using cellpose (version 4) with a custom-trained model.
Cytoplasmic ring masks were generated by binary dilation of nuclear masks by four pixels
using scikit-image. Nuclear area, centroid position and median ERK-KTR fluorescence
intensities in nuclear and cytoplasmic compartments were extracted for each cell. ERK
activity was quantified as the cytoplasm-to-nucleus fluorescence ratio (CNR). Cell
identities were tracked across frames using trackpy (version 0.8), with a maximum
allowed displacement of 50 pixels between consecutive frames.

== Feature engineering

Many feature engineering and augmentation methods were contemplated, which can be
clustered into distinct groups by their motivation: 1) enrichment of stimulation
features, 2) spatial/population features, 3) self-learned image features.

Stimulation features are first collapsed from their distributed representation between
multiple columns (`stim_exposure`, `stim_power`, and a hidden, experiment-level variable
of which stimulation hardware and light filters were used) into a single variable of
energy received by the cell per unit area (radiant exposure), calculated based on
#todo[method] and calibration curves of each of the microscope setups used in generating
the training data.

This feature, referred to henceforth as `fluence`, allowed us to compare data from
multiple acquisition setups. We then used it as a base to calculate auxiliary summary
statistics, that were fed to the model in an attempt to improve learning efficiency.
Among such summary statistics were:

- time since last pulse
- exponential moving average of stimulation across a short time window
- exponential moving average of stimulation across a long time window
- number of pulses in the last $N$ frames
- integral of fluence for this cell since the start of the experiment
- OLS slope of fluences over the last $N$ frames

Features regarding crowding at the level of the single cell and the field of view were
used. FOV-level features count the number of cells found by segmentation, and the
single-cell neighbourhood feature counts for each cell the number of other cells nearby.

After extensive testing and with a bigger dataset, we noticed that the significance of
hand-engineered features diminishes. In the end, a minimalistic set of features was used,
encoder inputs consisting of:

- CNR
- nuclear area
- stimulation (encoded as fluence)
- count of other cells within a radius of 200 px
- optoRTK expression

Out of those mentioned above, the optoRTK expression is worth mentioning in more detail.
The biosensor used for this readout has an overlapping activation spectrum with our
Cry2 — this means that whenever we want to do a readout of optoRTK expression, we also
activate the MAPK/ERK pathway. The reasonable assumption here is that expression of the
optoRTK receptor is something that changes on a much larger timescale than the MAPK
dynamics, and therefore the strategy we chose was to obtain the optoRTK expression value
for each cell before the experiment starts, wait for the MAPK system to reset, and then
start the experiment. Additionally, we image the mCitrine one last time at the end of the
experiment. This serves no purpose in terms of control, but is a useful metric when
attempting to explain behaviour of the cells during the control task.

#thesisfig(
  "expression-normalisation",
  [optoRTK expression is ranked within a session rather than rescaled, because sessions
   differ in the shape of the distribution and not only in its scale.],
  "fig-expression-norm",
)

== Dataset

Data used for training the predictive model was obtained by integrating a set of
experiments with various purposes into a single dataset. Out of those, we can factorise
the component datasets by the experiment:

- short experiments with distinct hardwired patterns of stimulation with varying
  `stim_exposure`, serving the characterisation of the optogenetic construct;
- Bayesian optimisation experiments for finding an optimal pattern of stimulation in
  order to induce oscillation;
- a bulk experiment for parametrising input space, consisting of a sweep of parameter
  space, generating 96 distinct patterns of stimulation;
- a single, long experiment (about 20× the length of all others).
  #todo[no experiment in the corpus matches this — the longest is 210 frames. Either it
  was dropped, or this is conflating the training corpus with the serving runs]

#thesisfig(
  "dataset-overview",
  [The training corpus: where the data comes from, one example trajectory with its
   delivered exposures, and the population response of one experiment from each family.],
  "fig-dataset",
)

All this data was filtered using a standard procedure involving removing cells that were
segmented but not alive, missegmented cells, and visual anomalies resulting in
non-meaningful features.

This final composition of all the experiment data resulted in a dataset of 72,441 cells
and 6.63 M frames.

#todo[summary table of every training experiment and every live run, with scalars and
each run's fate. Raw material: `materials/serving_runs_summary.parquet`]

== Model architecture

A deep learning model was designed for the CNR prediction task, with the goal of being
quick enough to be run in real time for many cells in parallel.

#figure-placeholder(
  [The predictive model: a full-history LSTM encoder, an MLP trunk, and a mixture-density
   prediction head.],
  "fig-model-schematic",
  height: 5cm,
)

This motivated the decision to split the model into three parts: encoder, trunk and
prediction head. The encoder is an LSTM network that encodes all the features from
previous timepoints of a given cell, and returns a hidden state vector. When running in
real time, the encoder hidden state from time $t-1$ is persisted in memory. This way,
when the microscope provides us with the next frame, we can run a single encoder step on
our hidden state and immediately obtain the next state, without re-encoding the entire
past again. After the encoder, the state vector is passed through an MLP trunk, and then
given to a prediction head.

Being able to take into account model uncertainty during the control task was deemed
important, and so the predictive architecture was evaluated for the best way of
quantifying uncertainty. We evaluated multiple options for achieving this goal. A model
ensemble can provide a notion of uncertainty by inspecting the variance of ensemble
predictions. Similar effects can be achieved by a technique called Monte Carlo dropout —
here we use a single model, trained with dropout, but while in ordinary use of this
technique the dropout layer is turned off in evaluation mode, here we use it to sample
from a distribution of plausible predictions from different sub-configurations of the
model. Finally, a mixture density network approach was tested, where the network outputs
a mixture of Gaussians. Each Gaussian consists of three features: mean, variance and
weight, which we constrain into meaningful values. The number of Gaussians used is a
hyperparameter, and can help us encode predictions for a non-homogeneous population. This
prediction head is trained using negative log likelihood loss on the running minibatch.

The prediction head outputs a single future CNR value at $t+1$.

#todo[FiLM conditioning; autoregressive unrolling and the predictive horizon in the
training task — and decide whether the autoregressive paragraph lives here or in
training methodology, where a version of it already exists]

== Training methodology

A single forward pass through the model results in an uncertainty-aware prediction of
CNR at time $t+1$. Since our acquisition window was one minute, and interesting ERK
dynamics have a resolution of multiple minutes, we extended the model's prediction by
autoregressive unrolling on its own predictions. The model, given features from the past,
makes a single prediction (a forecast one step into the future), then appends this
prediction to the history and generates the next forecast, effectively looking two steps
ahead. This unrolling can be done to an arbitrary depth. The number of autoregressive
steps done per single prediction is referred to as the predictive horizon henceforth.

Choice of a predictive horizon is made based on the characteristics of the biological
phenomenon we are controlling and the throughput needs of our control task. The horizon
must be large enough to capture dynamical features of the controlled system, while being
small enough to be evaluated quickly in a live experiment. It is also possible to run
training with a varying time horizon. The reasoning for such a design choice is that we
convey the fact that we are interested in phenomena at all temporal scales of the system.
The problem with this approach is that it makes training noisy and unstable — there is no
meaningful way to assign datapoints to horizon lengths, so they must be assigned at
random, which means that sometimes datapoints with little long-range dynamics will be
used to learn long-range dynamics, and the frequency of such assignments will vary
between runs.

For our system, we chose a predictive horizon of 30 frames, each one minute apart. An
important factor in slicing the original single-cell tracks into datapoints for the model
to train on is how to deal with reusing data from a track. From the statistical point of
view, we would like to have datapoints that are completely decorrelated from each other
and can be viewed as i.i.d. From the practical point of view, it is quite costly to
obtain more data by probing more cells, but comparatively cheap to run longer experiments
on the cells we already have, taking multiple datapoints from those longer tracks. That
leaves us with the problem of dealing with the implicit correlations in our dataset.
Another facet of this problem is that we want embeddings that integrate data from the
full past trajectory of the cell, instead of relying on the immediate past. To compromise
between all the above constraints, the datapoints on which the model trains are picked by
always using the entire available history (all past data from this track) in the encoder,
but never overlapping the unrolling decoder. By this solution, the model will sometimes
learn multiple things about the same cells, but at different points in time, while never
being asked to predict the same parts multiple times.

// Outline, not yet written: sliding-window approach; teacher forcing / scheduled
// sampling; regularisation (dropout, weight decay); FiLM conditioning across condition
// variables; composing batches / unbalanced data; block bootstrapping.

== Control scheme for model-predictive control

The model predictive control approach was based on the extending-horizon model. Before
the start of the experiment, a policy containing an objective function was loaded.

#figure-placeholder(
  [The inference path: the encoder state persisted per cell across frames, cross-entropy
   sampling over the discrete dose ladder, and the parallel decoder passes that score
   candidate plans.],
  "fig-mpc-schematic",
  height: 5cm,
)

The stimulation that the model is searching for is technically a continuous variable.
However, due to the constraints of the hardware (both for stimulation and for search over
possible values), the stimulation is collapsed into five discrete levels and treated as a
categorical variable. For the microscope, this simplifies processing the stimulation
masks, as collapsing to discrete levels means simply switching between five masks. For
the search process it helps by narrowing the theoretical search space to $5^L$ cases.

The inference process starts with per-cell encoder embeddings that are persisted across
time. At the beginning of the run they are zero-filled. At each new frame, the newly
received features are passed through the encoder and saved, then fed to the rest of the
model.

Then we start with finding stimulation candidates. The possible stimulation values start
off with a uniform distribution. We have `n_iter` iterations, at each of which we sample
`n_samples` sequences per cell, and then score them based on the objective function and
scoring function of the controller. The candidate stimulations are then ordered by score.
We then use the cross-entropy method to iteratively generate better candidate solution
distributions across `n_iter` by minimising the KL divergence between our sampler and the
empirical distribution of the best-scoring samples.

#todo[describe the single forward pass of inference: parametrising objectives, and
scoring. Decide whether these are implementation details or should be formalised here]

=== Unscored windows as room for an individual strategy

Since the model during its run learns some sort of embedding of the cell's response
phenotype, we were curious whether it exhibits diversity in the stimulation strategies it
promotes.

The basic check is looking at stimulation patterns in real experiments, aligning them to
each other by objective, and comparing them across different cells. The basic control
strategy requires us to specify a continuous objective function that is used to score
candidate solutions that the model comes up with. The effect is, however, that the model
is always evaluated on the entire path to the desired state. The creativity of the
solution is therefore constrained by the question we are asking the model — from state
$s_t$, what sort of stimulation $u$ do I need to find myself in state $S$ at time $t+1$?

An alternative test for whether the model can come up with diverse strategies for
different cells is having a window of unscored frames before a set objective. This way
the model is not evaluated on how it behaves prior to the frames we care about, and this
enables it to explore a diverse set of strategies for preparing the system for a later
trajectory. Three different ranges of unscored frame lengths are tested across multiple
objective patterns.

#thesisfig(
  "proposed-demands",
  [The free-window design: how much unscored time the controller is given before each
   demand opens.],
  "free window design",
)

// ═════════════════════════════════════════════════════════════════════════════
//  Experiments
// ═════════════════════════════════════════════════════════════════════════════

== Live experiments 

Experiments on the live cells were designed to probe the biological system as well as test the MPC pipeline. 
Standard length of the experiment lasted 12 hours,
and consisted of 8-12 fields of view on the microscope. 

=== Search for useful controller mechanisms

Predictive model can be evaluated offline, by observing its errors on already existing data. Due to its counterfactual nature,
controller is impossible to evaluate offline. Because of this, the first experiment (v10) aimed to evaluate usefulness of the base MPC controller and two
variations on its theme. I designed 4 experimental arms :
Arm 1 was a control, not utilising full MPC: it evaluated 5 possible futures at 30 frames - one for each choice of 
next stimulation, pretending that this exact stimulation will be applied on the next 30 frames. 
Arm 2 was a base MPC configuration - using L2 norm to score predictions and starting the sampling from a 
uniform distribution across choices of stimulation.
Arm 3 was a base MPC configuration with additional penalty for big 'jumps' between stimulation choices at 
subsequent frames - effectively promoting a smooth stimulation patterns over time.
Arm 4 had the same configuration as Arm 3, except the scoring method was changed into a band kernel.
We considered band kernel as a mechanism that could help specifically with asymetricity of control of our system - since 
we can only control to activate the optogenetic receptor but can only decrease CNR by waiting, overshooting the target has 
more serious implications than undershooting (since we can re-stimulate to correct for the undershoot in the next frame). 
L2 method is symmetric, scoring overshot and undershot predictions the same way. 
Band kernel could be designed to be more lenient on undershooting cells. 

=== Diversity of stimulation types found by the model

Early experiments (@fig-encoder) showed that model can encode single cell history into an embedding that is useful for predictions of its future state. #todo[weird sentence. feels discussion'y]

We wanted to investigate if this encoding is ever used as a discriminating factor not just for the quantity of the response given, but also its shape.
Orignal implementation for a controller receives an objective function, and scores the model's proposed solutions at every step, 
even before any interesting parts objective patterns start (for example, pre-experiment resting state). 
This limits the diversity of stimulation patterns, as it only asks the question of 'how to best follow the objective curve at every step'. 
The alternative approach would be to consider an approach where the solution's default frame is not coutered towards the score,
except when the frame is inside the objective interval.

This way, as the frame approaches the 'scored' section, the controller can use its information about driven cell and pick a 
pre-stimulation strategy that it considers best for crossing the 'scored' section.

Practical strategy for exploring this idea was to introduce a 'free window' of varying size before an objective pattern,
and observing stimulation patterns within this window.   

#figure(
  text(size: 8.5pt)[
    #table(
      columns: (1.15fr, 2.5fr, 1.25fr),
      align: (left, left, left),
      inset: 5.5pt,
      table.header([*Design*], [*What it asks of the loop*], [*Live runs*]),

      table.cell(colspan: 3, fill: luma(94%))[
        *Objectives* --- what the cells were asked to do],

      [Constant hold],
      [Bring every cell to one fixed CNR and keep it there. The simplest demand,
       and the only one whose reference never moves, so tracking error and
       measurement noise are not separable by shape.],
      [v12, v13, v14--v16],

      [Periodic waveform],
      [Drive CNR up and down on a repeating step train. Cells are split into
       phase groups, so one field carries the same waveform started at several
       different times.],
      [v10, v11, v12, \ v13, v17, v19],

      [Frequency staircase],
      [Blocks of shortening period with shrinking amplitude inside a single
       field, to find the period at which the response stops following.],
      [v12, v14--v16],

      [Arbitrary schedule],
      [Follow a demand curve defined by breakpoints rather than by a period, so
       that following cannot be achieved by locking to a rhythm.],
      [v14--v16, v21],

      [Segmented run-up],
      [Re-establish an anchor before each demand block, so that every block is
       entered from a known level rather than from wherever the previous block
       left the cells.],
      [v22, v23, v24],

      table.cell(colspan: 3, fill: luma(94%))[
        *Controllers and scoring* --- how the light was chosen],

      [Per-cell MPC],
      [The default: each cell's dose chosen from its own predicted trajectory
       over the planning horizon.],
      [every closed-loop \ run from v10 on],

      [Cost-function variants],
      [Whether the shape of the penalty matters: squared error against a
       dead-band that ignores small deviations, and a penalty on changing the
       dose between frames.],
      [v10, v11],

      [Open loop],
      [A fixed light sequence with no feedback. Serves both as the control arm
       and, when the sequence steps through the ladder, as a characterisation of
       the actuator itself.],
      [v13, v14--v16, v24],

      [Unscored window],
      [Leave a window before each scored span out of the cost, so the controller
       may pre-position the cell instead of being scored at every frame.],
      [v21, v22, v23],

      [Population MPC],
      [One dose shared across a group of cells rather than chosen per cell ---
       the comparison that isolates what per-cell control buys.],
      [v24],
    )
  ],
  caption: [#todo[caption. The experimental designs and the live runs that used
   them. Objectives and controllers are crossed rather than nested: one run
   carries several objectives across its fields, and one objective appears under
   several controllers. Runs before v10 are omitted, having been recorded
   without a controller policy. Per-run scalars and admissibility are in
   @fig-ledger.]],
) <tab-designs>

== Comparing single-cell control, population-level control and open loop stimulation 




#todo[After E2 is done, fill this one in] 
// ═════════════════════════════════════════════════════════════════════════════
//  RESULTS
// ═════════════════════════════════════════════════════════════════════════════
// Questions to answer:
// How does the offline model behave? Accuracy, uncertainty calibration, which features are important (encoder)
// Live experiment:
// - explanation of experiments, cadence slips, and moving resting CNR
// - comparison of prediction accuracy to the offline model. 
// - quantifying control drift (repeats, )

= Results

#todo[if E2 runs, open Results with the closed-loop against open-loop spread
contrast on one plate. The open-loop baseline is @fig-heterogeneity: within-field
p90#sym.minus#h(0.1em)p10 #sym.eq 1.73.]

== Model evaluation, or is a cell's response predictable from its own past?

// Outline brief: accuracy on held-out data; available history vs accuracy;
// horizon vs error; feature relevance over time (7c).

We evaluated the trained model on held-out trajectories from all the open-loop experiments dataset. 
Model achieved #todo[RMSE, R^2 stats for the whole held-out dataset]. on a 8-min horizon. 
This horizon is also the control horizon for the live experiment runs. 
Beyond this horizon, we observe the flattening of model's error. 

#thesisfig(
  "model-accuracy",
  [Forecasts from real context on held-out cells; error against lead time,
   compared with persistence (strategy where the forecast for future is that it will be the same as present ); predicted against observed at the control
   horizon.],
  "fig-model-performance",
)

#thesisfig(
  "forecast-examples",
  [A typical cell and a troublesome one, forecast from evenly spaced and from the
   cell's own highest-error starting points respectively.],
  "fig-forecast-examples",
)

=== Dependence on temporal context

Model showed heavy reliance on the encoded history of the cell. Using full history shows over 2-fold 
improvemnt in MAE error over a fresh context. 
Input features showed a large dependance on the encoded context.
OptoRTK expression rank in particular is very helpful early in the run (contributing 0.24 R^2), superseding even fluence. 
Spatial features (local crowding, field density) are not used by the model. 

#thesisfig(
  "encoder-needs",
  [Forecast error falls #todo[2.2]#sym.times as the encoder is given more of the
   cell's own past, saturating around 20 minutes.],
  "fig-encoder",
)

#thesisfig(
  "history-swap",
  [The past the encoder uses is cell-specific: replacing a cell's history with a
   level-matched donor's roughly doubles forecast error, and the penalty does not
   decay across the run.],
  "fig-history-swap",
)

#todo[gap: "feature relevance over time" (7c in the outline). No such figure
exists yet.]

// ------------------------------------------------------------------------------------
=== Calibrating uncertainty 
// ------------------------------------------------------------------------------------

// Outline brief: reliability — for every prediction and its confidence
// interval, count the observations that fall inside it. Does the 90% interval
// contain 90%? Does the 50%? Compare the full mixture against a collapsed
// single-Gaussian head. Is there symmetry in the tails — does the model tend to
// under- or over-estimate? Does coverage change with forecast lead time?

The gaussian mixture model employed outputs a 3-compoent gaussian mixture for each prediction step it makes. 
Since every prediction carries its own distribution (different mixture weights, means and variances), we used
Probability Integral Transform as a way to evaluate if the calibration is correct. 
We computed CDF of our mixture model and compared it to the empirical observations. 
The tests were conducted on n=7237 forecast starting points, computing 30 predictions steps for each. 
As a comparison, a gaussian z-score of the mixture was also used, to see if a single gaussian with full mixture's 
total spread would be sufficient to replicate its coverage. 
We observe that exact mixture is well calibrated by its coverage. Comparatively, the z-scored gaussian is underconfident, signifying that 
the choice of using a mixture was beneficial.

Former analysis is symmetric in nature, as coverage scans central intervals. Evaluating raw PIT densites, we notice that the model is generous in the lower tail of the distribution.
It anticipates more downward movement than there are in reality, especially so in the early steps of a forecast - at the fist prediction step lower
tail is -4.3 percentage points, and it reduces to 0 at step 30.

A separate issue from accuracy itself, is whether the mixture's standard deviation is tracking factual error.
To do that, RMSE over predictions is plotted against prediction's standard deviation binned into deciles,
falling on an identity line - the predicted standard deviation matches the realised RMSE in magnitude and not only in rank.
Across deciles the predicted standard deviation spans roughly a tenfold range and the realised error tracks it, so σ discriminates between easy and hard predictions rather than reporting one width everywhere.

#thesisfig(
  "uncertainty-calibration",
  [The stated intervals cover at their nominal rate; both tails are close to
   right; calibration holds across the whole planning horizon; and #sym.sigma
   tracks where the errors actually are.],
  "fig-calibration",
)

#todo[gap: the comparison of the mixture-density head against ensembles and MC
dropout is in the outline and has not been run. 
it is an offline analysis.]

// ------------------------------------------------------------------------------------
== Controling live experiments
// ------------------------------------------------------------------------------------
19 experiments were conducted in total. 
experiments (v10, v11, v16, v19, 21, v23, v24) were both technically sound and contained useful information.

Data from live experiments has a caveat in that all single-cell tracks being evaluated must 
necessarily have been tracked for at least 9 out of 12 hours of the experiment. 
This leaves a possibility of a survivorship bias. 
#todo[Figure idea from lab notebook - plot all of the events of a cell that stops being tracked at a given point in time. In order to see if there is some pattern and how the cells drop out of the experiment.]

// Outline brief: is the calibration of real vs. trained good — does the model
// have the same predictive characteristics on the rig? Overall accuracy
// (residuals); comparison of closed loop to open loop; distribution of
// residuals across stimulation-pattern types.

// The ledger comes first: it is the run-selection argument. Everything the rest
// of this section quotes is drawn from the four runs that clear both gates, and
// this is where that choice is stated rather than assumed.

Experiment v10 and v11 (@exp_v10_arm2) evaluated based MPC controller against two addioinal mechanisms: 
move penalty, and band kernel scoring across two different frequency
settings. 
This establishes a working experimental run and an early sign of success. Notably, the runs 
did encounter some problems. Cadence slip is a failure mode where as the microscope 
moves between fields of view, the full rotation through all fields takes longer than planned minute.
Early experiments encountered such problems early, due to a misconfiguration of the opearting system.
Experiments v10 and v11 worked at a cadence of 85s and 69s respectively instead of standard 60s. 
However, as this circumstance affects all conditions equally, the conclusions from this experiment
are not invalidated.

#thesisfig(
  "exp_v10_arm2_plot",
  [Experiment v10 demonstrates the model working in a real experiment, even despite cadence slips.],
  "exp_v10_arm2"
)

This experiment shows model's capability to push a population of cells into a desirable behavior. 
The controller manages to keep desired frequency of the oscillations throught the experiment.
However, across cycles controller's ability to push cells into full amplitude diminishes.
During the first cycle median CNR reached desired level, while in the last cycle median CNR reached only halfway the objective. 
This prompted the design of more experiments that could help with quantifying the effect of diminishing responsivity to control. 
The best controller arm is the base MPC one, scoring lowest RMSE values in both experiments (0.1916 in v10, 0.1554 in v11), and
thus was used in all later experiments.
#todo[Mann-Whitney U test w/ Holm corr for multiple testing to have p-value of the above claim?]

Having obtained a stable controller configuration, we moved into a more diverse set of objectives for experiments.
Generalization of accuracy and uncertainty from the offline evaluation into live experiment regime was done.
Both transfer well on at the forecast horizon, but uncertainty shows degraded performance in mid-horizon range. 
#thesisfig(
  "rig-calibration",
  [Accuracy and uncertainty comparison of the data from 4 live experiments running under the same controller to the results of offline evaluation.],
  "rig-calibration",
)
The uncertainty quantification is the worst at the 8-15 timestep horizon, (68% nominal outcomes are matched with 61% 
in live experiment at horizon of 15 frames, while offline evaluation returned 70),
but returns to trustworthy levels (95% nominal coverage is matched by 93% in live model). 
Interestingly, the direction of error of live deployment is the opposite of the offline one - while offline uncertainty was 
slightly too 'wide' (model assigned the border values such that when it thought 68% of samples will land there ,70% actually did)
, the live model assigned them narrowly (68% was supposed to land, but 61% did). 

A visible effect in a lot of experiments with repeating block objective was the gradual 
flattening of the response to stimulation (@sensitivity-decline).


#thesisfig(
  "sensitivity-decline",
  [Experiments with a repeating objective componenet, mean CNR, goal (dotted), and 95% confidence interval for the first repeat and the last repeat of the pattern in the experiment. In all experiments except 70m arm v19, last cycle is consistently lower, while taking up more light. ],
  "sensitivity-decline",
)

// This figure opens the section on purpose. It states what the instrument can
// and cannot do BEFORE any tracking number is quoted, so the limits read as a
// measured property of the preparation rather than as an excuse offered after
// the fact. It is also the only place the between-run drift appears.
#thesisfig(
  "reachability",
  [#todo[caption. (a) resting CNR in v21/v22/v23 with each run's demand marked —
   the population moved 0.67 → 0.92 → 1.10 between runs. (b) every cell's
   reachable range against the demand band: 35% start above the anchor, 8%
   cannot reach it. (c) distance from the demand predicts tracking error
   (#sym.rho #sym.eq +0.15) where the resting level alone does not
   (#sym.rho #sym.eq −0.01). (d) one-step forecast error does not predict
   tracking error, and is five times smaller — the model is not the bottleneck.]],
  "fig-reachability",
)

//#thesisfig(
//  "rig-calibration",
//  [*Whether a model trained under whole-field illumination is calibrated for a
//   rig that lights one nucleus at a time.* (a) The model is worse than assuming
//   the cell stays put until it has seen a few frames of that cell, and better
//   after; counts per bin are printed at the foot of each column. (b) The signed
//   error, so its direction is visible: the model expects more than it gets, and
//   most of that offset clears once the encoder has context. (c) What survives
//   when a cold start cannot be the explanation, using only cells with 150 or
//   more frames of history. In the dark the forecast is unbiased; the offset
//   grows with the commanded dose, which is what a model trained under
//   whole-field light and then served one nucleus at a time would do.
//   (d) The control: the gap between the two history bands is open at every hour
//   of the run, so the cold start is not the cells drifting over the session --
//   fresh cells are picked up throughout and each pays the same entry cost.],
//  "fig-rig-calibration",
//)

//#thesisfig(
//  "tracking",
//  [How well the demand was met, and by how much of the population — each cell
//   judged against its own drift.],
//  "fig-tracking",
//)

todo[describe the runs?]

#thesisfig(
  "bandwidth",
  [*How much of the demand arrives at each timescale.* (a) One cycle of each
   arm, folded on phase, against that arm's own demand. (b) Two ways of scoring
   the same thing: how much the population is modulated at all, and how much of
   what the objective actually asks -- hold to hold -- arrives. The gap between
   them is the cost of a hold too short for a cell with a roughly three-minute
   time constant to settle into: the 20 min arm holds high for 4 min, the 70 min
   arm for 29. (c) Phase of the response in degrees of the cycle rather than in
   minutes, so the four arms are comparable. The medians sit within a few
   degrees of locked at every period despite three to five minutes of dead time
   in the loop -- the controller's 30-frame lookahead pays that back. The bars
   are the interquartile range across cells. (d) The price of
   tracking, early against late in each run: the error barely moves while the
   dose required to hold it climbs.],
  "fig-bandwidth",
)

#todo[gap: closed loop against open loop. This is the section's central claim
and no valid experiment exists for it yet. The feedback-ladder policy
(policies/policy_8fov_feedback_ladder.toml) is written and unrun.]

// Outline brief: experiment with unset parts of the objective — breakdown of
// stimulation-pattern types under the SAME objective. Does the model find
// separate stimulation strategies for one goal? How would we show it —
// dimensionality reduction then clustering, ordered by L2 so the result is
// steerable cells and not dead ones?

#todo[Dose diversity analysis - How?]

#thesisfig(
  "response-modality",
  [Whether there are two kinds of cell or one very broad kind.],
  "fig-modality",
)

== Control and sensitivity drift

// Outline brief: average L2 / residuals over the course of an experiment,
// against elapsed time, against the integral of fluence, and as the increase in
// fluence needed per cycle over 12 h.

#todo[decline in sensitivity fig]


#todo[fig-decline puts the clock and cumulative light near-tied
(#todo[4.2] vs #todo[4.3]), not clock-dominant. The text must say that, and
separating them would need a duty-cycle arm — same total light, different
elapsed time.]

// ═════════════════════════════════════════════════════════════════════════════
//  DISCUSSION
// ═════════════════════════════════════════════════════════════════════════════

= Discussion

// Outline brief: the sensitivity-drift issue — possible interpretations.

#todo[write]

== Interpreting the sensitivity drift

#todo[the candidate readings: receptor internalisation and downregulation;
photobleaching or phototoxicity of the construct; medium exhaustion over a
12 h starved run; adaptation in the pathway itself. fig-decline bears on
which of these can be told apart with the data in hand, and which cannot.]

== Impact of resting CNR states on ridgid objectives

Objective during the current experiments was pre-set. During some experiments,
the baseline starting CNR level varies to the point, where #todo[Difficulty with designing and conducting good experiments because of a varying pre-stimulation baseline CNR.]

// ═════════════════════════════════════════════════════════════════════════════
//  FUTURE WORK
// ═════════════════════════════════════════════════════════════════════════════

= Future work

#todo[write]

- Make time an explicit axis in the model, either in the conditioning or as a
  feature (`time_since_last_measurement`). This makes it possible to learn
  multiple timescales, and may do better on long-range effects such as receptor
  internalisation or transcriptional feedback.
- Extend into the spatial dimension, possibly with a hierarchical model.
- Use the model to help fit a hybrid approach — a neural ODE, a
  physics-informed network, or similar.
- Use it to quantify the controllability of systems of this kind, and to ask
  whether the control drift itself can be influenced.

// ═════════════════════════════════════════════════════════════════════════════
//  APPENDIX
// ═════════════════════════════════════════════════════════════════════════════

#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1")

= Appendix

== Training loss curves

#figure-placeholder(
  [Training and validation loss for the reported model.],
  "fig-loss-curves",
)

== Light dose calculation

#todo[the energy emitted by a light pulse — the outline flags this as possibly
belonging in the appendix rather than in the methods]

== Additional figures


#thesisfig(
  "run-ledger",
  [#todo[caption. Every live run scored on the same two gates. (a) achieved
   cadence, median to p90, against the 1 min interval the model was trained on;
   seven runs slipped, from 1.16 to 5.72 min per frame. (b) share of closed-loop
   cell-frames sitting on the top rung of their own field's ladder, annotated
   with the ladders issued. Saturation must be measured per field: v14--v16 gave
   their closed-loop fields a 150 ms ladder while driving their open-loop fields
   to 600 ms, so a run-wide figure understates v16 by more than a factor of ten
   (5% against 73%). Three runs of nineteen clear both gates (v21, v23, v24);
   v16 and v19 hold cadence but saturate 73% and 32% of the time; v22 is
   excluded for a mis-set objective.]],
  "fig-ledger",
)

#thesisfig(
  "arm-tracks",
  [#todo[Describe the tracks] ],
  "experiment tracks",
)

#todo[park supplementary panels here as they are cut from the main text]

// ═════════════════════════════════════════════════════════════════════════════
//  BIBLIOGRAPHY
// ═════════════════════════════════════════════════════════════════════════════

#set heading(numbering: none)
#bibliography("refs.bib", style: "nature", title: "References")
