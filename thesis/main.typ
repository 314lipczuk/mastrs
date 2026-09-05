
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
utilising Mixture Density Network (MDN) head for communicating uncertainty. Predicive model is then integrated into a 
Model Predictive Control (MPC) controller, and used to steer experiments in real time. 

The trained model is accurate and calibrated enough to drive closed-loop control experiments. 
However, the sensitivity drift, objective geometry and the initial conditions of the controlled population 
can greatly influence the overall success. 



#todo[abstract last]

// ═════════════════════════════════════════════════════════════════════════════
//  INTRODUCTION
// ═════════════════════════════════════════════════════════════════════════════

= Introduction

This thesis closes a control loop around individual cells. The loop has three parts: a
learned model that predicts how a single cell's ERK activity will respond to light, a
controller that plans stimulation against those predictions, and an objective defined for
each cell rather than the population. A loop of this kind, with a learned predictor in
it, has been demonstrated for gene expression @Lugagne2024, which unfolds
over hours. A signalling cascade moves in minutes, and this fact changes what the loop must
do. It has to predict in real time, carry uncertainty it can plan against, and steer
through an actuator that only pushes activity up.

== ERK dynamics and cell fate

The MAPK/ERK pathway plays a key role in cellular proliferation signalling, utilising dynamics rather 
than stable states to control cell fate decisions @Ryu2015.
In PC12 cells, a sustained pulse of pathway activation drives differentiation while a
transient pulse drives proliferation @Marshall1995; modulating the frequency of
activation alone can rewire fate decisions @Ryu2015.

Cells partake in complex spatiotemporal population-level phenomena such as ERK-waves @Aoki2017, that affect processes such as wound healing, apoptosis resistance. 
Cancer cells have been shown to impact these high-level communications, interfering with the surrounding healthy tissue. 
#todo[citations: wound healing, cancer interference — ask MD which papers to name.]

//  Bridge note: talk about small population influencing dynamics of whole tissue — the bridge lives in the cancer's-book paragraph of the closed-loop section

== Why the population average is not enough

A standard problem in studying  biological systems is that a quantity easily modelled at
the level of a population statistic need not describe any individual in it. The
difficulty is not that the average is imprecise; it is that the events of interest
happen in single cells. A cell differentiates or it does not, commits to a cycle or
does not, dies or survives. An average over such events describes a state that no cell
need ever occupy.

For ERK this matters because the responses really are spread. Genetically identical
cells differ in protein abundance in ways that precede any stimulus @Spencer2009. That
shows up directly in this preparation: cells receiving an identical pulse train respond
over a wide and continuous range, with a substantial minority barely moving at all
(@fig-heterogeneity).

The consequence for control is direct. A loop that steers a population average can drive
that average onto a target while leaving the individual cells further from it than
before, and it has no way to tell the two outcomes apart. Deciding between them requires
an objective written for each cell, and light delivered to each cell separately.

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

== Mechanistic models and their limits

// (superseded stub, kept for reference: Using ODE or PDE models of a signalling
// cascade together with standard statistical machine-learning techniques, we
// can fit a model and use it to generate predictions about future states of a
// system.)

Predicting how a given stimulation of the pathway, be it via a growth factor 
or an optogenetic stimulation of an upstream receptor affects the downstream ERK dynamics
has been dominated by mechanistic approaches that encode a model of the biochemical
cascade with each participant quantified, and utilise ODE fitting approaches to adjust
model parameters to real measurements. 
This approach, while principled, can suffer from a number of issues. Data for the 
fitting of such model often only consists of an insufficient number of state variables of the system in question.

The model itself often suffers from parameter nonidentifiability @Gutenkunst2007: 
many parameter sets fit the data well, so the fitted values carry little meaning. 
Its construction requires expert knowledge, and it demands that
the system be encapsulable within the chosen level of abstraction: morphology or
mechanical stimulation, for example, have no natural place in a biochemical ODE model
of a pathway.

Crucially, the fitted mechanistic model is a static description: it cannot natively represent a change
in how a cell responds over time without being refit. Responses to change, like the sensitivity drift measured
in this work, are such a case. 

What this approach delivers is a mechanistically interpretable model: one that encodes
what is believed about the biology and can be interrogated as well as used to predict.
That advantage is bought at the cost of the limitations above, and of a reach that does
not extend past the conditions the model was fitted under.

== Learning to predict from data

// (superseded stub, kept for reference: While the mechanistic approach can test
// our understanding of the theory behind a biological system, another approach
// is to use data-driven techniques to predict the system's behaviour, and to
// learn to control it.)

If the goal is not understanding but prediction in itself, we can turn to models learned
directly from data. Neural networks are universal approximators @Hornik1989, but the
useful property here is a weaker and more practical one: given enough examples, a
sequence model can learn the input--output behaviour of a system without being told its
mechanism. Recurrent architectures such as the LSTM @lstm are built for exactly this,
carrying state forward so that a prediction can depend on a cell's whole measured
history rather than its present value alone.

The readout used throughout this work is such a sequence: the cytoplasm-to-nucleus ratio
of a translocation reporter, which tracks ERK activity rather than the abundance of any
kinase.

A work by @Klumpe2023 showed that deep neural networks were able to infer the underlying dynamics of a cell response
even in the presence of measurement noise and stochasticity in the biochemical reactions.

// brief: not explaining mechanisms, but uncovering more complex behaviour and
// learning through interactions.

== Closed-loop control as an instrument

#todo[closed-loop control literature — Khammash, FARO, the preceding grant]

Approaches such as optogenetics can be used to control cellular processes. 
Optogenetics is well suited to fine-grained control over dynamics. 
Light can be delivered with millisecond precision, targeted at subcellular resolution, and acts reversibly.

One property of this actuator shapes everything that follows: light drives ERK activity
up, and nothing drives it down. A cell descends only by its own decay, so a demand below
where a cell already rests is not hard to reach but unreachable, and what each cell can
be asked to do is fixed by where it happens to sit when the run begins.

Learning how cells process these signals internally can aid us in building better models of the tissues and their signalling. 
An interesting avenue for research is taking a note from the cancer's book and asking: 'Can *we* design a small, controllable populations that will allow us to 
influence the behavior of the big surrounding tissue?'
This thesis takes a step in the direction of this ambition, by reliably controlling individual cells in real time.

Control that plans against a predictive model of the system, rather than reacting to
error through a fixed calibrated response (PID), is called Model Predictive Control
(MPC) @DDSE. At each step the controller searches over candidate input sequences,
forecasts the system's response to each, and scores it against a cost that combines
distance from the desired trajectory with the price of the input itself. The
best-scoring sequence is found, but only its first step is applied: the horizon then
slides forward and the plan is recomputed from the new measurement. Replanning at every
step is what lets the loop absorb both a model that is imperfect and a system that
changes underneath it.

Any MPC loop needs a model that can forecast the system under a proposed input, and a
mechanistic ODE can serve that role. Control asks for prediction rather than
identifiability, so parameters that are not uniquely determined are not by themselves
disqualifying. What does disqualify is the last objection above: a fitted ODE cannot
follow a cell whose response changes over the run without being refit. A model learned
from observational data carries no such commitment, and that is the class used here.

Deep MPC has been demonstrated for gene expression: @Lugagne2024 steered expression levels
in thousands of single cells under blue light, planning with a neural predictor.
Gene expression, however, is a slow readout: it unfolds over hours on a transcriptional
timescale.

== Contributions:
#todo[write those after final analysis]

= Materials and methods

== OptoEGFR cell line and culture

A previously established NIH3T3 mouse fibroblasts cell line (ATCC CRL-1658) stably
expressing optoEGFR-mCitrine together with ERK-KTR-mScarlet3 and H2B-miRFP670nano3 was
used for all experiments. OptoEGFR and the downstream biosensors were expressed under
CAG promoters. Cells were grown and maintained in Dulbecco's Modified Eagle's Medium —
high glucose (Sigma-Aldrich \#D5671), supplemented with 10% (v/v) fetal bovine serum,
2% L-Glutamine (stable, 200 mM) and 1% penicillin/streptomycin at 37 °C and 5% CO#sub[2].
Mycoplasma contamination was routinely assessed by PCR.


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
possible values), the stimulation is collapsed into six discrete levels and treated as a
categorical variable. For the microscope, this simplifies processing the stimulation
masks, as collapsing to discrete levels means simply switching between six masks. For
the search process it helps by narrowing the theoretical search space to $6^L$ cases.

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

Three levels of structure recur throughout, and the distinction between them matters for
how the experiments are analysed. A *field of view* is one imaging position on the plate;
a run carries eight to twelve of them, and each field keeps the same controller
configuration for the whole experiment. A *block* is one repeat of the objective in time
typically a run-up followed by a demand, and a run carries ten to twelve of them
in sequence. Every field is imaged in every block, so fields and blocks are crossed
rather than nested: the objective at a given minute is the same in all fields, and the
controller configuration is the same in all blocks.

This has a direct consequence for what counts as a replicate. A treatment applied to
fields, such as the length of the unscored window, is replicated by fields, typically
two of them. A treatment applied to blocks, such as which demand pattern is being asked
for, is replicated by blocks, three of them per pattern. The individual cell is a
replicate of neither: a cell lives in one field and survives many blocks, so its
measurements are repeated observations rather than independent ones.

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
Original implementation for a controller receives an objective function, and scores the model's proposed solutions at every step, 
even before any interesting parts objective patterns start (for example, pre-experiment resting state). 
This limits the diversity of stimulation patterns, as it only asks the question of 'how to best follow the objective curve at every step'. 
The alternative approach would be to consider an approach where the solution's default frame is not counted towards the score,
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
        *Objectives*: what the cells were asked to do],

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
        *Controllers and scoring*: how the light was chosen],

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
      [One dose shared across a group of cells rather than chosen per cell:
       the comparison that isolates what per-cell control buys.],
      [v24],
    )
  ],
  caption: [The experimental designs and the live runs that used
   them. Objectives and controllers are crossed rather than nested: one run
   carries several objectives across its fields, and one objective appears under
   several controllers. Runs before v10 are omitted, having been recorded
   without a controller policy. Per-run scalars and admissibility are in
   @fig-ledger.],
) <tab-designs>

== Comparing single-cell control, population-level control and open loop stimulation 

Experiment 24 was designed to compare single cell level MPC with population
level MPC against an open loop stimulation matched to experiment beforehand.

#thesisfig(
  "e2-design",
  [The v24 design. Arm 1's four fields are split inside the dish: (particle / 4) % 2 decides which cells share one broadcast dose, so 1a against 1b is paired within a field rather than compared across dishes. Arm 2 delivers a flat 60 ms, set from v23's closed-loop arm means; arm 3 receives no stimulation light at all. The nine blocks are a complete 3 × 3 of levels against rates, each combination once, in a counterbalanced order.  ],
  "e2-design",
)

A split into blocks instead of splitting by field of view was used. Within FOVs 0,3,4,7, half of the cells were stimulated with the standard single-cell pipeline, 
and other half were pooled into a shared population, their features averaged and their control signal computed from the population average. By sharing the FOVs across
blocks, we avoid per-FOV effects confounding our result. 

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

We investigated how much does the identity of encoded history matter to a cell by matching 
two cells from the same experiment by their CNR level, and then using encoder state of the first to 
predict future of second. The results show that for 89% of cells the results of such swap 
yield prediction errors larger than cell's own.   

#thesisfig(
  "history-swap",
  [The past the encoder uses is cell-specific: replacing a cell's history with a
   level-matched donor's roughly doubles forecast error, and the penalty does not
   decay across the run.],
  "fig-history-swap",
)

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


// ------------------------------------------------------------------------------------
== Controlling live experiments
// ------------------------------------------------------------------------------------
19 experiments were conducted in total. 
After initial prototyping and troubleshooting the technical issues, 
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

Experiment v10 and v11 (@exp_v10_arm2) evaluated the base MPC controller against two additional mechanisms: 
move penalty, and band kernel scoring across two different frequency
settings. 
This establishes a working experimental run and an early sign of success. Notably, the runs 
did encounter some problems. Cadence slip is a failure mode where as the microscope 
moves between fields of view, the full rotation through all fields takes longer than planned minute.
Early experiments encountered such problems early, due to a misconfiguration of the operating system.
Experiments v10 and v11 worked at a cadence of 85s and 69s respectively instead of standard 60s. 
However, as this circumstance affects all conditions equally, the conclusions from this experiment
are not invalidated.

#thesisfig(
  "exp_v10_arm2_plot",
  [Experiment v10 demonstrates the model working in a real experiment, even despite cadence slips.],
  "exp_v10_arm2"
)

This experiment shows model's capability to push a population of cells into a desirable behavior. 
The controller manages to keep desired frequency of the oscillations throughout the experiment.
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
  [Experiments with a repeating objective component: mean CNR (solid), goal (dotted), and 95% confidence interval for the first repeat and the last repeat of the pattern in the experiment. In all experiments except 70m arm v19, last cycle is consistently lower, while taking up more light. ],
  "sensitivity-decline",
)

The experiments were designed in a way to maintain the same goals regardless of cells' starting position, resting state, and other characteristics.
This was done to see if a heterogeneous population can be controlled in a manner that makes it behave homogeneously, if given personalised stimulation.
However, it also introduces problems with reachability of the goal state by populations, especially as the distribution of
resting states across different experiments varies wildly, and are not known before starting the experiment.

#thesisfig(
  "reach-and-tracking",
  [Resting states of experiments compared to the tracking error within them. Experiments ordered by median tracking error. ],
  "reach-and-tracking",
)

Experimental outcome was better in experiments that had the objective aligned to the cells' resting CNR.  
The population's resting spread is wider than the light can move any one cell. 

#thesisfig(
  "how-cells-move",
  [ Influence of light on CNR. Rows are cells pooled across all seven runs and binned by the mean light that cell itself received. The null is v24's dark arm — 229 cells held with no stimulation light for twelve hours, median apparent reach 0.14 CNR and p90 0.30. Bars: p10–p90 faint, p25–p75 solid, tick at the median.],
  "how-cells-move",
)

== Comparing levels of control
// ------------------------------------


// v24 experiment
Experiment v24 aimed to compare population-level feedback control against a single-cell one and an open-loop control.
The pattern which the cells were supposed to reach has been planned based on the resting state and reach
of the previous experiment (v23), which had these features spread unusually high. Because of this demand,
experiment 24's goal was set above what the cells were actually able to achieve in 8 out of 9 blocks (@e2-arms).
Mismatch of baseline in the design meant that the open-loop constant arm that was meant to deliver average pulse light from one of the previous experiment in order
to reach the same level, failed to do so. It ends up delivering 60ms constant stimulation at every frame, while closed loop controllers deliver on average 112.5 ms / frame.  

#thesisfig(
  "e2-arms",
  [ What the three lit groups achieved. A cell's ceiling in panel (b) is the 95th percentile of its own CNR over the run — what it actually reached, under whatever light it was given. The resting lines are the dark arm, which never received any light. Panel (a) shows the median cell of each rung against the one reference all eight fields were given, with the interquartile band on the per-cell arm.],
  "e2-arms",
)

Despite all the problems, one block was reachable (initial hold pattern on demand level L). From it we can try to extract preliminary conclusions, which need further confirmation with a 
proper experiment. 

The clearest is block 01 itself. Over that hour the closed-loop arm held a per-cell
tracking error of 0.161 CNR, against 0.169 for constant illumination and 0.239 for
darkness, and it did so on less light than the constant arm spent. Neither of the
failures above applies here: the demand was achievable, and the loop chose to spend
under 60 ms per frame for most of the block. What the population median does over the
same hour is worth noting separately. The constant arm's median sits marginally closer
to the demand than the closed loop's, 0.024 below it against 0.034, while its individual
cells sit further away. The average is closer and the cells are worse, which is the
argument for per-cell objectives made in the introduction, appearing inside a single
block of a single run.


#thesisfig(
  "feedback-ladder-alt2",
  [],
  "feedback-ladder",
)

The second comparison is between the two halves of the closed-loop arm, and it is the
only place in this work where individuation is isolated. Cells sharing a field were split
by a fixed rule, half planned individually and half receiving one dose computed for the
field as a whole, so the comparison is paired inside the dish and every field-level
difference cancels within the pair. Per-cell dosing gave a median tracking error of 0.295
against 0.325 for the broadcast dose, on 113 against 123 ms of light per frame: closer,
and on less light. Three of the four fields favour per-cell dosing and one does not,
which at four fields is a direction rather than a result, with a sign test giving
p = 0.375.

The third comparison is the full ladder across all eight fields. It is the strongest
result statistically and the weakest in interpretation. Field median tracking error
orders monotonically with how much feedback the arm had: 0.284, 0.288, 0.309 and 0.349
for the closed-loop fields, 0.375 and 0.399 for constant illumination, 0.470 and 0.486
for darkness. Eight fields in groups of four, two and two admit 420 distinct
relabellings, and the observed ordering carries an exact permutation p of 0.0048, with
Spearman rho of +0.93. The caveat is the one already given: across the whole run the
closed-loop fields spent 116 to 124 ms per frame against the constant arm's 60. Part of
this ordering is the light rather than the feedback, and this run cannot separate them.

Taken together, v24 supports two claims. A closed loop holds a reachable level better
than constant illumination and better than nothing. Dosing cells individually is at least
no worse than dosing them together, while using less light to do it. It supports no claim
about following a waveform pattern, and none about performance at a matched dose.

== Are there multiple distinct strategies of stimulation that controller undertakes?

Visualising the activations in free windows across the different arms shows a continuum of strategies that the controller picks for single cells. 

#thesisfig(
  "freewindow-heatmap-by-demand",
  [Heatmap of activations in every pre-demand window, binned by the length of unscored time during which the controller could come up with pre-stimulations. Left sidebar marks the goal pattern that a given cell was tasked with reproducing. Visible clustering of those in one strategy could mean confounding of strategy with goal. ],
  "freewindow-heatmap-by-demand",
)

The ordering runs between two extremes rather than between two kinds. At one end there is
little if any stimulation until roughly seven minutes before the demand, followed by
strong stimulation; these rows sit at the top of each arm in
@freewindow-heatmap-by-demand. At the other there is medium stimulation up to that same
seven-minute mark and then nothing at all. Every intermediate between them is occupied,
and no edge separates one from the other.

#todo[mention free window size -> error drop, not significant ]

The extremes are also rare. Along this axis the density has a single peak near the middle
in every arm, with several times as many windows near the centre as in either tail. What
the unscored window changes is the width of that distribution rather than its shape: the
interquartile spread of the shape score doubles, from 0.079 in the arm scored throughout
to 0.159 in the arm given twenty free minutes, while remaining single-peaked in all four.
The two arms whose windows fall inside the loop's dead time, zero and four free minutes,
are almost indistinguishable at 0.079 and 0.085, and the widening begins only once that
dead time is cleared. More unscored time buys more variety, not more kinds.

The seven-minute mark is shared by both extremes rather than distinguishing them. It is
the lead time the actuator needs: driving flat out from the anchor, the cells take about
seven minutes to reach the demand, so light spent earlier than that has decayed before it
is scored and light spent later arrives too late.

The arms with a longer unscored windows exhibit more instances of using high light intensity, perhaps due to the fact that are not scored negatively
for an overshoot before demand, enabling strategy of 'dropping' into the desired state from a strong activation. 

An interesting result to tackle is that even in the control group (free window of size 0), there is diversity of stimulation types. This suggests that the model
will in fact trade off short term reward of following the current demand exactly for ability to meet later, harder demand. 
Nevertheless, with bigger unscored window comes higher diversity of strategies.

All of the demand patterns open above the cells' resting state, but they differ in what
they ask for afterwards. That leaves a possible confound: the stimulation the controller
chooses in the free window might depend on which demand is coming, so that what looks
like a spread of strategies is really a spread of objectives.

Testing this means comparing what the controller did before one demand against what it
did before another. Each window is summarised by its position on the first shape
component, in effect how early in the window the light was spent. Because the demand
pattern belongs to the block and not to the cell, the windows within a block are averaged
first, leaving one number per block and three numbers per demand pattern.

Two spreads can then be computed from those numbers, and the comparison between them is
the test. The first, *do the demands differ?*, measures how far apart the four pattern
averages are. The second, *do blocks of the same demand differ?*, measures how far
apart the three blocks of one pattern are from each other. If the first is no larger than the
second, then the differences between demands are within the range that repeats of a
single demand already produce, and nothing can be attributed to the objective.

That is what is found. In v23 the four pattern averages span 0.047 on a scale where a
single window's standard deviation is 0.077, while the three blocks of one pattern span
as much as 0.040 between them ($F(3,8) = 1.87$, $p = 0.21$); v21 agrees, with the spread
between patterns smaller than the spread within them. What the controller does in the
free window does not depend on the objective ahead.

#thesisfig(
  "freewindow-examples",
  [Two windows from opposite ends of the shape space. Chosen by a stated rule: the window nearest the 5th and nearest the 95th percentile of the first shape component, both at the median of the second. Dashed line: the demand. Bars: light commanded. Green: the unscored window. Both cells are shown over one whole block.],
  "freewindow-examples",
)


// ═════════════════════════════════════════════════════════════════════════════
//  DISCUSSION
// ═════════════════════════════════════════════════════════════════════════════

= Discussion

// Initial intro
Predicting cell responses is important for the understanding of population-level dynamics. 
This work shows a successful application of Deep Model Predictive Control for signaling dynamics 
of single cells in a closed loop experiment.
The technical pillars of conducting such experiments is a predictive model, controller that plans action, 
and an objective function.
Using data from previous experiments, we trained a predictive model of CNR and then used it
for closed-loop predictions, and steering individual cells with the help of a controller in real time.

// choice of an objective, pattern geometry
During our experiments, choice of an objective was an arbitrary, experiment-wide static decision. 
This helps with evaluation of multiple cells against a single target,
but is not the best possible fit to the problem
Within a population, there is a bigger dispersion of resting CNR values,
than an average cell is capable of moving (@reachability-runs). Because of this, a static, population-wide pattern will
always have a proportion of the population over the demanded state, and some that is incapable of reaching it.
Geometry of the objective matters as well, especially in a system that can only be perturbed one way - into
activation, and so the objective function should take into account realistic deactivation.

// future work - adaptive objective idea
This is something addressable in further research. 
One possible alternative would be to encode objective itself on a per-cell basis - for example based on the 
initial resting CNR state, position within a cluster or number of neighbours.
For probing population-level phenomena, this approach could be 
used to pre-select the cells that exhibit signs of 'good controllability', 
and then stimulate only those.


// Diversity of stim
//   - history-swap shows that personalised history is crucial for prediction accuracy 
//   - despite that, The stimulations used inside of
//     the free window experiment look as though they were taken from a
//   . single gradient across all windows, with only the difference being the steepness of the gradient.
//     A possible explanaiton for this is that the optimal stimulation differs in quantity rather than kind,
//     and the current stimulation ladder is too coarse to express it. 
// per cell control result
We found that a controller's choice for stimulation is sampled from a continuum of behaviors.
On one end, the strategy of stimulating only just before the demand is scored, and on the other end 
to stimulate early, then stop and let the CNR fall into the desired state. 

Increasing the duration of unscored window before the main objective changes the distribution of behaviors picked - 
widening the cases at the edges of the continuum (mentioned above) at the expense of the 
'constant stimulation' case that lives inbetween them. 
The constant stimulation strategy is more prevalent in
cases with no free window, perhaps because it does not allow the cells to fall into their own baseline, but 
instead commands them to hold an 'estimated baseline', leading some cells to need to be stimulated.
@history-swap shows how taking cells with matching CNR but swapping their histories has a severe negative influence
on their predictions, highlighting the importance of the historical embedding. Picking different behaviors for 
stimulation is how this effect transfers to the control task, making the stimulation type a readout of the 
individuality of the cell. 
The avenue of finding model's representation of the cell state would benefit from further analysis. 
Current evidence of continuum of response suggests a single discriminating parameter responsible for 
the diversity, but a difference in  experimental setup and objective function might paint a more 
complex picture.   
Another angle of approach could be an analysis of model's embeddings, and whether they can be used to classify
the response actually picked by the controller reliably. 

// Sensitivity drift
Over the course of the experiment we noticed a gradual shift in quality of the tracking (@sensitivity-decline),
as well as the amount of light energy used. 
Its cause is unknown, with probable candidates being transcriptional feedback, receptor internalization 
and pathway desensitization. 
The conroller learns to deal with the decrease in sensitivity by increasing light budget in a reactive manner,
while adapting the running encoder by integrating the newly inflated stimulations. 
An example way to integrate such pathway-level phenomena in future work would be to wire it at the level of the
model architecture, for example as an explicit time axis between observations. Such operation would not only
increase awareness of drift, but also could serve as a way to encode external events relevant to the 
internal state of the cell (time of starvation, etc), and could aid in disentangling time-based phenomena from
stimulation-driven ones.
Future work should also consider measuring not only the drift but also recovery, and how it affects 
contrallability metrics. 





// - Sensitivity decline explanation;
// 	- photobleaching
// 	- transcriptional feedback
// 	- receptor internalization
// Fields of view with no light - do they have different characteristics? 


== Interpreting the sensitivity drift

#todo[the candidate readings: receptor internalisation and downregulation;
photobleaching or phototoxicity of the construct; medium exhaustion over a
12 h starved run; adaptation in the pathway itself. fig-decline bears on
which of these can be told apart with the data in hand, and which cannot.]

== Impact of resting CNR states on rigid objectives

Objective during the current experiments was pre-set. During some experiments,
the baseline starting CNR level varies to the point, where #todo[Difficulty with designing and conducting good experiments because of a varying pre-stimulation baseline CNR.]

// ═════════════════════════════════════════════════════════════════════════════
//  FUTURE WORK
// ═════════════════════════════════════════════════════════════════════════════

= Future work


- Make time an explicit axis in the model, either in the conditioning or as a
  feature (`time_since_last_measurement`). This makes it possible to learn
  multiple timescales, and may do better on long-range effects such as receptor
  internalisation or transcriptional feedback.
- Extend into the spatial dimension, possibly with a hierarchical model.
- Use the model to help fit a hybrid approach — a neural ODE, a
  physics-informed network, or similar.
- Use it to quantify the controllability of systems of this kind, and to ask
  whether the control drift itself can be influenced.

```
In order to lower the data requirements and to improve the prediction accuracy and thus the
control performance, incoming sensor data are used to update the RNN online. 
```

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
  [Every live run scored on the same two gates. (a) achieved
   cadence, median to p90, against the 1 min interval the model was trained on;
   seven runs slipped, from 1.16 to 5.72 min per frame. (b) share of closed-loop
   cell-frames sitting on the top rung of their own field's ladder, annotated
   with the ladders issued. Saturation must be measured per field: v14--v16 gave
   their closed-loop fields a 150 ms ladder while driving their open-loop fields
   to 600 ms, so a run-wide figure understates v16 by more than a factor of ten
   (5% against 73%). Three runs of nineteen clear both gates (v21, v23, v24);
   v16 and v19 hold cadence but saturate 73% and 32% of the time; v22 is
   excluded for a mis-set objective.],
  "fig-ledger",
)

#thesisfig(
  "arm-tracks",
  [A raw plot of all the admissable experiments. Solid lines are median CNR of a given arm of experiment. Shaded parts represent IQR ],
  "experiment tracks",
)


#thesisfig(
  "reachability-runs",
  [#todo[Every cell in every experiment, sorted by initial resting CNR. Colored lines represent p95 of its CNR. Red vertical line shows median of demanded CNR in a given experiment, while the vertical shaded bar stands for demand IQR] ],
  "reachability-runs",
)

#todo[park supplementary panels here as they are cut from the main text]

// ═════════════════════════════════════════════════════════════════════════════
//  BIBLIOGRAPHY
// ═════════════════════════════════════════════════════════════════════════════

#set heading(numbering: none)
#bibliography("refs.bib", style: "nature", title: "References")
