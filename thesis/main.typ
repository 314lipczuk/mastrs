
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
/// `placement: auto` makes the figure a float. Without it a figure that does not
/// fit in the remaining space pushes to the next page and leaves the rest of the
/// current one blank; eight pages ended that way. Floats let the text close up
/// behind them, at the cost of a figure sometimes landing a page from its first
/// mention.
/// `float: false` pins a figure where it is written. Needed whenever a float
/// would cross a heading: a floated figure can be lifted above the section it
/// belongs to, which is how the loss curves ended up ahead of Appendix A.1.
#let thesisfig(path, caption, label-name, float: true) = {
  let p = if path.contains(".") { path } else { path + FIG-EXT }
  [#figure(
      image("figures/" + p, width: 100%),
      caption: caption,
      kind: image,
      placement: if float { auto } else { none },
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
  text(size: 10.5pt)[08.09.2026]
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

ERK signalling carries information in its dynamics rather than its level, and cells given
identical input respond over a continuous range, so a quantity measured on the population
describes no cell in particular. Steering individual cells therefore means predicting each
one separately and acting on it separately.

This thesis builds such a loop and runs it on live cells. A recurrent model is trained on
6.6 million frames of single-cell optogenetic experiments to forecast a cell's ERK activity,
read out as a cytoplasm-to-nucleus ratio (CNR), in response to a proposed sequence of light. It
returns a distribution rather than a point, through a mixture density head, and a
model-predictive controller replans for every cell at every frame of a twelve-hour
experiment, scoring candidate dose sequences on the predicted mean.

Offline, the model forecasts eight minutes ahead with a root-mean-square error of 0.136 CNR
against 0.223 for assuming the cell does not move, it needs the cell's own history rather
than its present level to do so, and its stated uncertainty is calibrated in the sense that
predicted spread tracks realised error across a tenfold range. These properties survive the
move to the microscope.

The control results are narrower. Closed-loop planning held a reachable demand better than
constant illumination and better than darkness, but the closed-loop arms spent 1.87 times
the constant arm's light, so dose and feedback are not separated by that comparison.
Planning per cell rather than sharing one dose across a group was both cheaper and closer in
three fields of four, which at that sample size is a direction rather than a result.

The limits are themselves findings. Responsiveness declines across a run; the resting level
drifts downward by 0.021 CNR per hour with no light at all; and the population's resting
spread is wider than light can move the median cell, so a demand shared across cells is
unreachable for a large fraction of them by construction.

// ═════════════════════════════════════════════════════════════════════════════
//  INTRODUCTION
// ═════════════════════════════════════════════════════════════════════════════

= Introduction

This thesis closes a control loop around individual cells. The loop has three parts: a
learned model that predicts how a single cell's ERK activity will respond to light, a
controller that plans stimulation separately for every cell in the field, and a reference
trajectory the cells are asked to follow. The planning is per-cell; the reference is one
demand shared across the run.

A loop of this kind, with a learned predictor in it, has been demonstrated for gene
expression @Lugagne2024, which unfolds over hours. A signalling cascade moves in minutes, which leaves the loop far
less room: it must decide for every cell between one frame and the next, and through an
actuator that only ever pushes activity up.

== ERK dynamics and cell fate

The MAPK/ERK pathway plays a key role in cellular proliferation signalling, utilising dynamics rather 
than stable states to control cell fate decisions @Ryu2015 @PurvisLahav2013 @GagliardiPertz2024.
In PC12 cells, a sustained pulse of pathway activation drives differentiation while a
transient pulse drives proliferation @Marshall1995; modulating the frequency of
activation alone can rewire fate decisions @Ryu2015 @Albeck2013. The principle is not
confined to this pathway: in p53, converting a naturally pulsed response into a sustained
one switches the downstream programme and the fate that follows, with no change in the
amount of signal @Purvis2012.

Cells partake in complex spatiotemporal population-level phenomena such as ERK waves
@Aoki2017, which propagate through epidermis in vivo and speed wound healing
@Hiratsuka2015, and which an apoptotic cell sends into its neighbours to protect them from
dying with it @Gagliardi2021. Cancer cells act through the same channel: oncogene-expressing
cells release ligands that drive ERK waves in surrounding wild-type tissue, changing the
behaviour of cells that carry no mutation themselves @Aikin2020.


== Why the population average is not enough

A common problem in studying biological systems is that a quantity easily modelled at
the level of a population statistic need not describe any individual in it
@Altschuler2010. The
difficulty is not that the average is imprecise; it is that the events of interest
happen in single cells. A cell differentiates or it does not, commits to a cycle or
does not, dies or survives. An average over such events describes a state that no cell
need ever occupy.

For ERK this matters because the responses really are diverse. Genetically identical
cells differ in protein abundance in ways that precede any stimulus @Spencer2009. That
shows up directly in this preparation: cells receiving an identical pulse train respond
over a wide and continuous range, with a substantial minority barely moving at all
(@fig-heterogeneity). The spread there is measured in each cell's own baseline units,
because the question is how far the same light moves a cell, not where it starts from.

The consequence for control is direct, though the loop itself is not described until
@sec-closed-loop. A controller that steers a population average can drive that average onto
a target while leaving the individual cells further from it than before, and it has no way
to tell the two outcomes apart. Telling them apart requires
measuring each cell and dosing each cell separately, which is what the loop built here
does. Acting on the difference in full would require more: the target itself would have to
be written per cell. It is not, in any experiment reported here, and that limit is
returned to in the Discussion.

#thesisfig(
  "heterogeneity",
  [The population average is not a cell. 7,141 cells receiving the same pulse,
    sorted by response. The response is continuously distributed over more
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

A work by Klumpe et al. @Klumpe2023 showed that deep neural networks were able to infer the underlying dynamics of a cell response
even in the presence of measurement noise and stochasticity in the biochemical reactions.

// brief: not explaining mechanisms, but uncovering more complex behaviour and
// learning through interactions.

== Closed-loop control as an instrument <sec-closed-loop>

Approaches such as optogenetics can be used to control cellular processes. The pairing of
an optogenetic actuator with a live ERK biosensor is established in this system
@Dessauges2022, although as an open-loop perturbation rather than inside a feedback loop. 
Optogenetics is well suited to fine-grained control over dynamics. 
Light can be delivered with millisecond precision, targeted at subcellular resolution, and acts reversibly.

An important property of this actuator is that light only drives ERK activity
up, and nothing drives it down. A cell descends only by its own decay, so a demand below
where a cell already rests is not hard to reach but unreachable, and what each cell can
be asked to do is fixed by where it happens to sit when the run begins.

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
disqualifying. A more severe objection is that a fitted ODE cannot
follow a cell whose response changes over the run without being refit. A model learned
from observational data carries no such commitment, and that is the class used here.

Deep MPC has been demonstrated for controlling gene expression in a synthetic system: Lugagne et al. @Lugagne2024 steered expression levels
in thousands of single cells under blue light, planning with a neural predictor.
Gene expression, however, is a slow readout: it unfolds over hours on a transcriptional
timescale.

The difference in timescale is important. Transcription integrates over hours, so a
controller for gene expression has room to measure, plan and wait. ERK activity moves in
minutes @Albeck2013: the reporter used here responds to a pulse within a few frames and relaxes on a
comparable scale, so the loop must close inside the one-minute acquisition interval and
its horizon spans half an hour rather than a day. The lag between commanding light
and seeing the cell move is a sizeable fraction of that horizon, which is what makes the
control predictive rather than reactive: there is no error to respond to until it is too
late to correct cheaply.

This thesis builds a loop for that regime and runs it on live cells. A model is learned
from observational data to forecast one cell's ERK response to a proposed light sequence; a
model-predictive controller plans against it, for every cell in the field, at every frame;
and the closed loop is tested in live experiments against the two comparisons that decide
whether the complexity earns its place, namely the same light delivered without feedback,
and one dose shared across cells instead of chosen per cell.

There is a longer-range motivation for working at this resolution. Cancer already
demonstrates that a small subpopulation can set the signalling behaviour of the tissue
around it. The question this suggests is whether such a population could be designed
deliberately and driven, using a few individually controlled cells to influence the
behaviour of the larger tissue they sit in. That is well beyond what is attempted here, but
it is what makes single-cell control worth building rather than population-level control.

= Materials and methods

== OptoEGFR cell line and culture

A previously established NIH3T3 mouse fibroblast cell line (ATCC CRL-1658) stably
expressing optoEGFR-mCitrine together with the ERK kinase translocation reporter
ERK-KTR-mScarlet3 @erkktr and H2B-miRFP670nano3 was used for all experiments. OptoEGFR and the downstream biosensors were expressed under
CAG promoters. Cells were grown and maintained in Dulbecco's Modified Eagle's Medium, high
glucose (Sigma-Aldrich \#D5671), supplemented with 10% (v/v) fetal bovine serum,
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
performed using the following excitation/emission configurations: H2B-miRFP670nano3
640 nm LED, Lumencor 645/30x excitation filter, Chroma 89100bs dichroic mirror and
Chroma ET705/72m emission filter; ERK-KTR-mScarlet3 555 nm LED, Lumencor 575/25x
excitation filter, Chroma 89100bs dichroic mirror and Chroma ET632/60m emission filter;
and optoEGFR-mCitrine 508 nm LED, Lumencor ET500/20x excitation filter, Chroma 69008bs
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
configurations: H2B-miRFP670nano3 640 nm LED, Lumencor 645/30x excitation filter, two
consecutive Semrock FF421/491/567/659/776-DI01 dichroic mirrors and FF01-441/511/593/684/817-25
emission filter; ERK-KTR-mScarlet3 555 nm LED, Lumencor 575/25x excitation filter, two
consecutive Semrock FF421/491/567/659/776-DI01 dichroic mirrors and Chroma 59022m
emission filter; and optoEGFR-mCitrine 477 nm Laser, Semrock FF01-391/477/549/639/741
excitation filter, two consecutive Semrock FF421/491/567/659/776-DI01 dichroic mirrors
and Semrock FF01-511/20-25 emission filter.

Targeted optogenetic stimulation of optoEGFR was done using the 470 nm LED of the
Spectra X LED Light Engine with a 470/40 excitation filter (at 10% power, corresponding
to 3499 µW at sample) through a Mightex Polygon 1000 DMD and one Semrock
FF421/491/567/659/776-DI01 dichroic mirror.

== Automated acquisition and stimulation

Image acquisition and optogenetic stimulation were controlled using the FARO software
framework @Hinderling2025. Blue-light stimulation was controlled by illumination intensity, exposure
duration and timing relative to image acquisition. For the targeted experiments, the DMD
was used to apply the cell-defined spatial and temporal optoEGFR stimulation pattern
while ERK-KTR and H2B fluorescence were recorded.

== Image analysis

Cell nuclei were segmented using cellpose (version 4) @cellpose with a custom-trained model.
Cytoplasmic ring masks were generated by binary dilation of nuclear masks by four pixels
using scikit-image @scikit-image. Nuclear area, centroid position and median ERK-KTR fluorescence
intensities in nuclear and cytoplasmic compartments were extracted for each cell. ERK
activity was quantified as the cytoplasm-to-nucleus fluorescence ratio (CNR). Cell
identities were tracked across frames using trackpy @trackpy, with a maximum
allowed displacement of 50 pixels between consecutive frames.

== Feature engineering

The model receives five channels per cell per frame, and nothing else: CNR, which is both
the quantity being predicted and the model's own most informative input; fluence, the
light delivered to that cell on that frame; nuclear area; the number of other cells within
a 200 px radius; and optoRTK expression, a static per-cell value measured once before the
run.

Stimulation reaches the raw data distributed across three quantities: the commanded
exposure, the LED power setting, and an experiment-level variable of which stimulation
hardware and filters were in use. None of them is comparable across setups on its own.
They are therefore collapsed into a single physical quantity, the radiant exposure
received by the cell per unit area, referred to throughout as fluence and computed from
each microscope's own power calibration curve (@light-dose-calculation). This is what allows
experiments from two setups to enter one training corpus.

The optoRTK expression channel is worth more detail.
The biosensor for this readout has an activation spectrum overlapping that of the Cry2
domain of the optogenetic construct, so any readout of expression also activates the MAPK
pathway. The assumption we rely on is that receptor expression changes on a far longer
timescale than MAPK dynamics, which licenses measuring it once before the experiment
begins and then waiting for the pathway to return to rest before starting. The mCitrine
channel is imaged once more at the end of the run; this plays no part in control, but is
useful when accounting for how individual cells behaved during it. Expression is not
comparable in absolute units between sessions, so what enters the model is a within-session
rank rather than a rescaled intensity.

#thesisfig(
  "expression-normalisation",
  [optoRTK expression is ranked within a session rather than rescaled, because sessions
   differ in the shape of the distribution and not only in its scale. (a) Every session
   divided by its own median; the bold curve is the largest session. If the sessions
   differed only in scale the curves would coincide, and they do not: the ratio of the
   90th to the 10th percentile still ranges from 1.57 to 3.70. (b) Quantiles of each
   session against those of the largest, matched at the median. The dashed line is
   agreement after rescaling, and curvature away from it cannot be removed by any scale
   factor, in raw units or in log, which is what leaves a rank as the only transform that
   aligns them. It matters because the model consumes this channel standardised against
   frozen training statistics, so a distribution that changes shape between sessions would
   make the same input value mean something different in each; a within-session rank means
   the same thing in every session by construction.],
  "fig-expression-norm",
)

Nuclear area and the local cell count describe the cell's size and its immediate
surroundings. Both were included on the strength of an early feature comparison, run while
the model was still being developed, in which every candidate channel carried some weight.
On the final corpus that stopped being true of the crowding count. Measured against a model
that has already watched the cell, it adds an R#super[2] of 0.000 after three minutes of
observation and 0.0004 after thirty (@sec-temporal-context). It stays in the input because
the channel set was fixed before that result was in, and the checkpoint used for every live
experiment reported here carries it.

Two checkpoints of this model appear in what follows. They were trained on the same corpus
and the same split and differ in one channel. The one every live experiment loaded is the
one described above. An earlier sibling carries the number of cells in the field, its
field density, in place of nuclear area, and it supplied the held-out forecast evaluation
and the encoder-context ablation of @sec-temporal-context, which is why field density is
reported there and nuclear area is not. The two are matched on everything those
comparisons rest on: 115,559 parameters each, the same 57,954 training and 7,237 held-out
samples, and a held-out mean absolute error of 0.085 CNR in both. They also agree on what
the channels are worth. Permuting a channel and remeasuring the validation likelihood
costs 0.95 to 0.97 for the delivered light and 0.059 for expression rank in both models,
while the two spatial channels cost 0.005 to 0.007 and nuclear area 0.012.

A considerably larger set of derived stimulation statistics was built and tested before
this one: time since the last pulse, fast and slow exponential moving averages of
delivered light, a count of pulses within a trailing window, the integral of fluence since
the start of the experiment, and the OLS slope of fluence over recent frames. Features
learned directly from the images were also considered and never built. None of the derived
statistics survived, for structural reasons rather than empirical. Once the encoder
was given the cell's entire history rather than a fixed window, every one of them became a
function of inputs the encoder already holds, and a recurrent network can compute a moving
average or a time-since-event counter for itself where that is useful. The ablation found
them carrying no weight and they were removed. What the model is given is deliberately
minimal, and the work of remembering the past is left to the encoder.

== Dataset

The data used to train the predictive model was assembled by pooling experiments that had
been run for several different purposes, and these fall into three families. Roughly three
quarters of the corpus, 4.92 M frames across four experiments, comes from Bayesian
optimisation runs searching for a stimulation pattern that would induce oscillation. About
a fifth, 1.44 M frames across three experiments, comes from bulk sweeps of the input space,
built to cover it rather than to optimise toward any particular response. The remaining
4%, 0.28 M frames, comes from short characterisation experiments with hardwired patterns
and varying commanded exposure, run to describe the behaviour of the optogenetic construct
itself rather than to train anything.

#thesisfig(
  "dataset-overview",
  [The training corpus: where the data comes from, one example trajectory with its
   delivered exposures, and the population response of one experiment from each family.],
  "fig-dataset",
)

All this data was filtered using a standard procedure involving removing cells that were
segmented but not alive, incorrectly segmented cells, and visual anomalies resulting in
non-meaningful features.

This final composition of all the experiment data resulted in a dataset of 72,441 cells
and 6.63 M frames.

== Model architecture

A deep learning model was designed for the CNR prediction task, with the goal of being
quick enough to be run in real time for many cells in parallel.

#thesisfig(
  "model-schematic",
  [The predictive model. (a) One step of the computation: the encoder reads every past
   frame of one cell on five channels, the decoder is a separate recurrent network stepped
   on just two numbers, the CNR fed back from the previous step and the light commanded for
   this one, and the head returns a mixture of three Gaussians rather than a point. The
   commanded dose enters three times, as a decoder input, as the FiLM conditioning that
   scales and shifts the decoder output, and again concatenated at the trunk input. (b) The
   same decoder unrolled over the 30-frame horizon. The light row is the candidate plan
   being scored, known ahead of time because it is what the controller is choosing, and the
   predicted spread widens because each step is conditioned on the previous step's
   prediction rather than on a measurement. Dropout and the learned per-step variance
   offset are part of the model and are omitted from the drawing.],
  "fig-model-schematic",
)

This motivated the decision to split the model into four parts: an encoder, a decoder, an
MLP trunk and a prediction head. The encoder is an LSTM network that reads all the
features from previous timepoints of a given cell and returns a hidden state. When running
in real time that state is persisted in memory, so when the microscope provides the next
frame we run a single encoder step on the state we already hold and immediately obtain the
next one, without re-encoding the entire past. The decoder is a second LSTM, initialised
from the encoder's state, and it is stepped once for each frame of the planning horizon.
Its input at each step is only the CNR fed back from the previous step and the light
commanded for this one, which are the two quantities that are known while a candidate plan
is being scored. Its output is modulated by a FiLM layer @film conditioned on that commanded
dose, then concatenated with the dose again and passed through the MLP trunk to the
prediction head. The dose therefore reaches the prediction by three routes rather than
having to survive in the recurrent state alone.

Being able to take model uncertainty into account was deemed
important, so the prediction head returns a distribution rather than a point. It is a
mixture density network @mdn: the network outputs a mixture of Gaussians, each described by a
mean, a variance and a weight, which we constrain into meaningful values. The number of
components is a hyperparameter, and lets the head express a prediction for a
non-homogeneous population. It is trained by minimising the negative log likelihood of the
observed CNR under that mixture, on the running minibatch.

The head emits one such mixture at every step of the planning horizon, rather than a
single value one frame ahead.

== Training methodology

The corpus is split by cell, 80/10/10 into training, validation and test, stratified within
each source experiment so that every protocol is represented in all three parts. A cell
belongs to exactly one split, so no held-out forecast is ever made from a history the model
was fitted on. The split is drawn once from a fixed seed and reused across every training
run reported here, which is what makes the feature comparisons in @sec-temporal-context
comparisons between feature sets rather than between draws. Standardisation constants are
computed on the training split alone and then frozen onto the model, so that evaluation and
live deployment use the same constants the model was fitted under.

The model reported here carries 115,559 parameters: hidden width 64 in both recurrent
networks, two layers each, a three-layer trunk of the same width with GELU activations and
dropout 0.1, and a three-component mixture head. It was trained for 300 epochs with Adam at
a learning rate of $10^(-3)$, weight decay $10^(-4)$ and a cosine schedule decaying to
$10^(-5)$, in batches of 256, with gradients clipped to unit norm and early stopping on
validation loss at a patience of 40 epochs. The best validation loss fell at epoch 291,
although 95% of the improvement was reached by epoch 94. Teacher forcing was annealed
linearly from 1 to 0 across the first 30% of training @scheduledsampling. Each batch drew its horizon length
uniformly between 3 and 30 frames, and cells were drawn by a response-magnitude stratified
sampler over four strata. Training used 57,954 samples against 7,250 for validation and took
2.1 hours on a single RTX 2080 Ti.

A single forward pass through the model produces an uncertainty-aware prediction of CNR at
time $t+1$. Frames arrive one per minute, while the ERK dynamics of interest play out over
many minutes, so a single-step forecast is too short a horizon to plan against. We
therefore extended the prediction by autoregressive unrolling on the model's own output. The model, given features from the past,
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
The problem with this approach is that it makes training noisy and unstable: there is no
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

== Control scheme for model-predictive control

The model predictive control approach was based on the extending-horizon model. Before
the start of the experiment, a policy containing an objective function was loaded.

#thesisfig(
  "schematic",
  [The closed loop, once per frame. FARO drives the microscope, segments and tracks the
   cells and extracts their features; the inference server takes one encoder step per cell
   on the state it has been carrying, runs the controller, and returns a dose per cell,
   which FARO turns into a DMD stimulation mask. The whole circuit completes inside the
   one-minute acquisition interval, and repeats for the length of the run.],
  "fig-pipeline",
)

What the controller chooses, once per frame for each cell, is the duration of the blue
pulse that cell receives. LED power is held fixed for the length of a run, so exposure time
is the only free variable on the actuator, and it maps onto the fluence the model consumes
through the calibration described in @light-dose-calculation. Exposure is continuous in
principle: the DMD can hold a cell's pixels open for any duration the acquisition loop
leaves room for.

It is nonetheless collapsed into a small set of discrete levels, the _dose ladder_, and
treated as a categorical variable, for two reasons. On the microscope, discrete levels mean switching
between a fixed set of masks rather than composing a new one for every cell on every
frame. In the search, it replaces an optimisation over a continuous input sequence with a
categorical distribution over $k$ levels at each of the $L$ horizon steps. That does not
make the space small, since $k^L$ is still astronomically large, but it makes it something
a sampler can be defined on and refit to, which is what the planner does.

The ladder is fixed for the whole of a run and shared by every arm within it, so arms are
always comparable inside a run, but it was not the same in every run. Runs v10, v11, v16
and v19 used five levels: 0, 20, 45, 85 and 150 ms. From v21 onward a sixth level of 300 ms
was added, and v21, v23 and v24 all use it. Two arms sit outside their own run's ladder by
design. The open-loop dose probe in v16 steps through 0, 85, 150, 300 and 600 ms, exceeding
what that run's closed-loop fields could choose from, because its purpose is to
characterise the actuator rather than to control anything. The constant arm in v24 delivers
a flat 60 ms, set from the mean dose of v23's closed-loop arms and deliberately placed
between settings so that it is not a choice the closed-loop arms could have made. Comparisons
of delivered light between runs therefore have to account for which ladder they were drawn
from.

Inference begins from the per-cell encoder state, which is persisted across frames and
zero-filled at the start of a run. Each new frame's features are passed through the
encoder, the resulting state is stored against that cell, and planning proceeds from it.

The planner searches for a dose sequence by the cross-entropy method, shown in
@fig-mpc-schematic. It holds a categorical distribution over the $k$ ladder levels at each
of the 30 horizon steps, uniform to begin with. Each pass draws 512 candidate sequences
per cell from that distribution and adds the $k$ constant-dose sequences, which are always
evaluated so the search cannot return a plan worse than the best constant dose. All of
them are rolled through the decoder in a single batched pass and scored. The cheapest
eighth, 64 sequences, are kept as elites, and the distribution is refit to the levels those
elites used, which for a categorical is the closed-form minimiser of the KL divergence to
their empirical distribution. A tenth of the uniform distribution is then mixed back in, so
that no step can collapse onto a single level and stop being searched. Three passes are
run, at a cost of 1,554 rollouts per cell per frame, and only the first step of the winning
sequence is applied before the horizon slides forward and the search is repeated. The
sampler is seeded on the frame index, so replaying a recorded run reproduces its plans
exactly.

#thesisfig(
  "mpc-schematic-alt",
  [The cross-entropy search, run on one real cell. Rows are the three passes; columns
   follow a single pass from left to right. The controller holds a categorical
   distribution over the six ladder settings at each of the thirty horizon steps, uniform to
   begin with; it draws 512 plans from it and adds the six constant-dose plans, which are
   always evaluated so the search can never come out worse than the best constant dose;
   all 518 are rolled through the decoder in one batch and scored against the demand. The
   cheapest 64 are the elites, the distribution is refit to them with a tenth of uniform
   mixed back in, and the pass repeats. Each column shares its scale down the three rows,
   so the distribution visibly peaks, the plans and their predicted trajectories collapse
   together, and the cost histogram slides left. Only the winning plan's first step is
   applied. The cost axis is truncated on the right, so the all-dark plan sits off it.],
  "fig-mpc-schematic",
)

The score is a tracking term plus a price on light. Writing $hat(c)_h$ for the predicted CNR
at horizon step $h$, $r_h$ for the reference there, $u_h$ for the exposure the plan commands
and $u_max$ for the largest setting on the ladder,

$ J(u) = 1/H sum_(h=1)^H (hat(c)_h - r_h)^2 + lambda_"dose" 1/H sum_(h=1)^H u_h / u_max $

and the plan with the lowest $J$ wins. Every run reported here uses
$lambda_"dose" = 0.089$. A third term, $lambda_"move"$ on the mean absolute change in
exposure between consecutive frames, is available and was used in v10 and v11; it is zero
elsewhere.

Two scoring kernels are implemented. The squared-error kernel above reads only the
predictive mean; a band kernel instead prices the probability that a plan leaves a band
around the reference, and is the one that consumes the head's full mixture. The band kernel
was tested as one arm of v10 and v11 and was not used in the runs that followed, so every
result reported here is produced by scoring on the mean.

A second reason, independent of what those two runs showed, was that with only a handful of
live experiments there was no evidence that the model's offline calibration carried into a
feedback regime, where it is scored against the consequences of its own earlier choices.
Scoring on the mean was the conservative option until enough live data existed to check. The
capability remains in place: the head is a mixture throughout, and the kernel is one line of
the policy.

//One consequence is worth stating plainly. The decoder feeds its own predictive mean forward
//at each horizon step, so a multimodal prediction cannot propagate: whatever the mixture
//expresses at one step is collapsed to a single number before the next. The three components
//can represent a bimodal one-step-ahead prediction, but not a bifurcating trajectory.

// ═════════════════════════════════════════════════════════════════════════════
//  Experiments
// ═════════════════════════════════════════════════════════════════════════════

== Live experiments <sec-live-experiments>

Experiments on live cells were designed both to probe the biological system and to test
the MPC pipeline. A standard experiment ran for 12 hours over 8 to 12 fields of view.

Nineteen runs were attempted in total and seven are used here. Each run was checked against
five conditions: whether a controller policy ran at all, whether the run completed, whether
the one-minute cadence held, whether the controller avoided spending the run pinned at the
largest setting its ladder offered, and whether the objective it was given was reachable by
the cells. The first two and the last are read off the run records; the two that have to be
measured, cadence and saturation, are plotted for every run in @fig-ledger. Three runs pass all five
and are called _admissible_: v21, v23 and v24. Four more fail exactly one condition in a
way that bounds rather than voids them, and are used within that bound. v10 and v11
slipped their cadence, so their arms may be compared with each other but their absolute
errors may not be compared across runs. v16 and v19 saturated, so they support everything
except claims about what happens at the top of the ladder. Of the remaining twelve, v22
held its cadence and never saturated, but was given a mis-set objective that left a third
of its cells unable to reach the demand at any dose; the rest either ran no controller,
did not complete, or failed more than one condition.

Cadence slip is a failure mode in which the microscope's full rotation through all fields
takes longer than the intended minute. The earliest runs met it, through a misconfiguration
of the acquisition computer: v10 and v11 ran at 85 s and 69 s per frame instead of 60 s. The
slip does not void those runs, but it is not neutral either. Every arm within a run shared
the same cadence, so comparisons between arms are unaffected. What does not transfer is the
absolute error. The model was trained on frames one minute apart, so each of its thirty-step
horizons spanned 42% and 16% more real time than any horizon it had been fitted on, and it
was planning against dynamics that unfolded correspondingly faster than it expected. The
tracking errors from v10 and v11 are therefore comparable with the other arms of their own
run and not with runs that held one frame per minute.

Several levels of structure recur throughout, and the distinction between them matters for
how the experiments are analysed. A _field of view_ is one imaging position on the plate;
a run carries eight to twelve of them, and each field keeps the same controller
configuration for the whole experiment. An _arm_ is the set of fields sharing one
configuration, typically two of them, and a run is built to compare its arms against each
other. A _block_ is one repeat of the objective in time, typically a run-up followed by a
demand, and a run carries nine to twelve of them in sequence. The run-up drives every cell
onto one fixed reference level, the _anchor_, and holds it there; each block opens and
closes on that level, so a block is entered from a known state rather than from wherever
the previous block left the cells. Every field is imaged in every block, so fields and
blocks are crossed rather than nested: the objective at a given minute is the same in all fields, and the
controller configuration is the same in all blocks.

This has a direct consequence for what counts as a replicate. A treatment applied to
fields, such as the length of the unscored window, is replicated by fields, typically
two of them. A treatment applied to blocks, such as which demand pattern is being asked
for, is replicated by blocks, three of them per pattern. The individual cell is a
replicate of neither: a cell lives in one field and survives many blocks, so its
measurements are repeated observations rather than independent ones.

Unless stated otherwise, every per-cell quantity reported from a live run is computed on
the scored window, which begins at frame 76, once the cells have settled after the start of
the run, and covers cells with at least 120 scored frames inside it. Tracking error is the root-mean-square
difference between a cell's CNR and the reference that cell was given, and delivered
light is that cell's mean commanded exposure over the same frames. Where a number is a
field-level or arm-level summary rather than a per-cell one, it is the mean or median
over the cells in that unit, and the text says which. Cell counts therefore differ
between analyses, because some of them require more: a resting estimate needs frames
with no light in the preceding five, which is a stricter condition than the scored
window on its own. Confidence intervals on summary figures are 95% percentile intervals
from a nonparametric bootstrap of 2,000 resamples over the units being summarised, drawn
from a fixed seed.

References are written in absolute CNR rather than in each cell's own baseline units.
Normalising to a cell's own baseline during a run would mean estimating that baseline from
its early frames, so any segmentation error in those frames would be carried into the
reference the controller tracks for the rest of the experiment; an absolute reference keeps
segmentation noise in the measurement alone.

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
      [Return every cell to the anchor before each demand block, so the loop must
       reset the population before it tracks anything.],
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

=== Search for useful controller mechanisms

The predictive model can be evaluated offline, by measuring its error on data already
collected. The controller cannot: what it is judged on is the trajectory that follows from
its own choices, and the trajectory a different controller would have produced is not
available. The first live experiment, v10, was therefore designed to ask whether the base
MPC controller is useful at all, and whether two variations on it help. Four arms were run,
identical in model, objective and dose ladder, and differing only in how candidate plans
were generated and scored.

Arm 1 was a control that does not use the full search. It evaluated one future per setting on
the ladder, five in that run, holding that single exposure fixed across all 30 frames of
the horizon, and applied the first step of whichever came out best. It therefore has the
model and the objective but no search over sequences.

Arm 2 was the base configuration: candidate sequences sampled from an initially uniform
distribution over the settings and scored by squared error against the reference.

Arm 3 added a penalty on large changes in exposure between consecutive frames, which
favours smooth stimulation patterns over ones that alternate between distant settings.

Arm 4 kept that penalty and replaced the scoring kernel with a band kernel, which prices
the probability of leaving a band around the reference rather than the squared distance
from it.

The band kernel was included because control of this system is asymmetric. Light can only
drive CNR up, and it comes down only by waiting, so an overshoot is more costly than an
undershoot of the same size: an undershoot can be corrected on the next frame, while an
overshoot cannot be corrected at all except by losing time. Squared error is symmetric and
prices the two identically. A band kernel can be made more lenient below the reference than
above it.

=== Diversity of stimulation types found by the model

The encoder compresses a cell's history into a state, and that state could be used by the
controller in two quite different ways: to decide how much light a given cell needs, or to
decide what shape of stimulation suits it.
The first would show up as different doses for different cells, the second as different
patterns. This subsection describes the design used to look for the second.

The obstacle is the objective itself. A controller that is scored at every frame is being
asked one question repeatedly, namely how best to sit on the reference right now, and that
question has close to one answer. Frames before the interesting part of the objective
begins, such as a resting period preceding a demand, are scored on the same terms as the
demand itself, so the approach to the demand is as constrained as the demand.

The alternative is to leave some frames out of the cost entirely. A frame outside the
scored interval contributes nothing to a plan's score, so any behaviour during it is free,
and the controller may use that room to prepare a cell for the section it will be scored
on. Whether it uses the room at all, and whether different cells use it differently, is
then an empirical question.

The design that follows is a free window: a stretch of unscored frames placed immediately
before each scored objective, with the stimulation the controller chooses inside it as the
observable. Each run carries four arms, assigned to fields rather than to cells so that a
field keeps its window length for the whole experiment, two fields per arm. The first arm
is the control, scored throughout, so its run-up is constrained frame by frame in the
ordinary way. The other three are given progressively more unscored time.

The lengths were chosen against the lead time the actuator needs rather than against round
numbers. Driving flat out from the anchor, a cell takes about seven minutes to reach the
demand, and the loop's own dead time, the lag between commanding light and seeing the cell
move, accounts for three to five of those. A window shorter
than that lead is free on paper only, because light commanded inside it lands after the
demand has already opened; such an arm tests whether unscored frames help, not whether free
time does. Windows comfortably longer than the lead leave room to act, and room to
overshoot and come back. The two runs bracket the lead from different sides: v21 uses 0, 4,
10 and 20 free minutes, v23 uses 0, 8, 14 and 20.

Every arm is read on the same twenty-minute tail of the run-up, so what is compared across
arms is identical frames differing only in whether they were scored. Two questions can then
be put to that tail. Whether the controller uses the room at all, which is a question about
how much light it spends and when. And whether what it does there is a property of the cell
or of the moment, which is what decides whether "diverse strategies" means anything beyond
variability.

#thesisfig(
  "proposed-demands",
  [The free-window design: how much unscored time the controller is given before each
   demand opens.],
  "free window design",
)


== Comparing single-cell control, population-level control and open loop stimulation 

Experiment v24 was designed to compare single-cell MPC with population
level MPC against an open loop stimulation matched to experiment beforehand.

#thesisfig(
  "e2-design",
  [The v24 design. Arm 1's four fields are split inside the dish: cells are assigned in alternating blocks of four consecutive tracking indices, half to per-cell control and half to a single dose shared across the group, so 1a against 1b is paired within a field rather than compared across dishes. Arm 2 delivers a flat 60 ms, set from v23's closed-loop arm means; arm 3 receives no stimulation light at all. The nine blocks are a complete 3 × 3 of levels against rates, each combination once, in a counterbalanced order.  ],
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

We evaluated the trained model on held-out trajectories drawn from all of the open-loop
experiments: 7,237 forecasts, each thirty frames long. Error grows with lead time and then
stops. One minute ahead the RMSE is 0.045 CNR; at eight minutes it is 0.136; at thirty it
is 0.145. Each lead's $R^2$ is taken against the variance of the targets at that lead
rather than a pooled variance, because the spread of the targets itself shrinks with lead
time, from a standard deviation of 0.353 CNR at one minute to 0.309 at thirty; on that
basis the three leads give 0.98, 0.84 and 0.78. The reference that makes those numbers mean
anything is persistence, the forecast that a cell stays where it was last seen, which
reaches 0.089, 0.223 and 0.326 at the same three leads. The model's advantage over it
therefore widens with lead time, from roughly two-fold at one minute to 2.25-fold at
thirty, and the flattening
is one-sided: the model's own error is essentially unchanged after about six minutes while
persistence keeps growing. The live controller plans over the full thirty frames.

#thesisfig(
  "model-accuracy",
  [Forecasts from real context on held-out cells; error against lead time,
   compared with persistence (strategy where the forecast for future is that it will be the same as present ); predicted against observed at the control
   horizon.],
  "fig-model-performance",
)

#thesisfig(
  "forecast-examples",
  [A typical cell and one in a model's failure mode, forecast from evenly spaced and from the
   cell's own highest-error starting points respectively.],
  "fig-forecast-examples",
)

=== Dependence on temporal context <sec-temporal-context>

The model leans heavily on the encoded history of the cell. Forecast error falls 2.21 times
as the encoder is given more of the cell's own past, from a median absolute error of 0.122
CNR with one minute of context to 0.055 with sixty, and the curve is flat beyond roughly
twenty minutes (@fig-encoder). The model is not reading a cell's immediate state and
extrapolating from it; it is using a window of the past considerably longer than the
dynamics it is asked to predict.

What that history is worth can be measured against the alternative of simply measuring the
cell. Panel (c) of @fig-encoder asks, for each input the model receives, how much it adds
once the model has already watched the cell for a given time. Watching alone carries an
$R^2$ of 0.02 after three minutes and 0.91 after thirty. Against that rising baseline,
optoRTK expression rank is worth a great deal early and almost nothing late: it adds 0.233
at three minutes, more than the delivered light itself at 0.120, and 0.001 by thirty. The
spatial channels contribute at no point on that axis. Local crowding adds 0.000 at three
minutes and 0.0004 at thirty; field density adds 0.004 and 0.000. A covariate measured once
before the run is worth having only while there is nothing else to go on, and a description
of the neighbourhood is not worth having at all.

#thesisfig(
  "encoder-needs",
  [Forecast error falls 2.21#sym.times as the encoder is given more of the cell's own
   past, from a median absolute error of 0.122 CNR with one minute of context to 0.055
   with sixty, saturating around 20 minutes.],
  "fig-encoder",
)

That the past matters is not the same as the past of this particular cell mattering. To
separate them, cells from the same experiment were matched on their CNR level and the
encoder state of one was used to forecast the other's future, so that the level is
preserved and only the identity is swapped. For 89% of cells the swap produces a larger
error than the cell's own history does (@history-swap). Both runs used here are closed
loop, so a donor's history carries the donor's own light sequence as well as its identity,
and matching on level does not match on the light that produced it. The result is therefore
consistent with the encoder state being specific to the cell rather than a compressed
description of where it currently sits, without establishing it; settling that would need
donors drawn from an open-loop run, where every cell in a field received the identical
pulse sequence.

#thesisfig(
  "history-swap",
  [Replacing a cell's history with that of a level-matched donor roughly doubles forecast
   error, for 89% of cells. Both source runs are closed loop, so the donor's history
   carries its light sequence as well as its identity; the comparison is consistent with
   cell specificity rather than isolating it.],
  "history-swap",
)

// ------------------------------------------------------------------------------------
=== Calibrating uncertainty 
// ------------------------------------------------------------------------------------

// Outline brief: reliability — for every prediction and its confidence
// interval, count the observations that fall inside it. Does the 90% interval
// contain 90%? Does the 50%? Compare the full mixture against a collapsed
// single-Gaussian head. Is there symmetry in the tails — does the model tend to
// under- or over-estimate? Does coverage change with forecast lead time?

The quantity used to test this is the probability integral transform. For each observation,
the model's own predictive distribution is evaluated at the value that actually occurred,
giving the probability mass it had placed at or below that value. If the distributions are
right, those numbers are uniform on the unit interval: a well-calibrated forecaster is
surprised exactly as often as it said it would be. Departures from uniformity say what is
wrong rather than only that something is. Mass piling up at both ends means the intervals
are too narrow and the model is overconfident; mass in the middle means they are too wide;
mass in one tail alone means the distribution is displaced in that direction. The transform
suits this problem because every prediction here carries its own distribution, with its own
mixture weights, means and variances, so there is no single interval whose coverage could
be checked in its place.

The mixture head returns three components per step, so the transform was computed from the
mixture's own cumulative distribution function, over 7,237 forecast origins at all thirty
steps. The mixture's central intervals cover at close to their nominal rate. A single
Gaussian carrying the mixture's total spread, used as a comparison over the same forecasts,
is underconfident, so the extra components are doing work.

Coverage scans central intervals and is therefore blind to asymmetry. The raw transform
densities show the model generous in the lower tail: it anticipates more downward movement
than occurs, by 4.3 percentage points at the first step, falling to zero by step thirty.

A separate issue from accuracy itself, is whether the mixture's standard deviation is tracking factual error.
To do that, RMSE over predictions is plotted against prediction's standard deviation binned into deciles,
falling on an identity line: the predicted standard deviation matches the realised RMSE in magnitude and not only in rank.
Across deciles the predicted standard deviation spans roughly a tenfold range and the realised error tracks it, so σ discriminates between easy and hard predictions rather than reporting one width everywhere.

These are properties of the model rather than components of the control results that
follow. Every admissible run scores plans by squared error on the predictive mean, so the
spread evaluated here does not enter any plan cost.

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
Seven of the nineteen runs are used here: v10, v11, v16, v19, v21, v23 and v24. Which runs
qualify, and what each may be used for, is defined in @sec-live-experiments.

Runs v10 and v11 (@exp_v10_arm2) tested the base controller against two variants: a
penalty on changing the dose between frames, and a dead-band kernel that ignores small
deviations, at two frequency settings. They are the first runs in which the loop closed
and held for a full experiment, and they set the pattern the later runs follow.

#thesisfig(
  "v10-arms",
  [The v10 and v11 controller arms. (a) v10's base MPC arm: median CNR with the
   interquartile range, every cell aligned to its own phase group. The four groups are
   offset by 0, 10, 20 and 30 minutes on a 40-minute cycle, so without that alignment the
   references cancel and the oscillation disappears. The reference does not move across the
   run, while the amplitude the arm achieves falls: the early cycles reach the demanded
   level and the late ones fall short of it. (b) Per-cell tracking error for every arm in
   both runs, median with the interquartile range. Base MPC is lowest in both; the ordering
   beneath it is not stable between them, and in v10 the non-MPC constant-dose control beat
   both arms carrying a move penalty.],
  "exp_v10_arm2"
)

The controller holds the demanded frequency for the whole run, but not the demanded
amplitude. With the reference unchanged throughout, the median cell reaches the demanded
level on the first cycle and only half of it on the last. This is the first sight of
declining responsiveness, and the reason the later experiments were designed to measure it.
Base MPC gave the lowest per-cell tracking error in both runs, 0.193 CNR in v10 and 0.162 in
v11, and was used in every later experiment. Below the winner the ordering does not
replicate: in v10 the constant-dose control came second at 0.213, ahead of the move-penalty
arm at 0.232 and the band-kernel arm at 0.247, while in v11 it came last at 0.241, behind
0.198 and 0.229.

With the controller configuration settled, the later runs moved to a more varied set of
objectives. Before reading them, the offline evaluation has to be repeated on the
microscope: accuracy and calibration were measured again on four live runs sharing one
controller (@rig-calibration). Both transfer at the forecast horizon; the calibration
degrades in the mid-horizon range.
#thesisfig(
  "rig-calibration",
  [Accuracy and uncertainty comparison of the data from 4 live experiments running under the same controller to the results of offline evaluation.],
  "rig-calibration",
)

Coverage is worst between horizons 8 and 15: at fifteen frames the nominal 68% interval
covers 61% of live outcomes against 70% offline. It recovers at the wider intervals, where
the nominal 95% covers 93%. The direction of the error also reverses. Offline the intervals
were slightly too wide, covering 70% where 68% was claimed; live they are too narrow,
covering 61%.

In runs with a repeating block objective the response to stimulation flattened over the
course of the run (@sensitivity-decline).

Part of that flattening is not a change in how cells respond to light but a change in where
they sit. v24 carried two arms that make this separable. Its dark arm received no
stimulation light for the whole run, and over the eleven hours of the scored window the
median CNR of its 417 cells fell from 0.995 to 0.812, a fitted slope of 0.021 CNR per hour
($R^2 = 0.87$). The constant 60 ms arm fell at a similar rate, 0.026 CNR per hour, from
1.243 to 0.910. A decline of that size with the light switched off is a drift in the resting
state itself, and it sits underneath every lit arm as well.

This has a direct consequence for a demand written in absolute CNR and held fixed for twelve
hours. v24's three levels were 1.20, 1.36 and 1.52 throughout, while the population they
were asked of fell by roughly 0.23 CNR over the same window. The gap the controller had to
close therefore widened by about that much between the start of the run and its end, with
the objective unchanged, so a fixed absolute demand becomes progressively harder purely from
baseline drift.


#thesisfig(
  "sensitivity-decline",
  [Experiments with a repeating objective component: mean CNR (solid), goal (dotted), and 95% confidence interval for the first repeat and the last repeat of the pattern in the experiment. In every experiment except the 70 min arm of v19, the last cycle is consistently lower while taking up more light.],
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

Experimental outcome was better in experiments whose objective sat inside the cells' resting
distribution (@reach-and-tracking). The two runs whose demand fell well above the resting
median, v16 and v24, carry the two highest median tracking errors at 0.275 and 0.354 CNR,
while the runs whose demand fell within the resting spread track between 0.18 and 0.23.
With seven runs that is an ordering rather than a tested relationship. The population's
resting spread is wider than the light can move any one cell (@reachability-runs). What
counts as having moved a cell at all has to be judged against a measured null rather than
against the per-frame noise floor: v24's dark arm held 229 cells under no stimulation light
for twelve hours, and those cells still register a median apparent rise of 0.14 CNR above
their own resting level, with a 90th percentile of 0.30.

== Comparing levels of control
// ------------------------------------


Experiment v24 compares population-level feedback against per-cell feedback and against
open loop. Its demand was planned from the resting level and reach measured in v23, where
both sat unusually high, so v24 asked for more than most cells could give. The design is a
complete three-by-three of levels against rates, so three of its nine blocks sit at each
level; only 30% of cells ever reached the middle level and 10% the highest, and even the
lowest was out of reach for 41% of them (@e2-arms). Six of the nine blocks therefore asked
for something most of the population could not do. The same mismatch defeated the open-loop
constant arm, which was set to deliver the average light of an earlier run in order to land
at the same level: it delivers 60 ms at every frame, against a closed-loop average of
112.5 ms.

#thesisfig(
  "e2-arms",
  [ What the three lit groups achieved. A cell's ceiling in panel (b) is the 95th percentile of its own CNR over the run, which is what it actually reached under whatever light it was given. The resting lines are the dark arm, which never received any light. Panel (a) shows the median cell of each arm against the one reference all eight fields were given, with the interquartile band on the per-cell arm.],
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
failure mode described in the introduction appearing inside a single block of a single
run.


#thesisfig(
  "feedback-ladder-alt2",
  [(a) Every scored cell placed on tracking error, jittered within its arm; open rings are field medians and the black bar is the arm's median of those. Thin lines join the two halves of one dish, which is the paired 1a against 1b contrast. (b) What each arm spent to get there, against the 60 ms arm 2 was set to from v23's closed-loop means. The closed-loop arms spent roughly twice that, so the step from arm 2 to arm 1 is not at a matched dose.],
  "feedback-ladder",
)

The second comparison is between the two halves of the closed-loop arm, and it is the
only place in this work where individuation is isolated. Cells sharing a field were split
by a fixed rule, half planned individually and half receiving one dose computed for the
field as a whole, so the comparison is paired inside the FOV and every field-level
difference cancels within the pair. Per-cell dosing gave a median tracking error of 0.291
against 0.320 for the broadcast dose, on 109 against 118 ms of light per frame: closer,
and on less light. Three of the four fields favour per-cell dosing and one does not,
which at four fields is a direction rather than a result: an exact sign test gives
p = 0.63 two-sided, and 0.31 one-sided.

The third comparison is the full ladder across all eight fields. It is the strongest
result statistically and the weakest in interpretation. Field median tracking error
orders monotonically with how much feedback the arm had: 0.284, 0.288, 0.309 and 0.349
for the closed-loop fields, 0.375 and 0.399 for constant illumination, 0.470 and 0.486
for darkness. Eight fields in groups of four, two and two admit 420 distinct
relabellings, and the observed ordering carries an exact permutation p of 0.0048, with
Spearman rho of +0.93. The problem, however, is that across the whole run the
closed-loop fields spent 116 to 124 ms per frame against the constant arm's 60. Part of
this ordering is the light rather than the feedback, and this run cannot separate them.

Taken together, v24 supports two claims. A closed loop holds a reachable level better
than constant illumination and better than nothing. Dosing cells individually is at least
no worse than dosing them together, while using less light to do it. It supports no claim
about following a waveform pattern, and none about performance at a matched dose.

== Are there multiple distinct strategies of stimulation that controller undertakes?

Visualising the activations in free windows across the different arms shows a continuum of strategies that the controller picks for single cells. 

#thesisfig(
  "freewindow-heatmap-simple",
  [Every pre-demand window in v21 and v23, one row each, running from twenty minutes before
   the demand opens to the moment it does. Colour is the exposure commanded in that minute,
   dark for none and bright for the top of the ladder. Rows are grouped into the four arms,
   which differ only in how many of those last minutes went unscored, and are sorted within
   each arm by shape, so windows that spend their light early sit at one end of a band and
   those that spend it late at the other. The dashed line marks the seven minutes a cell
   needs to climb from the anchor to the demand: light to the right of it arrives in time to
   count, light to the left has decayed before it is scored. The strip on the left is the
   demand each window was preparing for. Since rows are sorted by shape, a controller that
   prepared differently for different objectives would show those colours lining up with the
   ordering inside each arm; they stay mixed throughout.],
  "freewindow-heatmap-by-demand",
)

Each window is reduced to its shape before anything is compared. Its twenty commanded
exposures are divided by their own total, which removes how much light was spent and leaves
only when it was spent; principal components are then taken over all such normalised
windows. The shape score is a window's projection onto the first component, which orders
windows by how early within the window the light fell.

The ordering runs between two extremes rather than between two kinds. At one end there is
little if any stimulation until roughly seven minutes before the demand, followed by
strong stimulation; these rows sit at the top of each arm in
@freewindow-heatmap-by-demand. At the other there is medium stimulation up to that same
seven-minute mark and then nothing at all. Every intermediate between them is occupied,
and no edge separates one from the other.

The extremes are also rare. Along this axis the density has a single peak near the middle
in every arm, with several times as many windows near the centre as in either tail. What
the unscored window changes is the width of that distribution rather than its shape:
pooled across both runs the interquartile spread of the shape score doubles, from 0.079 in
the arm scored throughout to 0.159 in the arm given twenty free minutes, while remaining
single-peaked in all four.

The two runs place their middle arms at different lengths, and that is what locates the
threshold. In v21, whose arms are 0, 4, 10 and 20 free minutes, the spread is flat across
the first two, 0.096 against 0.094, and rises only at ten. In v23, whose arms are 0, 8, 14
and 20, it is already rising at eight, 0.046 against 0.066, and climbs monotonically from
there. Four free minutes buys nothing and eight buys something, which puts the threshold
between them and close to the seven minutes a cell needs to climb from the anchor to the
demand. More unscored time buys more variety, not more kinds.

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

Testing this would mean comparing what the controller did before one demand against what
it did before another at the level of the block, since the demand pattern belongs to the
block rather than to the cell. With only three blocks per pattern that comparison has very
little power, and this work does not settle the question.

#thesisfig(
  "freewindow-examples",
  [Two windows from opposite ends of the shape space. Chosen by a stated rule: the window nearest the 5th and nearest the 95th percentile of the first shape component, both at the median of the second. Dashed line: the demand. Bars: light commanded. Green: the unscored window. Both cells are shown over one whole block.],
  "freewindow-examples",
)


// ═════════════════════════════════════════════════════════════════════════════
//  DISCUSSION
// ═════════════════════════════════════════════════════════════════════════════

= Discussion & Future work

This work builds a closed loop that plans light for single cells against a learned model of
their ERK response, and runs it on live cells for twelve hours at a time. Its three parts
stand on very different footing, and the rest of this chapter takes them in turn.

The model is the firmest. It forecasts well ahead of persistence, it needs the cell's own
past rather than its present level to do so, and its stated spread is calibrated offline and
stays calibrated at the forecast horizon once it is running on the microscope. The control
claims are narrower, and the reason is structural: the comparison that would separate
feedback from dose was never run, because the closed-loop arms beat the constant arm while
spending 1.87 times its light. What the runs establish about individuation rests on four
fields and points one way without settling it.

The constraints the runs met are worth as much as either. Sensitivity declines over a run,
the resting level moves on its own in the dark, and the resting spread of the population is
wider than light can move any one cell, so a demand shared across cells is unreachable for a
large fraction of them by construction. These bound what a controller of this kind can be
asked to do, whatever model sits inside it.


== What the live setting changes about evaluation

Offline the model is scored against experiments done before it existed;
in a live run it is scored against the consequences of its own earlier decisions.
No held-out set stands in for that, which is why the evaluation had to be made again on the live experiment.

That distinction is also why the controller was never given the model's uncertainty to plan
with. The machinery is in place on both sides: the head is a mixture throughout, and the
kernel that reads it is one line of the policy. What was missing was the evidence needed to
rely on it inside the loop. The band kernel was tested in v10 and v11 and did not beat plain
squared error there, although two early runs at a slipped cadence are weak ground for a
general conclusion. More importantly, a probabilistic cost only earns its place if the
stated spread is right in the regime where it is used, and for a closed loop that means
calibrated against the consequences of the controller's own earlier choices rather than
against a held-out set. Establishing that would have taken calibration work and live
experiments of its own, and each live experiment is twelve hours on one microscope. With the
throughput available, testing every controller variant to that standard was not possible,
and the runs were spent on the questions that could be answered.

It is therefore left to future work with a starting point rather than as an open question.
Offline calibration does carry into live operation at the forecast horizon, degrading in the
mid-horizon range (@rig-calibration), so the transfer question now has a partial answer it
did not have when the choice was made. A serious attempt would also have to change the
rollout: the decoder feeds its own mean forward, so a cost that prices multimodality more
than one step ahead needs the full mixture carried through the horizon rather than collapsed
at every step.

== The objective, and what a population can be asked for

During our experiments, choice of an objective was an arbitrary, experiment-wide static decision. 
This helps with evaluation of multiple cells against a single target,
but is not the best possible fit to the problem.
Within a population, there is a bigger dispersion of resting CNR values,
than an average cell is capable of moving (@reachability-runs). Because of this, a static, population-wide pattern will
always have a proportion of the population over the demanded state, and some that is incapable of reaching it.
Geometry of the objective matters as well, especially in a system that can only be perturbed one way - into
activation, and so the objective function should take into account realistic deactivation.

This is something addressable in further research. 
One possible alternative would be to encode objective itself on a per-cell basis - for example based on the 
initial resting CNR state, position within a cluster or number of neighbours.
For probing population-level phenomena, this approach could be 
used to pre-select the cells that exhibit signs of 'good controllability', 
and then stimulate only those.

== Which cells the numbers describe

Every result from the live experiments is computed on cells that held at least 120 scored
frames, roughly two hours, since a shorter track carries too little of the run to be
compared with itself. That selection is not neutral. Cells leave the tracker for reasons that need not be
independent of what is being measured: they divide, they migrate out of the field, they
round up, or segmentation loses them when their morphology changes. If any of those
correlate with how a cell responds to light, the surviving population is not a random
sample of the population that was stimulated, and every per-cell median reported here
describes survivors.

The direction of that bias is not obvious and this work does not measure it. It could run
either way: a cell that responds strongly and rounds up is lost, biasing the sample toward
weak responders, while a cell unhealthy enough to be lost early may also respond weakly,
biasing it the other way. This is answerable with data already collected. Every run records
the frame at which each track ends, so dropout can be aligned to what the cell was doing
immediately before it was lost and pooled across runs, giving a hazard of leaving the
tracker as a function of response magnitude and of light received. If that hazard is flat,
the survivor medians stand as reported; if it is not, its slope gives both the size of the
correction and its direction. Future work should run that analysis before any of these
per-cell numbers is quoted outside the setting that produced it.


== How the controller stimulates

We found that a controller's choice for stimulation is sampled from a continuum of behaviors.
On one end, the strategy of stimulating only just before the demand is scored, and on the other end 
to stimulate early, then stop and let the CNR fall into the desired state. 

Increasing the duration of unscored window before the main objective changes the distribution of behaviors picked - 
widening the cases at the edges of the continuum (mentioned above) at the expense of the 
'constant stimulation' case that lives in between them. 
The constant stimulation strategy is more prevalent in
cases with no free window, perhaps because it does not allow the cells to fall into their own baseline, but 
instead commands them to hold an 'estimated baseline', leading some cells to need to be stimulated.
@history-swap shows how taking cells with matching CNR but swapping their histories has a severe negative influence
on their predictions, highlighting the importance of the historical embedding. Picking different behaviors for 
stimulation is how this effect transfers to the control task, making the stimulation type a marker of the 
individuality of the cell. 
The diversity we observe is ordered along a single axis, but this describes what the controller was
asked to express rather than how many parameters underlie its choice. A second axis would go unseen
here if no objective we posed called on it, or if the stimulation ladder were too coarse to resolve it.
The avenue of finding model's representation of the cell state would benefit from further analysis. 
Another angle of approach could be an analysis of model's embeddings, and whether they can be used to classify
the response actually picked by the controller reliably. 

== Per-cell against population dosing

Dosing cells individually was tested against dosing them together, paired inside each field so
that whatever the field shares (medium, focus, crowding, drift) falls out of the comparison.
Per-cell planning tracked the demand more closely, and spent less light doing so.
Winning on less light is what gives the comparison its force. The broader ordering across arms,
in which tracking improves monotonically with how much feedback an arm was given,
cannot be read the same way: those arms also received more light, and the run has no means of saying
which of the two produced the ordering. This one does.
The advantage was not unanimous across fields, and since the pairing is at the level of the field rather than the cell,
it is more fields and not more cells that would settle the matter, and on four the advantage
points one way without settling it.
It is nonetheless the point at which the individuality the model reads out of a cell's past is allowed
to change what that cell receives, and the cells finish closer to what was asked of them.

A single block of that run admits a sharper reading, being the only one whose demand the cells could 
actually reach.
Across that hour the population median of the constant arm sat closer to the demand than the median under closed-loop control,
while its individual cells sat further away. This is the situation the introduction gives as the reason
for working at single-cell resolution, and it is the one place here where it was observed rather than assumed:
a summary statistic can be driven onto a target by a controller that is moving cells away from it,
and the summary cannot report the difference.

== Declining responsiveness

Over the course of the experiment we noticed a gradual shift in quality of the tracking (@sensitivity-decline),
as well as the amount of light energy used. 
Its cause is unknown, with probable candidates being transcriptional feedback, receptor internalization 
and pathway desensitization. 
Two features of the decline sit awkwardly together: it follows elapsed time rather than the number of
stimulation cycles delivered, yet its size follows delivered power rather than the shape of the pattern.
Both are consistent with a single process accumulating with total exposure, since at fixed cadence dose
and elapsed time run together and our protocols cannot separate them. They are equally consistent with
two processes superimposed - one stimulation-independent and tied to the duration of the run, one
dose-dependent - and we cannot distinguish these possibilities with the current runs. The arm that
received no stimulation light bears on this directly, and it declines with the light off. Part of the decline is therefore stimulation-independent, and what the current
runs cannot say is how much of the rest is not.

The controller learns to deal with the decrease in sensitivity by increasing light budget in a
reactive manner, while adapting the running encoder by integrating the newly inflated
stimulations.

== Why the model does not follow the decline

The objection we raised against mechanistic fits, that they cannot follow a cell whose response
changes over a run without being refitted, applies to the model used here as well, though for a
different reason. It is tempting to blame the feature set, since nothing in the inputs
represents elapsed time. However, a recurrent encoder
stepped from a zero state has the number of steps it has taken implicitly available in its own
state, and could carry a clock without being handed one.

The binding constraint is the training distribution. The median track in the training corpus is
90 frames, an hour and a half, and the longest anywhere in it is 210 frames, three and a half
hours; a live run is 720. At the rate the dark arm declines, drift accumulates to roughly
0.03 CNR over a median training track, which is inside the noise. The corpus does not contain
the phenomenon in a form that could be learned from, at any length. This is distribution shift
rather than feature omission. It may also bear on why the model's uncertainty is least reliable
in the mid-horizon on the rig while being well calibrated offline, since live forecasts are made
from encoder states far longer than any seen in training, though the runs here cannot test that
directly.

Both repairs follow from that diagnosis rather than from the feature set. The first is to train
on long experiments that experience drift natively, which we did not have access to at the
outset. The second is to let the model keep learning during the run: online updating from
incoming measurements @Bieker2020 both lowers the data requirement and extracts more from each
experiment, and it addresses distribution shift directly, since a model that adapts as the run
proceeds does not need the training corpus to have contained twelve-hour cells.

An example way to integrate such pathway-level phenomena in future work would be to wire it at the level of the
model architecture, for example as an explicit time axis between observations. Such operation would not only
increase awareness of drift, but also could serve as a way to encode external events relevant to the 
internal state of the cell (time of starvation, etc), and could aid in disentangling time-based phenomena from
stimulation-driven ones.

The same omission appears one level up, in the controller. Its cost prices the light it spends but not
what spending it does to the cell's willingness to respond later, so sensitivity is absent from the state
the planner reasons over just as time is absent from the model's inputs. Should recovery prove to exist
on a measurable timescale, this becomes actionable rather than merely diagnosable: responsiveness could
be carried as a depleting and regenerating resource, and the loop could hold cells dark for a period in
order to restore the authority it needs for a later demand. The behaviour in the unscored window shows
the planner already trading present error against future reachability when it is given the room; drift
would give it the same trade over hours instead of minutes.
Future work should also consider measuring not only the drift but also recovery, and how it affects 
controllability metrics. 
v24's dark arm already carries half of that: with no light for twelve hours its cells still fell,
so the resting level moves on its own. What such an arm cannot report is whether
responsiveness moved with it. A dark arm carried through a future run should therefore end with a probe
pulse, and that single frame is what separates a decline driven by the run's duration from one driven by
the light we spend. 

// ═════════════════════════════════════════════════════════════════════════════
//  FUTURE WORK; (Now folded into the discussion.) 
// ═════════════════════════════════════════════════════════════════════════════

// - Make time an explicit axis in the model, either in the conditioning or as a
//   feature (`time_since_last_measurement`). This makes it possible to learn
//   multiple timescales, and may do better on long-range effects such as receptor
//   internalisation or transcriptional feedback.
// - Extend into the spatial dimension, possibly with a hierarchical model.
// - Use the model to help fit a hybrid approach — a neural ODE, a
//   physics-informed network, or similar.
// - Use it to quantify the controllability of systems of this kind, and to ask
//   whether the control drift itself can be influenced.
// 
// ```
// In order to lower the data requirements and to improve the prediction accuracy and thus the
// control performance, incoming sensor data are used to update the RNN online. 
// ```

// ═════════════════════════════════════════════════════════════════════════════
//  CONCLUSION
// ═════════════════════════════════════════════════════════════════════════════

= Conclusion

This thesis closed a control loop around individual mammalian cells. A recurrent model was
trained on 6.6 million frames of single-cell optogenetic experiments to forecast one cell's
ERK activity under a proposed sequence of light, and a model-predictive controller planned
against that forecast for every cell in the field, at every frame, across twelve-hour live
experiments.

The regime is what distinguishes it. A loop of this shape has been closed before on gene
expression @Lugagne2024, where the readout integrates over hours and the controller has room
to measure, plan and wait. A signalling cascade does not grant that room: the loop must
decide between one frame and the next, and its planning horizon is half an hour rather than
a day. The actuator is one-sided as well. Light drives ERK activity up and nothing drives it
down, so the controller commands rises and can only wait out falls, and the rate at which a
cell returns is set by the cell rather than by the loop. Both differences are what make the
control predictive rather than reactive, and both showed up in the runs rather than only in
the design.

Of the three parts of the loop, the model is the one that holds. It forecasts well ahead of
persistence, it requires the cell's own past rather than its present level, and the spread
it reports is calibrated offline and remains calibrated at the forecast horizon on the
microscope. The control results are weaker. Closed-loop planning tracked a
reachable demand better than constant light and better than darkness, but it also spent more
light than the arm it beat, so that comparison does not separate feedback from dose.
Planning per cell rather than broadcasting one dose was both closer and cheaper, which is
the comparison that does separate them, but it rests on four fields and points one way
without settling it.

What the runs establish most firmly is a constraint rather than a capability. The resting
spread of a population is wider than light can move any single cell, so a demand shared
across cells is out of reach for a large fraction of them before the controller does
anything, and responsiveness declines over the course of a run whether or not light is
delivered. Neither is a property of the model or of the controller. Both say that the
binding limit on control of this kind is actuation authority and reachability, not
prediction. The single-cell measurement that makes per-cell planning possible also makes
per-cell demands possible, and asking each cell for something it can reach is where the next
gain lies.


// ═════════════════════════════════════════════════════════════════════════════
//  APPENDIX
// ═════════════════════════════════════════════════════════════════════════════

#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1")

= Appendix

== Training loss curves

#thesisfig(
  "loss-curves",
  [Training and validation negative log likelihood for the reported checkpoint, over 300
   epochs. (a) The whole run. The shaded span is the teacher-forcing anneal: the decoder is
   fed the observed CNR at a linearly decreasing fraction of steps, reaching zero at
   epoch 90. Validation loss falls steeply throughout that span and the two curves cross
   as it ends, because until then the decoder is being scored on an easier task than the one it
   will face live. (b) The same curves after the anneal, which is the only part of training
   that measures free-running rollout, the regime the controller actually uses. The dashed
   line marks the epoch with the lowest validation loss, which is the checkpoint kept.],
  "fig-loss-curves",
  float: false,
)

== Light dose calculation <light-dose-calculation>

Two microscopes contributed to the training corpus, and they do not deliver the same light
at the same nominal setting, so commanded exposures had to be converted into a physical
dose before the two could be pooled. Each setup carries a bench calibration: a table of LED
power settings, from 0 to 100%, against the optical power at the sample and the
corresponding irradiance. A commanded power is converted by piecewise-linear interpolation
into that table, and the irradiance is multiplied by the exposure duration to give the
radiant exposure delivered by a single pulse,

$ "fluence" ["mJ/cm"^2] = "irradiance" ["mW/cm"^2] times "exposure" ["ms"] times 10^(-3) $

which is the quantity the model consumes and commands, and is referred to throughout as
fluence. Both curves convert optical power to irradiance through the same illuminated
area, 0.006362 cm#super[2], taken as a 900 µm diameter field, and it is that shared
constant that puts the two setups on a single dose axis.

Two power figures are quoted for each setting and they are not interchangeable. The values
given in the microscopy section are the output of the light engine before the neutral
density filter in the illumination path; the calibration table from which fluence is
computed is the attenuated measurement, which is what reached the cells. On the training
setup at 10% these differ by roughly sevenfold, about 340 µW against the 49.9 µW the table
records. Every fluence value in the corpus is computed from the attenuated column. The
curves carry no time dimension: they are assumed stable per instrument, and no correction
for lamp ageing is applied.

== Additional figures

#thesisfig(
  "reachability-runs",
  [Every cell in every experiment, sorted by initial resting CNR. Colored lines represent p95 of its CNR. Red vertical line shows median of demanded CNR in a given experiment, while the vertical shaded bar stands for demand IQR],
  "reachability-runs",  float: false,
)


#thesisfig(
  "run-ledger",
  [Every live run scored on the same two gates. (a) achieved
   cadence, median to p90, against the 1 min interval the model was trained on;
   seven runs slipped, from 1.16 to 5.72 min per frame. (b) share of closed-loop
   cell-frames sitting on the largest setting of their own field's ladder, annotated
   with the ladders issued. Saturation must be measured per field: v14--v16 gave
   their closed-loop fields a 150 ms ladder while driving their open-loop fields
   to 600 ms, so a run-wide figure understates v16 by more than a factor of ten
   (5% against 73%). Three runs of nineteen clear both gates (v21, v23, v24);
   v16 and v19 hold cadence but saturate 73% and 32% of the time; v22 is
   excluded for a mis-set objective.],
  "fig-ledger",  float: false,
)

#thesisfig(
  "arm-tracks",
  [A raw plot of all the admissible experiments. Solid lines are median CNR of a given arm of experiment. Shaded parts represent IQR ],
  "experiment tracks",  float: false,
)


// ═════════════════════════════════════════════════════════════════════════════
//  BIBLIOGRAPHY
// ═════════════════════════════════════════════════════════════════════════════

#set heading(numbering: none)
#bibliography("refs.bib", style: "nature", title: "References")
