# Real-Time Soft Sensor for Penicillin Fermentation

A Python pipeline that estimates penicillin titer from a fermenter's cheap online
sensors, and then asks a harder question: how early in a batch can you actually predict
how it will end? Built on IndPenSim V3, a simulation of 100 industrial-scale (100,000 L)
penicillin fed-batch fermentations.

`Soft Sensor` · `PLS / Multivariate Analysis` · `Process Analytical Technology (PAT)`
· `Bioprocess Engineering` · `Python` · `scikit-learn`

This is the third project in a set. The first (DoE) picked the recipe; the second
(SPC / root-cause) found that batches only start diverging around hour 100. This one
tries to predict titer in real time — and, coming at it from a totally different angle,
ends up pointing back at that same hour-100 mark.

> IndPenSim V3 is a simulator (Goldrick et al.), so I have the true titer at every
> timepoint to check against. That's the whole reason I can report honest held-out
> scores instead of hand-waving.

## Why bother

On a real fermenter the sensors you have cheaply and continuously — pH, dissolved oxygen,
feed rates, off-gas CO₂, oxygen uptake — are not the number you care about. The number
you care about is penicillin titer, and that comes from a slow lab assay you only run a
handful of times per batch. A soft sensor is the bridge: learn the relationship between
the cheap live signals and the expensive lab value, then read titer off the cheap signals
in real time.

Two questions, one per script:

1. Can we estimate titer live? (`step1`)
2. Can we predict a batch's *final* titer early enough to do something about it? (`step2`)

## Results

The soft sensor works: **R² = 0.92** predicting titer on whole batches the model never
saw during training.

Getting there took one real fix. My first version scored 0.77 and was frankly bad at the
two ends of a batch — it predicted *negative* titer in the first few hours and gave up on
the plateau near harvest, drooping to ~19 g/L when the real value was ~29. Both problems
had the same cause. Titer is a quantity that *accumulates* over the batch, but the sensors
mostly report the culture's *current* activity. Early and late, the instantaneous sensors
look similar (quiet), while the accumulated titer is totally different (empty vs. full).
The model had no way to tell those apart because it only ever saw one instant at a time.

The fix was to give it a memory: cumulative sugar fed and cumulative oxygen uptake —
running totals, the "odometer" the raw sensors don't provide. That single change took it
from 0.77 to 0.92 and made the plateau problem disappear. It's the most satisfying part of
the project, honestly, because the fix fell straight out of thinking about what the sensors
physically mean rather than tweaking the model.

Worth stating plainly: penicillin is never an input. Only sensors a real plant would have
live. Otherwise you're just handing the model the answer.

### The early-warning part is where it gets interesting

`step2` reframes the question: using only the first N hours of a batch, predict its final
titer. Then sweep N from 20 to 160 hours and watch how the accuracy climbs.

First pass, the accuracy was basically flat and near zero until about hour 100. Which
looked like a dead model — but before believing that, I checked whether my own features
were the problem. `step2` squashes each early window into averages, and an average throws
away shape: two batches with the same mean feed rate can be climbing vs. flat and headed
for completely different endings. So I added slope and latest-value features to capture
that shape.

They helped, and that's the actual finding: early R² went from under 0.1 to about 0.5 by
hour 100. So there *is* a real early signal, and averaging had been hiding some of it. But
it never crosses 0.8, and the curve flattens out after hour 100. Meaning: you can get a
rough early read on a batch, but its fate genuinely isn't settled until late. You can't
reliably call the ending from the opening hours no matter how you slice the features.

The thing I like about this is where the curve bends. It climbs hard up to ~hour 100, then
plateaus. That's the exact hour my SPC project (different data slice, different method,
control charts instead of regression) flagged as the point where good and bad batches start
to separate. Two unrelated methods landing on the same number is a much better feeling than
either one alone.

![soft sensor parity and live tracking](assets/soft_sensor_results.png)
![early-warning cutoff sweep](assets/early_warning_results.png)

## Pipeline

One shared loader that every step imports. It keeps the full time axis, which is the
opposite of my SPC loader (that one collapsed each batch to a single value). A soft sensor
needs the moment-to-moment trajectory, so the spine had to be rebuilt for this project.

```
100_Batches_IndPenSim_V3.csv
        |
        v
step0_data_loader.py     spine: time-resolved load, Parquet cache, resolves the batch axis
        |
        +-- step1_soft_sensor.py     live titer (PLS + cumulative features), R2 = 0.92
        +-- step2_early_warning.py   predict final titer from first N hours, sweep N
```

| Step | File | What it does |
|---|---|---|
| 0 | `step0_data_loader.py` | Loads only the columns needed, caches to Parquet, exposes the sensor list so nothing downstream re-guesses it. |
| 1 | `step1_soft_sensor.py` | The soft sensor. Whole-batch split, grouped CV to choose PLS components, cumulative memory features, parity + time-trace plots. |
| 2 | `step2_early_warning.py` | Predicts final titer from the first N hours; sweeps N to find the earliest reliable call; adds trajectory features. |

### The batch-ID trap

Leaving this in because it cost me time and it's the same trap the SPC project hit. The
column actually named `Batch ID` is useless — ~47,000 unique values, it's a row counter.
The real per-batch axis is hiding in a column called `1-Raman spec recorded`, which has
exactly 100 unique values. The loader resolves that and hard-fails if the batch count
isn't in a sane range (`2 ≤ n ≤ 500`), so a wrong column blows up immediately instead of
quietly producing garbage groupings three steps later.

## Validation, and why it's set up this way

Every split is by whole batch, never by row (`GroupShuffleSplit` in step1,
`cross_val_predict` in step2). This matters more than it sounds. Consecutive rows within a
batch are nearly identical, so if you shuffle rows and split randomly, the same batch ends
up in both train and test and the model just recognizes it. You get a beautiful,
meaningless score. Splitting whole batches out is the only way the number means "works on a
new batch."

PLS component count is chosen by grouped cross-validation rather than picked by hand, so
there's a data-driven answer to "why that many."

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib pyarrow

python step1_soft_sensor.py      # live titer soft sensor
python step2_early_warning.py    # how early can we predict the final titer?
```

Both import `step0` automatically. First run parses the CSV and writes a Parquet cache;
everything after is fast. Drop `100_Batches_IndPenSim_V3.csv` in the repo root first
(it's git-ignored — download separately from the link below).

## Data

IndPenSim (Industrial-scale Penicillin Simulation), Goldrick et al. — a first-principles
model of a 100,000 L *Penicillium chrysogenum* fermentation, validated against historical
industrial data.

- Download: http://www.industrialpenicillinsimulation.com/
- Mirror / DOI: https://data.mendeley.com/datasets/pdnjz7zz5x/2

## Limitations

PLS is linear. A nonlinear model might squeeze out a bit more, but I don't think it'd
change the early-warning story — the plateau looks like a real property of the process,
not something a fancier model fixes. I'd rather report the ceiling than torture the data
past it.

The cumulative features are running sums, not true time-weighted integrals. IndPenSim's
timestep is constant so it doesn't matter here (the difference is a constant that scaling
removes), but on irregular timesteps you'd want `(rate * dt).cumsum()`.

It's simulated data. No sensor drift, no fouling, no strain variation. A real deployment
would need periodic recalibration against actual lab samples.

And the early-warning result stays honestly moderate (R² ~0.5–0.6). I'm reporting it as-is
rather than dressing it up, because the "you can't fully call it early" half is part of the
point.
