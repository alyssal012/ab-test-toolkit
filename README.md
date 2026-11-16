# A/B Test Toolkit

A small, tested Python toolkit for the three questions every A/B test raises:

1. **Before:** how much traffic do we need?
2. **After:** did B actually win, and how sure are we?
3. **During:** why can't we just check the dashboard every day and stop when it looks good?

```
$ python -m abtest analyze 10000 520 10000 580
Conversion  A: 5.20%   B: 5.80%
Lift        +0.60 percentage points (+11.5% relative)

Frequentist  p = 0.0627  (not significant at alpha=0.05)
             95% CI on difference: [-0.03, +1.23] pp

Bayesian     P(B > A) = 96.9%
             95% interval on relative lift: [-0.6%, +25.1%]
             expected loss if we ship B: 0.004 pp of conversion
```

The same data gives a "not significant" p-value and a 97% chance that B is better. Both are correct; they answer different questions. The p-value asks how surprising this data would be if there were no difference. The Bayesian view asks what the business actually wants to know: *how likely is B better, and what does it cost us if we're wrong?* Here the expected loss from shipping B is tiny, which makes a decision possible even without p < 0.05.

## The peeking problem

```
$ python -m abtest peeking
A/A tests (no real difference), 28 days, 500 visitors/day per arm:
  check once at the end:          5.2% false positives
  peek daily, stop at first p<.05: 27.6% false positives
```

This simulates 2,000 tests where A and B are **identical**. Checking once gives the promised ~5% false-positive rate. Checking every day and stopping the first time p < 0.05 raises it to **28%**, more than five times as many "winners" that aren't real. This is one of the most common mistakes in product analytics. The fix is to commit to a sample size up front (`abtest size`) or use a sequential method built for repeated looks.

## Sample size

```
$ python -m abtest size --baseline 0.05 --mde 0.10
31,234 visitors per variant (62,468 total) to detect a +10% lift on a 5.0% baseline with 80% power.
```

Halving the effect you want to detect roughly quadruples the traffic you need. That's the main reason small tests on small sites are usually inconclusive.

## What's inside

`abtest/core.py`:

- `z_test`: two-proportion z-test with a confidence interval on the difference
- `bayesian`: Beta-Binomial posterior, giving P(B > A), expected loss for each choice and a credible interval on the lift
- `sample_size`: per-variant traffic for a given baseline, minimum detectable effect, alpha and power
- `peeking_false_positive_rate`: the A/A simulation above

The tests check the z-test against SciPy's chi-square test, the sample size against standard online calculators (about 31k per arm for 5% baseline and +10% lift), and that peeking at least doubles false positives.

## Run it

```bash
pip install -r requirements.txt
python -m abtest analyze 10000 520 10000 580
python -m abtest size --baseline 0.05 --mde 0.10
python -m abtest peeking
pytest -q
```

## Stack

Python, NumPy, SciPy, pytest
