"""CLI.

  python -m abtest analyze 10000 520 10000 580
  python -m abtest size --baseline 0.05 --mde 0.10
  python -m abtest peeking
"""
from __future__ import annotations

import argparse

from .core import Variant, bayesian, peeking_false_positive_rate, sample_size, z_test


def main() -> None:
    ap = argparse.ArgumentParser(prog="abtest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyze", help="analyze a finished test")
    an.add_argument("a_visitors", type=int)
    an.add_argument("a_conversions", type=int)
    an.add_argument("b_visitors", type=int)
    an.add_argument("b_conversions", type=int)
    an.add_argument("--alpha", type=float, default=0.05)
    sz = sub.add_parser("size", help="visitors needed per variant")
    sz.add_argument("--baseline", type=float, required=True)
    sz.add_argument("--mde", type=float, required=True, help="relative lift, e.g. 0.10 for +10%%")
    sz.add_argument("--power", type=float, default=0.8)
    sub.add_parser("peeking", help="show how daily peeking inflates false positives")
    args = ap.parse_args()

    if args.cmd == "analyze":
        a = Variant(args.a_visitors, args.a_conversions)
        b = Variant(args.b_visitors, args.b_conversions)
        f, bay = z_test(a, b, args.alpha), bayesian(a, b)
        print(f"Conversion  A: {a.rate:.2%}   B: {b.rate:.2%}")
        print(f"Lift        {100 * f.lift_abs:+.2f} percentage points ({f.lift_rel:+.1%} relative)")
        print(f"\nFrequentist  p = {f.p_value:.4f}  ({'significant' if f.significant else 'not significant'} at alpha={args.alpha})")
        print(f"             {100 * (1 - args.alpha):.0f}% CI on difference: [{100 * f.ci_low:+.2f}, {100 * f.ci_high:+.2f}] pp")
        print(f"\nBayesian     P(B > A) = {bay.prob_b_better:.1%}")
        print(f"             95% interval on relative lift: [{bay.lift_rel_ci[0]:+.1%}, {bay.lift_rel_ci[1]:+.1%}]")
        print(f"             expected loss if we ship B: {100 * bay.expected_loss_choosing_b:.3f} pp of conversion")
    elif args.cmd == "size":
        n = sample_size(args.baseline, args.mde, power=args.power)
        print(f"{n:,} visitors per variant ({2 * n:,} total) to detect a {args.mde:+.0%} lift "
              f"on a {args.baseline:.1%} baseline with {args.power:.0%} power.")
    else:
        r = peeking_false_positive_rate()
        print("A/A tests (no real difference), 28 days, 500 visitors/day per arm:")
        print(f"  check once at the end:          {r['check_once_at_end']:.1%} false positives")
        print(f"  peek daily, stop at first p<.05: {r['peek_daily_stop_early']:.1%} false positives")


if __name__ == "__main__":
    main()
