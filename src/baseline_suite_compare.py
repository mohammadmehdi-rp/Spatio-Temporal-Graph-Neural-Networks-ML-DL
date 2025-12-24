#!/usr/bin/env python3
"""
baseline_suite_compare.py — compare Naive, MA(5), AR(best), ARIMA(p,1,0), KF lead-1 on the same split.
Assumes processed.csv exists and ar_gridsearch.py is available (for best AR).
"""
import argparse, subprocess, sys, shlex

def run(cmd):
    print("$", cmd)
    r=subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout.strip())
    if r.returncode!=0: print(r.stderr, file=sys.stderr)
    return r.returncode

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--label", default="backlog_pkts")
    ap.add_argument("--ar_lags", default="1,2,3,5,10,15")
    ap.add_argument("--ar_l2", default="1,10,50,200,1000")
    ap.add_argument("--arima_p", default="1,2,3,5,10")
    ap.add_argument("--arima_l2", default="1,10,50,200,1000")
    ap.add_argument("--kf_Q", default="50")
    ap.add_argument("--kf_R", default="200")
    args=ap.parse_args()

    # Naive/MA and Ridge ORCL/MASK come from baseline_eval_v2.py you already have
    run(f"python3 baseline_eval_v2.py --processed processed.csv --masked processed_masked.csv --lags 10 --l2 50")
    # Best AR
    run(f"python3 ar_gridsearch.py --processed processed.csv --label {args.label} --lags {args.ar_lags} --l2 {args.ar_l2}")
    # ARIMA(p,1,0)
    run(f"python3 arima_p10_eval.py --processed processed.csv --label {args.label} --p_list {args.arima_p} --l2_list {args.arima_l2}")
    # KF lead-1
    run(f"python3 kalman_lead1_eval.py --processed processed.csv --label {args.label} --Q {args.kf_Q} --R {args.kf_R}")

if __name__=="__main__":
    main()
