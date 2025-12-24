import numpy as np
import matplotlib.pyplot as plt

D = np.load("stability_ensemble_results.npz", allow_pickle=True)
ks = D["ks"]
m_mean = D["ens_micro_mean"]; m_std = D["ens_micro_std"]
M_mean = D["ens_macro_mean"]; M_std = D["ens_macro_std"]

plt.figure()
plt.errorbar(ks, m_mean, yerr=m_std, fmt='o-')
plt.xlabel("Ensemble size")
plt.ylabel("Test micro RMSE")
plt.title("Ensemble-size curve (nowcast micro RMSE)")
plt.grid(True)
plt.tight_layout()
plt.savefig("ensemble_curve_micro.pdf")
plt.savefig("ensemble_curve_micro.png")
plt.close()

plt.figure()
plt.errorbar(ks, M_mean, yerr=M_std, fmt='o-')
plt.xlabel("Ensemble size")
plt.ylabel("Test macro RMSE")
plt.title("Ensemble-size curve (nowcast macro RMSE)")
plt.grid(True)
plt.tight_layout()
plt.savefig("ensemble_curve_macro.pdf")
plt.savefig("ensemble_curve_macro.png")
plt.close()

print("[OK] wrote ensemble_curve_micro/macro (pdf+png)")
