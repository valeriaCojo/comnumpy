import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, LogFormatterMathtext

### S2 ###
df1_1 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S1\\SER_vs_dpTotT_seg20_SNR20_MCMA_S1_1.csv")
df2_1 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S1\\SER_vs_dpTotT_seg20_SNR20_MCMA_to_DD_S1_1.csv")
df3_1 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S1\\SER_vs_dpTotT_seg20_SNR20_MCMA_Soft_S1_1.csv")
df4_1 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S1\\SER_vs_dpTotT_seg20_SNR20_DD_Czegledi_S1_1.csv")

# df1_2 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S2\\SER_vs_dpTotT_seg20_SNR20_MCMA_S2_2_v2.csv")
# df2_2 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S2\\SER_vs_dpTotT_seg20_SNR20_MCMA_to_DD_S2_2_v2.csv")
# df3_2 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S2\\SER_vs_dpTotT_seg20_SNR20_MCMA_Soft_S2_2_v2.csv")
# df4_2 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S2\\SER_vs_dpTotT_seg20_SNR20_DD_Czegledi_S2_2_v2.csv")


# # # Row-wise mean between A and B
# mean_df1 = (0.5*df1_1["SER"] + 0.5*df1_2["SER"])
# mean_df2 = (0.5*df2_1["SER"] + 0.5*df2_2["SER"])
# mean_df3 = (0.5*df3_1["SER"] + 0.5*df3_2["SER"])
# mean_df4 = (0.5*df4_1["SER"] + 0.5*df4_2["SER"])

### S2 ###
# df1 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S2\\SER_vs_dpTotT_seg20_SNR20_MCMA_S2.csv")
# df2 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S2\\SER_vs_dpTotT_seg20_SNR20_MCMA_to_DD_S2.csv")
# df3 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S2\\SER_vs_dpTotT_seg20_SNR20_MCMA_Soft_S2_V2.csv")
# df4 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S2\\SER_vs_dpTotT_seg20_SNR20_DD_Czegledi_S2.csv")


# df1 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_S3_1.csv")
# df2 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_to_DD_S3_1.csv")
# df3 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_MCMA_Soft_S3_1.csv")
# df4 = pd.read_csv("src\\comnumpy\\optical\\pdm\\validation\\results\\S3\\SER_vs_dpTotT_seg20_SNR20_DD_Czegledi_S3_1.csv")

plt.figure(figsize=(7, 5))
plt.loglog(df1_1['dp_tot_T'], df1_1['SER'], marker='o', linewidth=2, label='MCMA')
plt.loglog(df2_1['dp_tot_T'], df2_1['SER'], marker='o', linewidth=2, label='MCMA to Czegledi')
plt.loglog(df3_1['dp_tot_T'], df3_1['SER'], marker='o', linewidth=2, label=f'MCMA Soft')
plt.loglog(df4_1['dp_tot_T'], df4_1['SER'], marker='o', linewidth=2, label=f'DD-Czegledi')
plt.xlabel(r'$\Delta p_{\mathrm{tot}} \cdot T$')
plt.ylabel("Symbol Error Rate (SER)")
plt.title(f"SER vs Δp_tot·T (SNR=18 dB)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

ax = plt.gca()
ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.yaxis.set_minor_formatter(NullFormatter())
plt.legend()
plt.tight_layout()
plt.show()