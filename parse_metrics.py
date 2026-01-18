import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os

# --- SETTINGS ---
# Folder containing your files (Current directory '.')
FILE_FOLDER = "/home/aglamazlarefe/ns-allinone-3.35/ns-3.35/contrib/opengym/examples/TCP-RL" 

FILES = {
    "Cubic": os.path.join(FILE_FOLDER, "results_cubic.txt"),
    "NewReno": os.path.join(FILE_FOLDER, "results_newreno.txt"),
    "DRL-TCP": os.path.join(FILE_FOLDER, "results_rl.txt")
}

def parse_metrics(filename):
    """Reads metrics from the file (Throughput, RTT, and Packet Loss)."""
    data = {"Time": [], "Throughput": [], "RTT": [], "Loss": []}
    
    if not os.path.exists(filename):
        print(f"WARNING: {filename} not found!")
        return pd.DataFrame(data)
        
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if not lines: return pd.DataFrame(data)

            # Skip header
            start_idx = 0
            if "Time" in lines[0] or not lines[0][0].isdigit():
                start_idx = 1
            
            for line in lines[start_idx:]:
                parts = line.strip().split(',')
                # Must have at least 3 metrics, 4th one is Packet Loss if present
                if len(parts) >= 3:
                    try:
                        t = float(parts[0])
                        tp = float(parts[1]) / 1e6   # Mbps
                        rtt = float(parts[2]) * 1000 # ms
                        
                        # Get Packet Loss if 4th column exists, else 0
                        loss = int(parts[3]) if len(parts) > 3 else 0
                        
                        data["Time"].append(t)
                        data["Throughput"].append(tp)
                        data["RTT"].append(rtt)
                        data["Loss"].append(loss)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading file: {e}")

    return pd.DataFrame(data)

def run_analysis_and_plot():
    # 1. Load Data
    dfs = {name: parse_metrics(path) for name, path in FILES.items()}
    
    # Do not proceed if DRL data is missing
    if dfs["DRL-TCP"].empty:
        print("CRITICAL ERROR: 'results_rl.txt' data is missing.")
        return

    # --- 1. COMPARISON: CUBIC vs DRL ---
    if not dfs["Cubic"].empty:
        print("\n" + "="*20 + " CUBIC vs DRL " + "="*20)
        
        cubic_tp = dfs["Cubic"]["Throughput"]
        drl_tp = dfs["DRL-TCP"]["Throughput"]
        cubic_rtt = dfs["Cubic"]["RTT"]
        drl_rtt = dfs["DRL-TCP"]["RTT"]

        # Statistics
        diff_tp = ((drl_tp.mean() - cubic_tp.mean()) / cubic_tp.mean()) * 100
        _, p_val_tp = stats.ttest_ind(cubic_tp, drl_tp, equal_var=False)
        
        diff_rtt = ((drl_rtt.mean() - cubic_rtt.mean()) / cubic_rtt.mean()) * 100
        _, p_val_rtt = stats.ttest_ind(cubic_rtt, drl_rtt, equal_var=False)

        print(f"Throughput Difference    : %{diff_tp:.2f} (P-Value: {p_val_tp:.5f})")
        print(f"RTT (Latency) Difference : %{diff_rtt:.2f} (P-Value: {p_val_rtt:.5f})")
        
        if diff_rtt < 0: print(">> DRL achieved lower latency than Cubic! 🚀")
    else:
        print("WARNING: Cubic data not found, comparison cannot be performed.")

    # --- 2. COMPARISON: NEW RENO vs DRL ---
    if not dfs["NewReno"].empty:
        print("\n" + "="*18 + " NEW RENO vs DRL " + "="*19)
        
        newreno_tp = dfs["NewReno"]["Throughput"]
        drl_tp = dfs["DRL-TCP"]["Throughput"] # DRL is the same
        newreno_rtt = dfs["NewReno"]["RTT"]
        drl_rtt = dfs["DRL-TCP"]["RTT"]

        # Statistics
        diff_tp_nr = ((drl_tp.mean() - newreno_tp.mean()) / newreno_tp.mean()) * 100
        _, p_val_tp_nr = stats.ttest_ind(newreno_tp, drl_tp, equal_var=False)
        
        diff_rtt_nr = ((drl_rtt.mean() - newreno_rtt.mean()) / newreno_rtt.mean()) * 100
        _, p_val_rtt_nr = stats.ttest_ind(newreno_rtt, drl_rtt, equal_var=False)

        print(f"Throughput Difference    : %{diff_tp_nr:.2f} (P-Value: {p_val_tp_nr:.5f})")
        print(f"RTT (Latency) Difference : %{diff_rtt_nr:.2f} (P-Value: {p_val_rtt_nr:.5f})")
        
        if diff_rtt_nr < 0: print(">> DRL achieved lower latency than New Reno! 🚀")
    else:
        print("WARNING: New Reno data not found.")
    
    print("="*55 + "\n")

    # --- PLOTTING ---
    save_path = FILE_FOLDER

    # 1. Throughput Plot
    plt.figure(figsize=(12, 6))
    for name, df in dfs.items():
        if not df.empty:
            plt.plot(df["Time"], df["Throughput"], label=name, linewidth=2)
    plt.title("Throughput Comparison (Mbps)")
    plt.ylabel("Throughput (Mbps)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_path, "graphs/comparison_throughput.png"))
    print("Plot saved: comparison_throughput.png")

    # 2. RTT Plot
    plt.figure(figsize=(12, 6))
    for name, df in dfs.items():
        if not df.empty:
            plt.plot(df["Time"], df["RTT"], label=name, linewidth=2)
    plt.title("RTT (Latency) Comparison (ms)")
    plt.ylabel("RTT (ms)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_path, "graphs/comparison_rtt.png"))
    print("Plot saved: comparison_rtt.png")

    # 3. Packet Loss Plot (Cumulative)
    plt.figure(figsize=(12, 6))
    for name, df in dfs.items():
        if not df.empty:
            # Plotting cumulative loss instead of instantaneous loss
            cumulative_loss = df["Loss"].cumsum()
            plt.plot(df["Time"], cumulative_loss, label=name, linewidth=2)
    plt.title("Cumulative Packet Loss (Total)")
    plt.ylabel("Total Packets Lost")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_path, "graphs/comparison_loss.png"))
    print("Plot saved: comparison_loss.png")

if __name__ == "__main__":
    run_analysis_and_plot()