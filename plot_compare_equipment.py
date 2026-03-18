import pandas as pd
import matplotlib.pyplot as plt
import os
import io

def generate_comparison_chart():
    base_dir = r"c:\Users\seksa\OneDrive\Desktop\งานปี 4\Predictive-maintenance"
    output_folders = ["outputA_CH-06", "outputA_Jockey_pump", "outputA_Cooling_Pump"]

    def parse_forecast_section(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        start_idx = -1
        for i, line in enumerate(lines):
            if '=== 5. NEXT PERIOD FORECAST ===' in line:
                start_idx = i + 1
                break
                
        if start_idx == -1: return None
            
        forecast_lines = []
        for line in lines[start_idx:]:
            if line.strip() == '': break
            forecast_lines.append(line)
            
        if not forecast_lines: return None
        return pd.read_csv(io.StringIO("".join(forecast_lines)))

    data = {}
    units = {}

    # รวบรวมข้อมูล Next Period (Predicted)
    for folder in output_folders:
        filepath = os.path.join(base_dir, folder, "vibration_full_report.csv")
        if not os.path.exists(filepath): continue
        
        df = parse_forecast_section(filepath)
        if df is None or df.empty: continue
        
        # หาส่วนของ Next Period (Predicted)
        pred_col = [col for col in df.columns if 'Predicted' in col or 'Next Period' in col]
        if not pred_col: continue
        pred_col = pred_col[0]
        
        for _, row in df.iterrows():
            param = row['Parameter']
            val = row[pred_col]
            units[param] = row['Unit']
            
            if param not in data:
                data[param] = {}
                
            if pd.notna(val):
                try:
                    data[param][folder] = float(val)
                except ValueError:
                    pass

    if not data:
        print("No valid forecast data to compare.")
        return

    # Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    params = list(data.keys())
    fig, axes = plt.subplots(len(params), 1, figsize=(10, 4 * len(params)))
    if len(params) == 1: axes = [axes]

    fig.suptitle('Next Period Forecast - Equipment Comparison', fontsize=16, y=1.02, fontweight='bold')

    for i, param in enumerate(params):
        ax = axes[i]
        folders = list(data[param].keys())
        values = list(data[param].values())
        
        # ตัดชื่อโฟลเดอร์ให้สั้นลงตอนพอร์ต เช่น outputA_CH-06 -> CH-06
        labels = [f.replace("outputA_", "") for f in folders]

        # Assign colors by value rank: red=highest, orange=middle, green=lowest
        sorted_idx = sorted(range(len(values)), key=lambda k: values[k])
        color_map = {}
        rank_colors = ['green', 'orange', 'red']
        for rank, idx in enumerate(sorted_idx):
            color_map[idx] = rank_colors[rank]
        colors = [color_map[j] for j in range(len(values))]
        
        bars = ax.bar(labels, values, color=colors, edgecolor='black', zorder=3)
        unit_text = f" ({units[param]})" if pd.notna(units.get(param)) else ""
        ax.set_title(f"Predicted {param}{unit_text}", fontsize=14, pad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.4f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, h), 
                        xytext=(0, 3), 
                        textcoords="offset points", 
                        ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(base_dir, "equipment_comparison_forecast.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison plot successfully at: {save_path}")

if __name__ == "__main__":
    generate_comparison_chart()
