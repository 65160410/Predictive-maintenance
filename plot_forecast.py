import pandas as pd
import matplotlib.pyplot as plt
import os
import io

def plot_forecast_data(base_dir, output_folders):
    """
    ฟังก์ชันสำหรับอ่านข้อมูล '=== 5. NEXT PERIOD FORECAST ===' และสร้าง Bar Chart
    """
    def parse_forecast_section(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        start_idx = -1
        for i, line in enumerate(lines):
            if '=== 5. NEXT PERIOD FORECAST ===' in line:
                start_idx = i + 1
                break
                
        if start_idx == -1:
            return None
            
        forecast_lines = []
        for line in lines[start_idx:]:
            if line.strip() == '':
                break
            forecast_lines.append(line)
            
        if not forecast_lines:
            return None
            
        df = pd.read_csv(io.StringIO("".join(forecast_lines)))
        return df

    # ปรับสไตล์ของกราฟ
    plt.style.use('seaborn-v0_8-whitegrid')

    for folder in output_folders:
        filepath = os.path.join(base_dir, folder, "vibration_full_report.csv")
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        df = parse_forecast_section(filepath)
        if df is None or df.empty:
            print(f"No Forecast data found in: {folder}")
            continue
            
        # หาคอลัมน์เวลาโดยตัดคอลัมน์ที่ไม่เกี่ยวข้องออกไป (เช่น Jun-24, Sep-24, Oct-24, Next Period)
        exclude_cols = ['Parameter', 'Unit', 'Δ vs Last Period', 'Δ% vs Last Period', 'Trend', 'Risk Indicator']
        time_cols = [col for col in df.columns if col not in exclude_cols]
        
        # สร้าง subplot สำหรับแต่ละ Parameter
        fig, axes = plt.subplots(len(df), 1, figsize=(10, 4 * len(df)))
        fig.suptitle(f'Forecast vs Historical Data - {folder}', fontsize=16, y=1.02, fontweight='bold')
        
        if len(df) == 1:
            axes = [axes]
            
        for i, row in df.iterrows():
            param_name = row['Parameter']
            unit = row['Unit']
            
            values = []
            labels = []
            for col in time_cols:
                val = row[col]
                if pd.notna(val):
                    try:
                        values.append(float(val))
                        labels.append(str(col))
                    except ValueError:
                        pass
            
            ax = axes[i]
            
            # กำหนดสีต่างกันระหว่าง Historical (น้ำเงิน) และ Predicted (ส้ม)
            colors = ['#1f77b4' if 'Predicted' not in l and 'Next Period' not in l else '#ff7f0e' for l in labels]
            
            bars = ax.bar(labels, values, color=colors, edgecolor='black', zorder=3)
            ax.set_title(f"{param_name} ({unit if pd.notna(unit) else ''})", fontsize=14, pad=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
            
            # ใส่ข้อมูลตัวเลขบนบาร์
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        save_path = os.path.join(base_dir, folder, "forecast_barchart.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot successfully at: {save_path}")

if __name__ == "__main__":
    base_directory = r"c:\Users\seksa\OneDrive\Desktop\งานปี 4\Predictive-maintenance"
    # ระบุโฟลเดอร์ output ที่ต้องการ (ผมใส่ outputA_Cooling_Pump ให้ครบจากที่คุณพิมพ์ซ้ำครับ)
    target_folders = ["outputA_CH-06", "outputA_Jockey_pump", "outputA_Cooling_Pump"]
    
    plot_forecast_data(base_directory, target_folders)
