# ============================================================
#  Vibration Waveform Analysis — ISO 10816-3
#  Google Colab Ready  |  v5 (3-period trend)
#
#  v5 เพิ่มใหม่:
#   - รองรับ 2 หรือ 3 ช่วงเวลา (Jun / Sep / Oct)
#   - Trend line plot แสดง CF, RMS, Peak ทั้ง 3 จุด
#   - CSV section 6: Trend summary ทุก period
# ============================================================

# !pip install scipy matplotlib numpy pandas -q

# ── 1. Imports ───────────────────────────────────────────────
import re, warnings, os, sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# แก้ปัญหาพิมพ์ Emoji ใน Terminal Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew
warnings.filterwarnings('ignore')

# ── 2. Local Files ───────────────────────────────────────────
import os
import tkinter as tk
from tkinter import filedialog

print("=" * 62)
print("  📂  อ่านไฟล์ Waveform .txt จาก Local")
print("  กรุณาเลือกไฟล์จากหน้าต่างป๊อปอัป (รองรับ 2–3 ไฟล์)")
print("  จัดเรียงตามวันที่ใน metadata อัตโนมัติ")
print("=" * 62)

root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)  # ให้หน้าต่างเลือกไฟล์อยู่ออนท็อป
file_paths = filedialog.askopenfilenames(
    title="เลือกไฟล์ Waveform (.txt)",
    filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
)
root.destroy()

file_paths = list(file_paths)
file_names = [os.path.basename(p) for p in file_paths]

if not file_paths:
    raise ValueError("ไม่ได้เลือกไฟล์ กรุณารันโค้ดใหม่แล้วเลือกไฟล์ .txt")

print(f"\nไฟล์ที่ได้รับ ({len(file_names)} ไฟล์):")
for i, fn in enumerate(file_names):
    print(f"  [{i}] {fn}")


# ── 3. Outlier Config ────────────────────────────────────────
#  ▶▶▶  ปรับค่าตรงนี้ได้  ◀◀◀
TRIM_START_MS  = 1.0       # ตัด startup transient (ms)  |  0 = ไม่ตัด
IQR_MULTIPLIER = 8.0       # spike threshold k×IQR       |  ลด = ตัดเพิ่ม
REPLACE_METHOD = 'interpolate'  # 'interpolate' | 'median' | 'zero'
OUTPUT_DIR = 'output'      # โฟลเดอร์สำหรับบันทึกไฟล์ผลลัพธ์
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 4. Parser ────────────────────────────────────────────────
from parser import parse_waveform_txt

# ── 5. Outlier Removal & Metrics ─────────────────────────────
from signal_processing import remove_outliers, compute_metrics, iso_zone, compute_fft, top_freqs, ISO_ZONES

# ── 7. Parse datetime for sorting ────────────────────────────
DATE_FMTS = ['%d-%b-%y %H:%M:%S', '%d-%b-%Y %H:%M:%S',
             '%d-%m-%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S']
def parse_dt(s):
    from datetime import datetime
    for fmt in DATE_FMTS:
        try: return datetime.strptime(s.strip(), fmt)
        except: pass
    return datetime.min

# ── 8. Load, Clean, Compute — all files ──────────────────────
print("\n" + "="*62)
print("  🔍  Loading & Cleaning all files")
print("="*62)

all_ds = []
for fpath, fn in zip(file_paths, file_names):
    with open(fpath, 'rb') as f:
        content_bytes = f.read()
    meta, t_raw, a_raw = parse_waveform_txt(content_bytes)
    m_raw = compute_metrics(a_raw, t_raw)
    dt_str = meta.get('datetime','')
    # derive period label from metadata date
    try:
        d = parse_dt(dt_str)

        # ตรวจสอบว่าเดือนใน metadata ตรงกับชื่อไฟล์ไหม
        month_from_file = fn.split('_')[-1][:3]   # Jun / Sep / Oct
        if d.strftime('%b') != month_from_file:
            raise ValueError("metadata month mismatch")

        label = d.strftime('%b-%y')

    except:
        # fallback จาก filename
        import re
        m = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d{2})', fn)
        if m:
            label = f"{m.group(1)}-{m.group(2)}"
        else:
            label = fn[:10]


    print(f"\n  ── {label}  ({fn}) ──")
    print(f"     RAW   N={len(a_raw)}  RMS={m_raw['rms']:.4f}G  "
          f"Peak={m_raw['peak']:.4f}G  CF={m_raw['crest']:.3f}  Kurt={m_raw['kurt']:.3f}")

    t_c, a_c, rpt = remove_outliers(t_raw, a_raw, label=label)
    m_cln = compute_metrics(a_c, t_c)
    z, zinfo = iso_zone(m_cln['crest'])
    freq, fmag = compute_fft(a_c, m_cln['fs'])

    print(f"     CLEAN N={len(a_c)}  RMS={m_cln['rms']:.4f}G  "
          f"Peak={m_cln['peak']:.4f}G  CF={m_cln['crest']:.3f}  "
          f"Kurt={m_cln['kurt']:.3f}  Zone {z}")

    all_ds.append({
        'label': label, 'dt_obj': parse_dt(dt_str),
        'fname': fn, 'meta': meta,
        'time_raw': t_raw, 'accel_raw': a_raw, 'metrics_raw': m_raw,
        'time': t_c, 'accel': a_c, 'metrics': m_cln,
        'outlier': rpt,
        'iso_zone': z, 'iso_info': zinfo,
        'fft_freq': freq, 'fft_mag': fmag,
        'top_freqs': top_freqs(freq, fmag, n=8),
    })

# sort chronologically by metadata date
all_ds.sort(key=lambda x: x['dt_obj'])
n_ds = len(all_ds)
print(f"\n📅 Period order: {' → '.join(d['label'] for d in all_ds)}")


# ── 9. Metric comparison table ───────────────────────────────
mk_list = [('rms','RMS (G)'),('peak','Peak (G)'),('p2p','P2P (G)'),
           ('crest','Crest Factor'),('kurt','Kurtosis'),('n','N Samples')]
hdr = f"{'Metric':<22}" + "".join(f"{d['label']:>14}" for d in all_ds)
print("\n" + "─"*len(hdr))
print(hdr); print("─"*len(hdr))
for mk, lbl in mk_list:
    row = f"  {lbl:<20}"
    for d in all_ds:
        row += f"{d['metrics'][mk]:>14.4f}"
    print(row)


# ── 10. Color palette ────────────────────────────────────────
PALETTE = ['#29b6f6','#66bb6a','#ef5350','#ffa726','#ab47bc']
COLORS  = [PALETTE[i % len(PALETTE)] for i in range(n_ds)]


# ── 11. Plots ────────────────────────────────────────────────
plt.style.use('dark_background')

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#0d1117')
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=7)
    if xlabel: ax.set_xlabel(xlabel, color='#78909c', fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color='#78909c', fontsize=8)
    ax.tick_params(colors='#546e7a', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#1c2a3a')
    ax.grid(True, color='#1c2a3a', lw=0.5, alpha=0.8)

# ── Figure layout: rows depend on n_ds ───────────────────────
# Row 0      : waveforms  (1 per dataset)
# Row 1      : FFT        (1 per dataset)
# Row 2 col0 : Trend line  col1: before/after bar (last vs first)
# Row 3 col0 : overlay     col1: outlier summary

n_cols = max(n_ds, 2)
fig = plt.figure(figsize=(10*n_cols, 22))
fig.patch.set_facecolor('#07090f')
gs = gridspec.GridSpec(4, n_cols, figure=fig, hspace=0.50, wspace=0.28)

# Row 0: Waveforms
for col, (ds, color) in enumerate(zip(all_ds, COLORS)):
    ax = fig.add_subplot(gs[0, col])
    ax.plot(ds['time_raw'], ds['accel_raw'],
            color='#455a64', lw=0.5, alpha=0.55, label='Raw')
    ax.plot(ds['time'],     ds['accel'],
            color=color, lw=0.85, label='Cleaned')
    rpt = ds['outlier']
    if rpt['spike_count'] > 0:
        ax.scatter(rpt['spike_times_ms'], rpt['spike_values_g'],
                   color='#ff7043', s=45, zorder=5,
                   label=f"Spike ×{rpt['spike_count']}")
    m = ds['metrics']
    ax.axhline( m['rms'], color='#ffa726', lw=1, ls='--', alpha=0.7)
    ax.axhline(-m['rms'], color='#ffa726', lw=1, ls='--', alpha=0.7)
    style_ax(ax,
             f"Waveform · {ds['label']}  [{ds['meta'].get('datetime','')}]\n"
             f"CF={m['crest']:.2f}  Kurt={m['kurt']:.2f}  Zone {ds['iso_zone']}  "
             f"(removed: {rpt['spike_count']+rpt['trimmed_count']}pt)",
             'Time (ms)', 'Accel (G)')
    ax.legend(fontsize=7.5, facecolor='#0d1117',
              edgecolor='#1c2a3a', labelcolor='white')

# Row 1: FFT
for col, (ds, color) in enumerate(zip(all_ds, COLORS)):
    ax   = fig.add_subplot(gs[1, col])
    freq = ds['fft_freq'];  fmag = ds['fft_mag']
    fmax = min(ds['metrics']['fs']/2, 2000)
    mask = freq <= fmax
    bw   = freq[1]-freq[0] if len(freq)>1 else 1.
    ax.bar(freq[mask], fmag[mask], width=bw, color=color, alpha=0.75, linewidth=0)
    for f0, m0 in ds['top_freqs'][:5]:
        if f0 <= fmax:
            ax.annotate(f'{f0:.0f}Hz', xy=(f0,m0),
                        xytext=(f0+fmax*0.025, m0*1.1),
                        color='white', fontsize=7,
                        arrowprops=dict(arrowstyle='->', color='#78909c', lw=0.7))
    style_ax(ax, f"FFT · {ds['label']}  (0–{fmax:.0f} Hz)",
             'Frequency (Hz)', 'Magnitude (G)')

# Row 2 col 0: Trend Lines (CF, RMS, Peak)
ax_trend = fig.add_subplot(gs[2, 0])
labels_t  = [d['label'] for d in all_ds]
x_t       = np.arange(len(all_ds))
cf_vals   = [d['metrics']['crest'] for d in all_ds]
rms_vals  = [d['metrics']['rms']   for d in all_ds]
pk_vals   = [d['metrics']['peak']  for d in all_ds]

ax_trend.plot(x_t, cf_vals,  'o-', color='#ef5350', lw=2, ms=8, label='Crest Factor')
ax_trend.plot(x_t, rms_vals, 's-', color='#29b6f6', lw=2, ms=7, label='RMS (G)')
ax_trend.plot(x_t, pk_vals,  '^-', color='#ffa726', lw=2, ms=7, label='Peak (G)')
for xi, cf, rms, pk in zip(x_t, cf_vals, rms_vals, pk_vals):
    ax_trend.annotate(f'{cf:.2f}',  (xi, cf),  textcoords='offset points',
                      xytext=(0, 10), ha='center', color='#ef5350', fontsize=9)
    ax_trend.annotate(f'{rms:.3f}', (xi, rms), textcoords='offset points',
                      xytext=(0,-14), ha='center', color='#29b6f6', fontsize=9)
ax_trend.axhline(3.0, color='#ffa726', lw=1.2, ls='--', alpha=0.7, label='CF=3 (Zone B/C)')
ax_trend.axhline(4.0, color='#ef5350', lw=1.2, ls='--', alpha=0.7, label='CF=4 (Zone C/D)')
ax_trend.set_xticks(x_t);  ax_trend.set_xticklabels(labels_t, color='#cdd9e5', fontsize=10)
style_ax(ax_trend, 'Trend: Crest Factor / RMS / Peak (clean data)', '', 'Value')
ax_trend.legend(fontsize=9, facecolor='#0d1117',
                edgecolor='#1c2a3a', labelcolor='white')

# Row 2 col 1: CF Before/After bar for each period
ax_ba = fig.add_subplot(gs[2, 1])
x_ba  = np.arange(n_ds);  w = 0.35
raw_cf  = [d['metrics_raw']['crest'] for d in all_ds]
cln_cf  = [d['metrics']['crest']     for d in all_ds]
b1 = ax_ba.bar(x_ba-w/2, raw_cf, w, label='CF Raw',   color='#546e7a', alpha=0.8)
b2 = ax_ba.bar(x_ba+w/2, cln_cf, w, label='CF Clean', color='#ef5350', alpha=0.85)
for b in list(b1)+list(b2):
    h = b.get_height()
    if h < 30:
        ax_ba.text(b.get_x()+b.get_width()/2, h+0.05,
                   f'{h:.2f}', ha='center', va='bottom', color='white', fontsize=8)
ax_ba.axhline(3.0, color='#ffa726', lw=1.1, ls='--', alpha=0.8, label='CF=3')
ax_ba.axhline(4.0, color='#ef5350', lw=1.1, ls=':',  alpha=0.8, label='CF=4')
ax_ba.set_xticks(x_ba);  ax_ba.set_xticklabels(labels_t, color='#cdd9e5', fontsize=9)
style_ax(ax_ba, 'Crest Factor: Raw vs Cleaned', '', 'CF')
ax_ba.legend(fontsize=9, facecolor='#0d1117',
             edgecolor='#1c2a3a', labelcolor='white')

# Row 3 col 0: Waveform Overlay (clean, first 200 ms)
ax_ov = fig.add_subplot(gs[3, 0])
for ds, color in zip(all_ds, COLORS):
    mask = ds['time'] <= 200
    ax_ov.plot(ds['time'][mask], ds['accel'][mask],
               color=color, lw=0.9, alpha=0.85,
               label=f"{ds['label']}  CF={ds['metrics']['crest']:.2f}")
style_ax(ax_ov, 'Overlay Clean Waveform (0–200 ms)', 'Time (ms)', 'Accel (G)')
ax_ov.legend(fontsize=9, facecolor='#0d1117',
             edgecolor='#1c2a3a', labelcolor='white')

# Row 3 col 1: Outlier summary
ax_sp = fig.add_subplot(gs[3, 1])
n_trim  = [d['outlier']['trimmed_count'] for d in all_ds]
n_spike = [d['outlier']['spike_count']   for d in all_ds]
ax_sp.bar(x_ba-w/2, n_trim,  w, label='Startup trim', color='#ffa726', alpha=0.85)
ax_sp.bar(x_ba+w/2, n_spike, w, label='IQR spike',    color='#ff7043', alpha=0.85)
for xi,yt,ys in zip(x_ba, n_trim, n_spike):
    if yt: ax_sp.text(xi-w/2, yt+.05, str(yt), ha='center', color='white', fontsize=9)
    if ys: ax_sp.text(xi+w/2, ys+.05, str(ys), ha='center', color='white', fontsize=9)
ax_sp.set_xticks(x_ba);  ax_sp.set_xticklabels(labels_t, color='#cdd9e5')
style_ax(ax_sp, 'Outlier Removal Summary', '', 'Points Removed')
ax_sp.legend(fontsize=9, facecolor='#0d1117',
             edgecolor='#1c2a3a', labelcolor='white')

fig.suptitle(
    f"Vibration Analysis  |  {all_ds[0]['meta'].get('equipment','—')}  |  ISO 10816-3",
    color='white', fontsize=14, fontweight='bold', y=0.995)

PLOT_FILE = os.path.join(OUTPUT_DIR, 'vibration_analysis.png')
plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
# plt.show()
print(f"\n💾 บันทึกรูป: {PLOT_FILE}")


# ── 12. DataFrames ───────────────────────────────────────────
def dpct(v2, v1):
    return round((v2-v1)/abs(v1)*100, 2) if v1 != 0 else None

# ── 12a. Metrics per period ──────────────────────────────────
metric_rows = []
for ds in all_ds:
    m  = ds['metrics'];  mr = ds['metrics_raw'];  rpt = ds['outlier']
    z  = ds['iso_zone'];  zi = ds['iso_info']
    metric_rows.append({
        'Period':              ds['label'],
        'Date/Time':           ds['meta'].get('datetime','-'),
        'Equipment':           ds['meta'].get('equipment','-'),
        'Meas. Point':         ds['meta'].get('meas_point','-'),
        'N Raw':               mr['n'],
        'Startup Trimmed':     rpt['trimmed_count'],
        'Spikes Removed':      rpt['spike_count'],
        'N Clean':             m['n'],
        'Fence Low (G)':       rpt.get('fence_lo','-'),
        'Fence High (G)':      rpt.get('fence_hi','-'),
        'RMS raw (G)':         round(mr['rms'],5),
        'RMS clean (G)':       round(m['rms'],5),
        'Peak raw (G)':        round(mr['peak'],5),
        'Peak clean (G)':      round(m['peak'],5),
        'Peak-to-Peak (G)':    round(m['p2p'],5),
        'Std Dev (G)':         round(m['std'],5),
        'CF raw':              round(mr['crest'],4),
        'CF clean':            round(m['crest'],4),
        'Kurtosis raw':        round(mr['kurt'],4),
        'Kurtosis clean':      round(m['kurt'],4),
        'Skewness':            round(m['skew'],4),
        'Fs (Hz)':             round(m['fs'],2),
        'dt (ms)':             round(m['dt'],4),
        'Duration (ms)':       round(m['dur'],3),
        'ISO Zone':            z,
        'Zone Description':    zi['label'],
    })
df_metrics = pd.DataFrame(metric_rows)

# ── 12b. Trend table (all periods, key metrics) ──────────────
trend_rows = []
for mk, lbl in [('rms','RMS (G)'),('peak','Peak (G)'),('p2p','P2P (G)'),
                ('std','Std Dev (G)'),('crest','Crest Factor'),
                ('kurt','Kurtosis'),('skew','Skewness')]:
    row = {'Parameter': lbl}
    vals = []
    for ds in all_ds:
        v = ds['metrics'][mk]
        row[ds['label']] = round(v, 4)
        vals.append(v)
    # delta first→last
    if len(vals) >= 2:
        row['Delta (first→last)'] = round(vals[-1]-vals[0], 5)
        row['Delta % (first→last)'] = dpct(vals[-1], vals[0])
    # flag on CF
    flag = ''
    last_cf = all_ds[-1]['metrics']['crest']
    if mk == 'crest':
        if last_cf > 4.0:   flag = '🔴 CF>4 — Bearing damage likely'
        elif last_cf > 3.0: flag = '⚠ CF>3 — Bearing wear suspected'
        if row.get('Delta % (first→last)') and abs(row['Delta % (first→last)']) > 20:
            flag += ' | ⚠ Change >20%'
    elif mk == 'kurt' and all_ds[-1]['metrics']['kurt'] > 3.0:
        flag = '⚠ Impulsive signal'
    row['Flag'] = flag
    trend_rows.append(row)
df_trend = pd.DataFrame(trend_rows)

# ── 12c. Top Frequencies ─────────────────────────────────────
n_top = 8
all_top = [ds['top_freqs'] for ds in all_ds]
n_f = max(len(t) for t in all_top)
freq_dict = {'Rank': list(range(1, n_f+1))}
for ds, tf in zip(all_ds, all_top):
    lbl = ds['label']
    freq_dict[f'{lbl} Freq (Hz)'] = [round(tf[i][0],2) if i<len(tf) else None for i in range(n_f)]
    freq_dict[f'{lbl} Mag (G)']   = [round(tf[i][1],6) if i<len(tf) else None for i in range(n_f)]
# flag new freqs in last period vs first
if n_ds >= 2:
    first_freqs = [f for f,_ in all_top[0]]
    last_freqs  = [f for f,_ in all_top[-1]]
    freq_dict['New in last period'] = [
        ('⚠ New' if (f is not None and not any(abs(f-(ff or 0))<=5 for ff in first_freqs)) else '')
        for f in freq_dict[f'{all_ds[-1]["label"]} Freq (Hz)']
    ]
df_freq = pd.DataFrame(freq_dict)

# ── 12d. Assessment ───────────────────────────────────────────
last_ds = all_ds[-1];  last_m = last_ds['metrics']
cf_last = last_m['crest']
if cf_last > 4.0:   action, interval = 'URGENT — Plan Bearing Replacement immediately', 'Weekly'
elif cf_last > 3.0: action, interval = 'CAUTION — Bearing inspection; schedule PM', 'Monthly'
else:               action, interval = 'NORMAL — Continue routine PM', 'Quarterly'

assess_rows = [
    ('Section','A — Equipment Info','',''),
    ('Equipment', all_ds[0]['meta'].get('equipment','-'),'',''),
    *[(f"Meas. {ds['label']}", ds['meta'].get('meas_point','-'),'','') for ds in all_ds],
    *[(f"Date {ds['label']}",  ds['meta'].get('datetime','-'),'','')   for ds in all_ds],
    ('Section','B — Key Indicators (latest clean data)','',''),
    ('Period assessed', last_ds['label'],'',''),
    ('RMS (G)',   round(last_m['rms'],5),'',''),
    ('Peak (G)',  round(last_m['peak'],5),'',''),
    ('CF',        round(cf_last,4),'',
        '🔴 >4.0' if cf_last>4 else ('⚠ >3.0' if cf_last>3 else '✓ Normal')),
    ('Kurtosis',  round(last_m['kurt'],4),'',
        '⚠ Impulsive' if last_m['kurt']>3 else '✓ Normal'),
    ('ISO Zone',  last_ds['iso_zone'], last_ds['iso_info']['label'],''),
    ('Section','C — ISO Zone per Period','',''),
    *[(f"Zone {ds['label']}", ds['iso_zone'], ds['iso_info']['label'],'') for ds in all_ds],
    ('Section','D — Top Frequencies (latest period)','',''),
    *[(f"Top{i+1} Hz", round(all_top[-1][i][0],2) if i<len(all_top[-1]) else '-',
       f"Mag={round(all_top[-1][i][1],5)} G" if i<len(all_top[-1]) else '-','')
      for i in range(5)],
    ('Section','E — Recommendations','',''),
    ('Overall Condition', action,'',''),
    ('Monitoring Interval', interval,'',''),
    ('Action 1','Measure Velocity (mm/s) for direct ISO 10816-3 comparison','',''),
    ('Action 2','Perform Envelope Analysis for Bearing defect frequency','',''),
    ('Action 3','Calculate BPFO / BPFI / BSF to confirm bearing fault','',''),
    ('Action 4','Check bearing temperature and lubrication','',''),
    ('Action 5','If CF >4.0 at next measurement: execute Bearing Replacement','',''),
]
df_assess = pd.DataFrame(assess_rows, columns=['Item','Value','Detail','Flag'])

# ── 12e. Outlier detail ───────────────────────────────────────
out_rows = []
for ds in all_ds:
    rpt = ds['outlier']
    out_rows.append({
        'Period':            ds['label'],
        'N Raw':             rpt['n_raw'],
        'Startup Trim (pts)':rpt['trimmed_count'],
        'Spikes Removed':    rpt['spike_count'],
        'N Clean':           rpt['n_clean'],
        'IQR k':             rpt['iqr_k'],
        'Q1 (G)':            rpt.get('q1','-'),
        'Q3 (G)':            rpt.get('q3','-'),
        'IQR (G)':           rpt.get('iqr','-'),
        'Fence Low (G)':     rpt.get('fence_lo','-'),
        'Fence High (G)':    rpt.get('fence_hi','-'),
        'Peak Raw (G)':      rpt['spike_max_raw'],
        'Peak After (G)':    rpt['peak_after'],
        'Replace Method':    rpt['method'],
        'Spike Times (ms)':  str(rpt['spike_times_ms']),
        'Spike Values (G)':  str(rpt['spike_values_g']),
    })
df_outlier = pd.DataFrame(out_rows)


# ── 13. Print Signal Processing Results ─────────────────────
for title, df in [
    ('SECTION 1 — VIBRATION METRICS',         df_metrics),
    ('SECTION 2 — TREND (all periods)',        df_trend),
    ('SECTION 3 — TOP DOMINANT FREQUENCIES',   df_freq.fillna('-')),
    ('SECTION 4 — ASSESSMENT SUMMARY',         df_assess),
    ('SECTION 5 — OUTLIER REMOVAL DETAIL',     df_outlier),
]:
    print('\n'+'='*74)
    print(f'  📋 {title}')
    print('='*74)
    print(df.to_string(index=False))


# ============================================================
# ██████████████████████████████████████████████████████████
#  ML SECTION  —  3 modules
#  ① Anomaly Detection   (Isolation Forest)
#  ② CF Trend Forecast   (Linear + Polynomial Regression)
#  ③ Zone Classification (Rule-based vs Random Forest)
# ██████████████████████████████████████████████████████████
# ============================================================
from ml_models import detect_anomalies, forecast_trend, classify_zones

print("\n" + "█"*60)
print("  🤖  ML ANALYSIS")
print("█"*60)

# ============================================================
# ML-①  ANOMALY DETECTION — Isolation Forest
# ============================================================
print("\n" + "─"*60)
print("  ① ANOMALY DETECTION  (Isolation Forest)")
print("─"*60)

scores, preds, anomaly_labels, df_anomaly, iso, X_all, feat_names = detect_anomalies(all_ds)
labels_t = [ds['label'] for ds in all_ds]

print(f"\n  {'Period':<10} {'Score':>10} {'Status':<16} {'CF':>8} {'Kurt':>8}")
print("  " + "─"*56)
for lbl, sc, al, ds in zip(labels_t, scores, anomaly_labels, all_ds):
    cf   = ds['metrics']['crest']
    kurt = ds['metrics']['kurt']
    print(f"  {lbl:<10} {sc:>10.4f} {al:<16} {cf:>8.3f} {kurt:>8.3f}")

print(f"\n  📌 Score ยิ่งต่ำ = ยิ่งผิดปกติ  |  threshold = 0")
print(f"  📌 Isolation Forest ใช้ feature {len(feat_names)} ตัว: "
      f"RMS, Peak, CF, Kurtosis, Skewness, Top-5 Frequencies")

# Feature importance (approximated via score sensitivity)
feature_importance = []
baseline_scores = iso.decision_function(X_all)
for fi, fname in enumerate(feat_names):
    X_perturb = X_all.copy()
    X_perturb[:, fi] = np.mean(X_all[:, fi])   # zero out feature
    perturb_scores = iso.decision_function(X_perturb)
    sensitivity = float(np.mean(np.abs(baseline_scores - perturb_scores)))
    feature_importance.append((fname, sensitivity))

feature_importance.sort(key=lambda x: x[1], reverse=True)
print(f"\n  Top-5 features ที่มีผลต่อ anomaly detection:")
for fname, imp in feature_importance[:5]:
    bar = '█' * int(imp * 200)
    print(f"    {fname:<12} {imp:.5f}  {bar}")


# ============================================================
# ML-②  CF TREND FORECAST  —  Linear + Polynomial Regression
# ============================================================
print("\n" + "─"*60)
print("  ② CF TREND FORECAST  (Linear + Polynomial Regression)")
print("─"*60)

lin_reg, poly_fn, future_dates, cf_lin_f, cf_poly_f, df_forecast, avg_interval_days, slope, intercept = forecast_trend(all_ds, ISO_ZONES)
x_idx = np.arange(len(all_ds), dtype=float)

print(f"\n  Fitted model:")
print(f"    Linear    : CF = {slope:+.4f} × period + {intercept:.4f}  (R²=...)")
print(f"    Polynomial: degree={min(2,len(all_ds)-1)}  (R²=...)")
print(f"    Avg interval between periods: {avg_interval_days:.0f} days")

print(f"\n  {'Period':<12} {'Date (est.)':>14} {'CF Linear':>12} {'CF Poly':>12} {'Zone (Lin)':>12}")
print("  " + "─"*66)
for i, (fd, cf_l, cf_p) in enumerate(zip(future_dates, cf_lin_f, cf_poly_f)):
    z_l = next((z for z,info in ISO_ZONES.items() if cf_l <= info['max']), 'D')
    tag = fd.strftime('%b-%y')
    print(f"  {tag:<12} {fd.strftime('%d-%b-%Y'):>14} {cf_l:>12.3f} {cf_p:>12.3f} {('Zone '+z_l):>12}")

dt_objs  = [ds['dt_obj'] for ds in all_ds]
print(f"\n  Threshold crossing forecast (Linear model):")
for thresh in [3.0, 4.0, 5.0]:
    if slope > 0:
        x_cross = (thresh - intercept) / slope
        if x_cross > x_idx[-1]:
            days_from_last = (x_cross - x_idx[-1]) * avg_interval_days
            cross_date = dt_objs[-1] + timedelta(days=days_from_last)
            print(f"    CF = {thresh:.1f}  →  คาดถึงประมาณ {cross_date.strftime('%b-%Y')} "
                  f"(อีก ~{int(days_from_last)} วัน)")
        elif x_cross <= x_idx[-1]:
            print(f"    CF = {thresh:.1f}  →  เกินแล้ว ({labels_t[-1]})")
    else:
        print(f"    CF = {thresh:.1f}  →  trend ไม่เพิ่ม (slope={slope:.4f})")


# ============================================================
# ML-③  ZONE CLASSIFICATION  —  Rule-based vs Random Forest
# ============================================================
print("\n" + "─"*60)
print("  ③ ZONE CLASSIFICATION  (Rule-based vs Random Forest)")
print("─"*60)

rf_pipe, rf_preds, rf_probas, rf_classes, df_classify = classify_zones(all_ds, ISO_ZONES)

X_A = gen_samples((1.0,2.5),  (-1.0, 0.5), (0.05,0.30))
X_B = gen_samples((2.5,3.0),  (-0.5, 1.5), (0.20,0.50))
X_C = gen_samples((3.0,4.0),  ( 0.0, 3.0), (0.30,0.70))
X_D = gen_samples((4.0,8.0),  ( 1.0,10.0), (0.40,1.00))

X_train = np.vstack([X_A, X_B, X_C, X_D])
y_train = (['A']*len(X_A) + ['B']*len(X_B) +
           ['C']*len(X_C) + ['D']*len(X_D))

# ── Train Random Forest ───────────────────────────────────────
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=300, random_state=42,
                                  class_weight='balanced'))
])
rf_pipe.fit(X_train, y_train)

# ── Predict real data ─────────────────────────────────────────
X_real = np.array([[ds['metrics']['rms'],
                    ds['metrics']['peak'],
                    ds['metrics']['crest'],
                    ds['metrics']['kurt'],
                    ds['metrics']['std']] for ds in all_ds])

rf_zones  = rf_pipe.predict(X_real)
rf_probas = rf_pipe.predict_proba(X_real)
rf_classes= rf_pipe.classes_

print(f"\n  {'Period':<10} {'Rule Zone':>10} {'RF Zone':>10} {'Match':>8}  Probabilities A/B/C/D")
print("  " + "─"*72)
clf_rows = []
for ds, rz, rfz, prob in zip(all_ds, [d['iso_zone'] for d in all_ds],
                               rf_zones, rf_probas):
    match = '✅' if rz == rfz else '⚠ Diff'
    prob_str = '  '.join([f"{c}:{p:.2f}" for c,p in zip(rf_classes,prob)])
    print(f"  {ds['label']:<10} {('Zone '+rz):>10} {('Zone '+rfz):>10} {match:>8}  {prob_str}")
    clf_rows.append({
        'Period':       ds['label'],
        'Rule Zone':    rz,
        'RF Zone':      rfz,
        'Match':        'Yes' if rz==rfz else 'No',
        **{f'P(Zone {c})': round(p,4) for c,p in zip(rf_classes,prob)},
        'CF':           round(ds['metrics']['crest'],4),
        'Kurtosis':     round(ds['metrics']['kurt'],4),
        'RMS':          round(ds['metrics']['rms'],5),
    })

print(f"\n  📌 Random Forest trained on synthetic data ({len(X_train)} samples)")
print(f"     Features: RMS, Peak, Crest Factor, Kurtosis, Std Dev")
print(f"     ⚠  ถ้า Rule vs RF ต่างกัน → ควรตรวจสอบข้อมูลเพิ่มเติม")

# RF Feature importance
rf_model   = rf_pipe.named_steps['rf']
feat_imp   = rf_model.feature_importances_
feat_names_clf = ['RMS','Peak','CF','Kurtosis','Std Dev']
print(f"\n  RF Feature Importance:")
for fn, fi in sorted(zip(feat_names_clf, feat_imp), key=lambda x:-x[1]):
    bar = '█' * int(fi * 40)
    print(f"    {fn:<12} {fi:.4f}  {bar}")

df_classify = pd.DataFrame(clf_rows)


# ============================================================
# ML PLOT — 3 subplots
# ============================================================
fig_ml, axes = plt.subplots(1, 3, figsize=(22, 7))
fig_ml.patch.set_facecolor('#07090f')

ZONE_COLORS_MAP = {'A':'#26a69a','B':'#66bb6a','C':'#ffa726','D':'#ef5350'}

# ── Plot ①: Anomaly Score ─────────────────────────────────────
ax = axes[0]; ax.set_facecolor('#0d1117')
bar_colors = ['#ef5350' if p==-1 else '#29b6f6' for p in preds]
bars = ax.bar(labels_t, scores, color=bar_colors, alpha=0.85, edgecolor='white', lw=0.5)
ax.axhline(0, color='#ffa726', lw=1.5, ls='--', label='Threshold (0)')
for b, sc, al in zip(bars, scores, anomaly_labels):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
            f'{sc:.3f}', ha='center', va='bottom', color='white', fontsize=9)
ax.set_title('① Anomaly Detection\n(Isolation Forest Score)',
             color='white', fontweight='bold', fontsize=11)
ax.set_ylabel('Anomaly Score (lower = more anomalous)', color='#78909c', fontsize=9)
ax.tick_params(colors='#546e7a'); ax.legend(fontsize=9, facecolor='#0d1117', labelcolor='white')
for sp in ax.spines.values(): sp.set_edgecolor('#1c2a3a')
ax.grid(True, color='#1c2a3a', lw=0.5, alpha=0.8)

# ── Plot ②: CF Trend Forecast ─────────────────────────────────
ax = axes[1]; ax.set_facecolor('#0d1117')
x_hist = np.arange(len(all_ds))
x_fut  = np.arange(len(all_ds), len(all_ds)+n_future)
x_all_plot = np.linspace(0, len(all_ds)+n_future-1, 200)

# zone background bands
for z, info, alpha in [('A',ISO_ZONES['A'],0.15),('B',ISO_ZONES['B'],0.15),
                        ('C',ISO_ZONES['C'],0.15),('D',ISO_ZONES['D'],0.10)]:
    lo_b = 0 if z=='A' else ISO_ZONES[chr(ord(z)-1)]['max']
    hi_b = min(info['max'], max(cf_vals.max(), cf_poly_f.max())*1.3)
    ax.axhspan(lo_b, hi_b, alpha=alpha, color=ZONE_COLORS_MAP[z], label=f'Zone {z}')

# fitted lines
ax.plot(x_all_plot,
        lin_reg.predict(x_all_plot.reshape(-1,1)),
        color='#29b6f6', lw=1.5, ls='--', alpha=0.8, label='Linear fit')
ax.plot(x_all_plot, poly_fn(x_all_plot),
        color='#ab47bc', lw=1.5, ls='-.', alpha=0.8, label=f'Poly fit')

# actual points
ax.scatter(x_hist, cf_vals, color='white', s=80, zorder=5, label='Actual CF')
for xi, cf, lbl in zip(x_hist, cf_vals, labels_t):
    ax.annotate(f'{lbl}\nCF={cf:.2f}', (xi,cf),
                textcoords='offset points', xytext=(0,12),
                ha='center', color='white', fontsize=8)

# forecast points
ax.scatter(x_fut, cf_lin_f,  color='#29b6f6', s=60, marker='D',
           zorder=5, label='Forecast (Lin)')
ax.scatter(x_fut, cf_poly_f, color='#ab47bc', s=60, marker='D',
           zorder=5, label='Forecast (Poly)')
for xi, cf_l, cf_p, fd in zip(x_fut, cf_lin_f, cf_poly_f, future_dates):
    ax.annotate(fd.strftime('%b-%y'), (xi, cf_l),
                textcoords='offset points', xytext=(0,10),
                ha='center', color='#29b6f6', fontsize=7.5)

all_labels = labels_t + [fd.strftime('%b-%y') for fd in future_dates]
ax.set_xticks(np.arange(len(all_labels)))
ax.set_xticklabels(all_labels, color='#cdd9e5', fontsize=8, rotation=30)
ax.set_title('② CF Trend Forecast\n(Linear + Polynomial Regression)',
             color='white', fontweight='bold', fontsize=11)
ax.set_ylabel('Crest Factor', color='#78909c', fontsize=9)
ax.tick_params(colors='#546e7a')
ax.legend(fontsize=7.5, facecolor='#0d1117', edgecolor='#1c2a3a',
          labelcolor='white', ncol=2)
for sp in ax.spines.values(): sp.set_edgecolor('#1c2a3a')
ax.grid(True, color='#1c2a3a', lw=0.5, alpha=0.8)

# ── Plot ③: Zone Classification Probability ───────────────────
ax = axes[2]; ax.set_facecolor('#0d1117')
x3    = np.arange(len(all_ds))
width = 0.2
for ci, (cls, color) in enumerate(ZONE_COLORS_MAP.items()):
    if cls in rf_classes:
        ci_idx = list(rf_classes).index(cls)
        probs  = rf_probas[:, ci_idx]
        bars3  = ax.bar(x3 + ci*width, probs, width,
                        label=f'Zone {cls}', color=color, alpha=0.85)
        for b, p in zip(bars3, probs):
            if p > 0.05:
                ax.text(b.get_x()+b.get_width()/2, p+0.01,
                        f'{p:.2f}', ha='center', va='bottom',
                        color='white', fontsize=7.5)

ax.set_xticks(x3 + width*1.5)
ax.set_xticklabels(labels_t, color='#cdd9e5', fontsize=10)
ax.set_title('③ Zone Classification\n(Random Forest Probability)',
             color='white', fontweight='bold', fontsize=11)
ax.set_ylabel('Probability', color='#78909c', fontsize=9)
ax.set_ylim(0, 1.15)
ax.tick_params(colors='#546e7a')
ax.legend(fontsize=9, facecolor='#0d1117', edgecolor='#1c2a3a', labelcolor='white')
for sp in ax.spines.values(): sp.set_edgecolor('#1c2a3a')
ax.grid(True, color='#1c2a3a', lw=0.5, alpha=0.8)

fig_ml.suptitle(
    f"ML Analysis  |  {all_ds[0]['meta'].get('equipment','—')}  |  ISO 10816-3",
    color='white', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()

ML_PLOT_FILE = os.path.join(OUTPUT_DIR, 'vibration_ml_analysis.png')
plt.savefig(ML_PLOT_FILE, dpi=150, bbox_inches='tight',
            facecolor=fig_ml.get_facecolor())
# plt.show()
print(f"\n💾 บันทึกรูป ML: {ML_PLOT_FILE}")

# ── Print ML results ──────────────────────────────────────────
for title, df in [
    ('SECTION 6 — ANOMALY DETECTION (Isolation Forest)', df_anomaly),
    ('SECTION 7 — CF TREND FORECAST',                    df_forecast),
    ('SECTION 8 — ZONE CLASSIFICATION (RF)',             df_classify),
]:
    print('\n'+'='*74)
    print(f'  🤖 {title}')
    print('='*74)
    print(df.to_string(index=False))


# ── 14. Export CSV (all 8 sections) ──────────────────────────
CSV_FILE = os.path.join(OUTPUT_DIR, 'vibration_full_report.csv')
with open(CSV_FILE, 'w', encoding='utf-8-sig', newline='') as f:
    f.write('=== 1. VIBRATION METRICS (clean) ===\n');       df_metrics.to_csv(f,index=False);        f.write('\n')
    f.write('=== 2. TREND (all periods) ===\n');              df_trend.to_csv(f,index=False);          f.write('\n')
    f.write('=== 3. TOP DOMINANT FREQUENCIES ===\n');         df_freq.fillna('-').to_csv(f,index=False);f.write('\n')
    f.write('=== 4. ASSESSMENT SUMMARY ===\n');               df_assess.to_csv(f,index=False);         f.write('\n')
    f.write('=== 5. OUTLIER REMOVAL DETAIL ===\n');           df_outlier.to_csv(f,index=False);        f.write('\n')
    f.write('=== 6. ANOMALY DETECTION (Isolation Forest) ===\n'); df_anomaly.to_csv(f,index=False);   f.write('\n')
    f.write('=== 7. CF TREND FORECAST ===\n');                df_forecast.to_csv(f,index=False);       f.write('\n')
    f.write('=== 8. ZONE CLASSIFICATION (Random Forest) ===\n');  df_classify.to_csv(f,index=False)

print(f"\n💾 Exported: {CSV_FILE}  (8 sections)")

print("\n✅ Done.")
