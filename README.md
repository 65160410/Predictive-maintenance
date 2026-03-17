# การวิเคราะห์รูปคลื่นความสั่นสะเทือน — ISO 10816-3 (Vibration Waveform Analysis)

เวอร์ชัน 8 | รูปแบบ Jupyter Notebook

---

## ภาพรวม (Overview)

เครื่องมือวิเคราะห์รูปคลื่นความสั่นสะเทือนผ่าน Jupyter Notebook สำหรับติดตามสภาพการทำงานของเครื่องจักรหมุน (Rotating Machinery) Notebook นี้จะประมวลผลข้อมูลความเร่ง (Accelerometer) ดิบรูปแบบ .txt, ตรวจสอบคุณภาพของสัญญาณ, ตัดข้อมูลที่ผิดปกติ (Outliers) ออก, คำนวณค่าตัวชี้วัดความสั่นสะเทือนที่สำคัญ, และสร้างกราฟวิเคราะห์พร้อมรายงาน CSV ที่สอดคล้องกับมาตรฐาน ISO 10816-3

ออกแบบมาสำหรับการวัดผล 2–5 ระยะเวลา (Periods) และในเวอร์ชันนี้มีการรวบรวม **Session 7** สำหรับทำนาย (Forecast) ค่า RMS / CF / Kurtosis ของ period ถัดไปด้วย Quadratic Regression (Polynomial Degree 2) และแสดงกราฟแนวโน้มอัตโนมัติ

การทำงานของระบบแบ่งออกเป็น 4 ขั้นตอนหลัก:
1. **Signal Processing & Cleaning** — กรอง Outlier/Spike และตรวจสอบความถูกต้องของสัญญาณ
2. **Metrics Computation** — คำนวณ RMS, Peak, Crest Factor (CF), FFT
3. **ISO 10816-3 Zoning** — จัดหมวดหมู่โซน A / B / C / D
4. **Trend Analysis & Forecast** — วิเคราะห์แนวโน้มด้วย Quadratic Regression และคาดการณ์ period ถัดไป

---

## ความต้องการของระบบ (Requirements)

```
Python >= 3.9
scipy
matplotlib
numpy
pandas
scikit-learn
```

ติดตั้งไลบรารี:

```bash
pip install scipy matplotlib numpy pandas scikit-learn
```

---

## โครงสร้างไฟล์ (File Structure)

```
project/
├── vibration_analysis_v2.ipynb    # Notebook หลัก
├── output/                        # สร้างอัตโนมัติเมื่อรันครั้งแรก
│   ├── outlier_diag_<period>.png
│   ├── raw_data_<period>.png
│   ├── signal_quality_dashboard.png
│   ├── vibration_analysis.png
│   └── vibration_full_report.csv
└── README.md
```

---

## รูปแบบไฟล์ข้อมูลนำเข้า (Input File Format)

ไฟล์ข้อความธรรมดา (.txt) โดยแต่ละไฟล์ต้องประกอบด้วย:

- บรรทัดส่วนหัว (Header) ที่มีฟิลด์ `Equipment:`, `Meas. Point:`, `Date/Time:`, `Amplitude:`
- ข้อมูลตัวเลข 2 คอลัมน์: เวลา (ms) และ ความเร่ง (G)

ตัวอย่างส่วนหัวของไฟล์:

```text
Equipment:    Motor Compressor OAH-06_A
Meas. Point:  A_CH-06
Date/Time:    28-Jun-24 08:56:47  Amplitude: G
---
0.0000   0.0123
0.1000  -0.0045
...
```

รูปแบบวันที่ที่รองรับ: `DD-Mon-YY HH:MM:SS`, `DD-Mon-YYYY HH:MM:SS`, `DD-MM-YYYY HH:MM:SS`, `YYYY-MM-DD HH:MM:SS`

---

## ตัวชี้วัดที่คำนวณ (Vibration Parameters)

| Parameter    | ความหมาย                           |
| ------------ | ---------------------------------- |
| RMS          | พลังงานการสั่น (Root Mean Square) |
| Peak         | ค่าความเร่งสูงสุด                 |
| Peak-to-Peak | ความต่างระหว่างค่าสูงสุด–ต่ำสุด  |
| Std Dev      | การกระจายของสัญญาณ                |
| Crest Factor | อัตราส่วน Peak/RMS — บ่งชี้ impulse fault |
| Kurtosis     | ความแหลมของการกระจาย — บ่งชี้ bearing damage |
| Skewness     | ความเบ้ของ distribution            |

---

## เซสชั่นต่างๆ (Sessions)

Notebook ถูกแบ่งออกเป็นส่วนๆ (Session) ต้องรันตามลำดับ

### Session 0 — Setup and Imports
นำเข้าไลบรารีทั้งหมดและตั้งค่าตัวแปรมาตรฐาน (ต้องรันก่อน Session อื่นๆ)

### Session 0b — Parser
กำหนดฟังก์ชัน `parse_waveform_txt()` สำหรับอ่านไฟล์ .txt

### Session 0c — Signal Processing
กำหนดฟังก์ชันหลัก: `remove_outliers()`, `compute_metrics()`, `iso_zone()`, `compute_fft()`, `top_freqs()`, `style_ax()`

> ⚠ **สำคัญ:** `cleaning_reports = []` ถูก initialize ที่นี่ ต้องรัน Session 0c **ก่อน** Session 3 เสมอ

### Session 1 — Load Files
เปิดหน้าต่างเลือกไฟล์ (tkinter) เพื่อเลือกไฟล์ .txt จำนวน 2–5 ไฟล์

### Session 2 — Outlier Config
ตั้งค่าพารามิเตอร์การจัดการ Outlier **(ปรับก่อนรัน Session 3)**

| พารามิเตอร์ | ค่าแนะนำ | คำอธิบาย |
|---|---|---|
| `TRIM_START_MS` | 1.0 | ตัดข้อมูลช่วง startup transient (ms) |
| `IQR_MULTIPLIER` | **1.5** | ตัวคูณ IQR fence — ค่าน้อย = เข้มงวดกว่า (ค่า 8.0 กว้างเกินไป) |
| `REPLACE_METHOD` | interpolate | วิธีแทนที่ spike: `interpolate`, `median`, `zero` |

### Session 3 — Parse and Clean Data
โหลดและทำความสะอาดข้อมูลแต่ละไฟล์ พร้อมกราฟวินิจฉัย 3 กราฟย่อยต่อไฟล์ ผลลัพธ์เก็บในตัวแปร `all_ds` และ `cleaning_reports`

### Session 3b — Raw Data Plots
แสดงกราฟข้อมูลดิบ 4 กราฟย่อยต่อไฟล์: Waveform / Zoom / FFT / Histogram

### Session 3c — Outlier Removal Audit
ตรวจสอบรายการ spike ที่ถูกแก้ไข และแสดงตารางเปรียบเทียบ Raw vs Cleaned

### Session 4 — Data Validation
ตรวจสอบคุณภาพสัญญาณแต่ละไฟล์อัตโนมัติ เกณฑ์ที่ใช้:

| เกณฑ์ | ค่าจำกัด (เริ่มต้น) | สิ่งที่ตรวจพบ |
|---|---|---|
| `PEAK_MAX_G` | 50 G | Sensor overrange หรือแรงกระแทก |
| `CF_RAW_MAX` | 30 | CF สูงผิดปกติ |
| `KURT_RAW_MAX` | 500 | การกระแทกในจุดเดียว (ไม่ใช่การสั่นต่อเนื่อง) |
| `FLAT_RATIO_MAX` | 30% | ข้อมูลขาดหาย (เซ็นเซอร์หลุด/สายมีปัญหา) |

### Session 4b — Dataset Selection
สร้างตัวแปร `ds_valid` จากไฟล์ที่ผ่านเกณฑ์

```python
USE_VALID_ONLY = True   # True = ใช้เฉพาะไฟล์ที่ผ่านเกณฑ์
```

### Session 5 — Metric Summary Table
ตารางเปรียบเทียบค่าตัวชี้วัดตลอดช่วงเวลาทั้งหมด

### Session 6 — Main Analysis Plots
กราฟภาพรวมบันทึกใน `output/vibration_analysis.png`:
- แถว 0: Cleaned waveform + spike markers + RMS reference
- แถว 1: FFT spectrum per period
- แถว 2: Trend lines (CF / RMS / Peak) + CF comparison
- แถว 3: 200ms overlay + Outlier removal summary

### Session 7 — Diagnostic Summary & Trend (Merged)
สรุปสภาพเครื่องพร้อมกราฟพยากรณ์และแนวโน้ม (Quadratic Regression Degree 2):
1. **Diagnostic Summary**: ISO Zone + ค่า CF/RMS/Peak/Kurtosis พร้อม flag เตือน
2. **Trend Analysis (Forecast)**: ตารางทำนายค่าของ period ถัดไป พร้อม **ISO Zone พยากรณ์** สำหรับค่า Crest Factor
3. **Dominant Frequencies**: Top 3 dominant frequencies ต่อ period
4. **Recommendation**: คำแนะนำการซ่อมบำรุงตามค่าจริงและค่าพยากรณ์
5. **Visualization**: กราฟแนวโน้ม 2x2 (RMS, Peak, CF, Kurtosis) แสดงเส้นโค้ง Polynomial

### Session 7a — RMS / CF / Kurtosis Trend Forecast (Detailed Graphs)
สำหรับผู้ที่ต้องการดูพยากรณ์แนวโน้มแบบละเอียด:
- แสดงตารางพยากรณ์พร้อมเปอร์เซ็นต์การเปลี่ยนแปลง (Δ%)
- แสดงกราฟแนวโน้ม **2x2 Grid Visualization** (RMS, Peak, CF, Kurtosis) แยกต่างหาก
- ใช้ Quadratic Regression (Degree 2) ในการคำนวณเส้นโค้ง

ผลลัพธ์เก็บใน `forecast_results` (dict) เพื่อส่งต่อ Session 8

### Session 8 — Build DataFrames, Forecast Summary & Export CSV
จัดทำ DataFrame 6 ชุด แล้วส่งออกไฟล์ `output/vibration_full_report.csv` (รองรับ Quadratic structure):

| ส่วน | ข้อมูล |
|---|---|
| 0 | Data Validation results |
| 1 | Vibration Metrics (Raw & Clean) |
| 2 | Trend — delta %, slope |
| 3 | Dominant Frequencies |
| 4 | Outlier Detail |
| **5 ★** | **Next Period Forecast** (Quadratic + **ISO Zone**) |

แสดงกล่องสรุปท้าย output:
```
██████████████████████████████████████████████████
  🔮  สรุปคาดการณ์ Period ถัดไป
██████████████████████████████████████████████████
  RMS คาดว่า  : 0.6234 G  (↑ +5.3%)
  CF คาดว่า   : 2.5412    (↑ +2.1%)
  คำแนะนำ    : ✅ NORMAL — ค่าทุกตัวอยู่ในเกณฑ์ปกติ
```

---

## ลำดับการรัน (Run Order)

```
Session 0 → 0b → 0c → 1 → 2 → 3 → 3b → 3c → 4 → 4b
         → 5 → 6 → 7 → (7a) → 8
```

---

## ส่วนอ้างอิงมาตรฐาน ISO 10816-3

ระบบใช้ Crest Factor เทียบโซนตาม ISO 10816-3 (การประเมินโดยตรงต้องวัด Velocity mm/s แยกต่างหาก)

| Zone | ช่วงค่า CF | สภาวะเครื่องจักร |
|---|---|---|
| A | < 2.5 | ใหม่ / ทำงานปกติดี |
| B | 2.5 – 3.0 | ทำงานได้ต่อเนื่องในระยะยาว |
| C | 3.0 – 4.0 | ระมัดระวัง — เฝ้าระวังอย่างใกล้ชิด |
| D | > 4.0 | เสี่ยงพัง — หยุดและดำเนินการทันที |

---

## ข้อจำกัดที่ควรทราบ (Known Limitations)

- ISO 10816-3 ใช้ Velocity (mm/s) ไม่ใช่ความเร่ง — CF เป็นแค่การประมาณเบื้องต้น
- Trend/Forecast ในเวอร์ชันนี้ใช้ **Quadratic Regression (Degree 2)** เพื่อให้เห็นความโค้งของแนวโน้ม — แนะนำให้มีข้อมูล 3-5 periods ขึ้นไปเพื่อความแม่นยำ
- `IQR_MULTIPLIER = 8.0` กว้างเกินไปสำหรับข้อมูลที่มี spike ขนาดใหญ่ — แนะนำใช้ **1.5**
- FFT ที่ Fs = 10,000 Hz, N = 4096 samples ให้ความละเอียด ~1.9 Hz/bin
- ไม่มี Envelope Analysis / Bearing defect frequency (BPFO/BPFI/BSF) — ต้องคำนวณแยก

---

## สิ่งที่ควรทำเมื่ออยู่ใน Zone C หรือ D

1. วัด Velocity (mm/s) เทียบ ISO 10816-3 โดยตรง
2. ทำ Envelope Analysis/Demodulation หา Bearing defect frequency
3. คำนวณ BPFO/BPFI/BSF เทียบ FFT peaks
4. ตรวจอุณหภูมิ Bearing และสภาพ Lubrication
5. ถ้า CF คาดว่าเกิน 4.0 ใน period ถัดไป → วางแผนเปลี่ยน Bearing ทันที

---

## ไฟล์เอาต์พุต (Output Files)

| ชื่อไฟล์ | คำอธิบาย |
|---|---|
| `outlier_diag_<period>.png` | กราฟวินิจฉัย Outlier รายไฟล์ (3 subplots) |
| `raw_data_<period>.png` | กราฟข้อมูลดิบรายไฟล์ (4 subplots) |
| `signal_quality_dashboard.png` | Dashboard คุณภาพสัญญาณก่อน cleaning |
| `vibration_analysis.png` | กราฟภาพรวมผลวิเคราะห์ทุก period |
| `vibration_full_report.csv` | รายงานเต็มรูปแบบ 6 sections รวม Forecast |
