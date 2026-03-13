import re
import numpy as np

# ── 4. Parser ────────────────────────────────────────────────
NUM_RE = re.compile(r'-?(?:\d+\.?\d*|\.\d+)')
SKIP_KW = ('---', 'Time (', 'Amplitude', '***',
           'Equipment', 'Meas.', 'Date/Time', 'Waveform')

def parse_waveform_txt(content_bytes):
    """
    Parses a CSV/TXT format raw vibration waveform file.
    Args:
        content_bytes (bytes): The raw file contents.
    Returns:
        tuple (meta, t, a): Metadata dictionary, Time Array (ms), Acceleration Array (G).
    """
    text  = content_bytes.decode('utf-8', errors='ignore')
    meta  = {}
    pairs = []
    for raw in text.splitlines():
        line = raw.rstrip('\r\n');  s = line.strip()
        if 'Equipment:'   in line: meta['equipment']  = line.split('Equipment:')[-1].strip()
        if 'Meas. Point:' in line: meta['meas_point'] = line.split('Meas. Point:')[-1].strip()
        if 'Date/Time:'   in line:
            meta['datetime'] = line.split('Date/Time:')[-1].split('Amplitude:')[0].strip()
        if 'Amplitude:' in line and 'Date/Time:' not in line:
            meta['unit'] = line.split('Amplitude:')[-1].strip()
        if not s or any(k in s for k in SKIP_KW): continue
        toks = NUM_RE.findall(s)
        for i in range(0, len(toks)-1, 2):
            try:
                tv, av = float(toks[i]), float(toks[i+1])
                if tv >= 0: pairs.append((tv, av))
            except ValueError: pass
    if not pairs:
        raise ValueError("ไม่พบข้อมูล numeric — ตรวจสอบ format ไฟล์")
    pairs.sort(key=lambda x: x[0])
    t = np.array([p[0] for p in pairs])
    a = np.array([p[1] for p in pairs])
    _, idx = np.unique(t, return_index=True)
    return meta, t[idx], a[idx]
