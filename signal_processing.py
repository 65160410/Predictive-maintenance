import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq

# ── 3. Outlier Config ────────────────────────────────────────
TRIM_START_MS  = 1.0       # ตัด startup transient (ms)  |  0 = ไม่ตัด
IQR_MULTIPLIER = 8.0       # spike threshold k×IQR       |  ลด = ตัดเพิ่ม
REPLACE_METHOD = 'interpolate'  # 'interpolate' | 'median' | 'zero'

# ── 5. Outlier Removal ───────────────────────────────────────
def remove_outliers(time_ms, accel_g, label='',
                    trim_ms=TRIM_START_MS,
                    iqr_k=IQR_MULTIPLIER,
                    method=REPLACE_METHOD):
    t, a = time_ms.copy(), accel_g.copy()
    rpt  = {'label': label, 'n_raw': len(a),
            'trim_ms': trim_ms, 'iqr_k': iqr_k, 'method': method,
            'trimmed_count': 0, 'spike_count': 0,
            'spike_max_raw': float(np.max(np.abs(a))),
            'spike_times_ms': [], 'spike_values_g': []}

    # Step 1: startup trim
    if trim_ms > 0:
        keep = t >= trim_ms
        rpt['trimmed_count'] = int(np.sum(~keep))
        t, a = t[keep], a[keep]
        if rpt['trimmed_count']:
            print(f"  [{label}] ✂ Startup trim: {rpt['trimmed_count']} pts (t<{trim_ms}ms)")

    # Step 2: IQR fence
    q1, q3 = np.percentile(a, 25), np.percentile(a, 75)
    iqr    = q3 - q1
    lo, hi = q1 - iqr_k*iqr, q3 + iqr_k*iqr
    rpt.update({'q1':round(float(q1),5), 'q3':round(float(q3),5),
                'iqr':round(float(iqr),5),
                'fence_lo':round(float(lo),5), 'fence_hi':round(float(hi),5)})

    spike_mask = (a < lo) | (a > hi)
    n_sp = int(np.sum(spike_mask))
    rpt['spike_count'] = n_sp

    if n_sp > 0:
        s_idx = np.where(spike_mask)[0]
        rpt['spike_times_ms']  = t[s_idx].tolist()
        rpt['spike_values_g']  = [round(v,4) for v in a[s_idx].tolist()]
        # replace
        if method == 'interpolate':
            a_c = a.copy().astype(float)
            for idx in s_idx:
                lft = [j for j in range(idx-1,-1,-1) if not spike_mask[j]]
                rgt = [j for j in range(idx+1,len(a)) if not spike_mask[j]]
                if lft and rgt: a_c[idx] = (a[lft[0]]+a[rgt[0]])/2
                elif lft:       a_c[idx] = a[lft[0]]
                elif rgt:       a_c[idx] = a[rgt[0]]
                else:           a_c[idx] = 0.0
            a = a_c
        elif method == 'median':
            a[spike_mask] = float(np.median(a[~spike_mask]))
        else:
            a[spike_mask] = 0.0
        print(f"  [{label}] 🔧 Spikes removed: {n_sp} pts  fence=[{lo:.3f}, {hi:.3f}]G  →  '{method}'")
    else:
        print(f"  [{label}] ✅ No spikes  fence=[{lo:.3f}, {hi:.3f}]G")

    rpt.update({'n_clean': len(a),
                'peak_after': round(float(np.max(np.abs(a))),5),
                'rms_after':  round(float(np.sqrt(np.mean(a**2))),5)})
    return t, a, rpt


# ── 6. Metrics / FFT / ISO ───────────────────────────────────
def compute_metrics(a, t):
    dt = float(np.median(np.diff(t))); fs = 1000./dt
    return {'rms':   float(np.sqrt(np.mean(a**2))),
            'peak':  float(np.max(np.abs(a))),
            'p2p':   float(np.max(a)-np.min(a)),
            'crest': float(np.max(np.abs(a))/np.sqrt(np.mean(a**2))) if np.mean(a**2)>0 else 0,
            'kurt':  float(kurtosis(a)),
            'skew':  float(skew(a)),
            'std':   float(np.std(a)),
            'fs':    fs, 'dt': dt,
            'n':     len(a),
            'dur':   float(t[-1]-t[0])}

ISO_ZONES = {'A':{'max':2.5,'label':'Newly Commissioned / Good'},
             'B':{'max':3.0,'label':'Unrestricted Operation'},
             'C':{'max':4.0,'label':'Restricted Operation — Monitor'},
             'D':{'max':9999,'label':'Damage Risk — Act Immediately'}}

def iso_zone(cf):
    for z,info in ISO_ZONES.items():
        if cf<=info['max']: return z,info
    return 'D',ISO_ZONES['D']

def compute_fft(a, fs):
    f = rfftfreq(len(a), d=1./fs)
    m = np.abs(rfft(a))*2./len(a)
    return f, m

def top_freqs(freq, mag, n=8, min_hz=5.):
    mask = freq >= min_hz
    ff,mm = freq[mask], mag[mask]
    idx = np.argsort(mm)[-n:][::-1]
    return [(float(ff[i]),float(mm[i])) for i in idx]
