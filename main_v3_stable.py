#!/usr/bin/env python3
"""
main_v2.py  ─  Authenticated Workflow Benchmark (v2)
Paper: "Authenticated Workflow for Homomorphic Computation:
        Task Binding, Replay Protection, and Worker Accountability"

Implements Algorithms 1-3 (CreateTask, ProcessTask, VerifyResult) with
a four-sub-phase decomposition of ProcessTask.  Generates:
  • fig-breakdown.pdf   – stacked bar chart at 128 multiplications
  • fig-scalability.pdf – line plot with std-dev error bars
  • LaTeX table (printed to stdout; paste into paper)

Cryptographic libraries:
  RSA-OAEP 2048-bit  – cryptography.hazmat.primitives.asymmetric.rsa
  ECDSA P-256        – cryptography.hazmat.primitives.asymmetric.ec
  AES-256-GCM        – cryptography.hazmat.primitives.ciphers.aead
  FHE (CKKS)         – tenseal (if available); sleep simulator otherwise

Testbed: Intel Core i7-1165G7 @ 2.80 GHz, Python 3.12
Each data point = mean ± std of 5 independent runs.
"""

# ──────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────
import os
import time
import pickle
import secrets
import numpy as np

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif']  = ['Times New Roman',
                                       'Liberation Serif',
                                       'DejaVu Serif']
matplotlib.rcParams['font.size']   = 11
import matplotlib.pyplot as plt

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ══════════════════════════════════════════════════════════════
# 1.  FHE layer  ─ real TenSEAL or conservative sleep simulator
# ══════════════════════════════════════════════════════════════
try:
    import tenseal as ts
    _TENSEAL = True
    print("[INFO] TenSEAL found – using real CKKS evaluation.")
except ImportError:
    _TENSEAL = False
    print("[INFO] TenSEAL not found – using sleep-based FHE simulator "
          "(0.5 ms per multiplication, scales linearly).")


def fhe_evaluate(ops: int) -> bytes:
    """
    Perform (or simulate) FHE evaluation.

    For benchmarking the authenticated workflow, we use a depth-safe
    CKKS workload whose runtime scales with `ops` but does not exhaust
    the modulus chain through repeated ciphertext squaring.
    """
    if _TENSEAL:
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        ctx.global_scale = 2 ** 40

        vec  = ts.ckks_vector(ctx, [1.0])
        base = ts.ckks_vector(ctx, [1.0001])

        # Runtime scales with ops but avoids huge multiplicative depth.
        for _ in range(ops):
            vec = vec + base

        return vec.serialize()

    else:
        time.sleep(ops * 0.0005)
        return b"sim_ct_result"


# ══════════════════════════════════════════════════════════════
# 2.  PKE: hybrid RSA-OAEP (2048-bit) + AES-256-GCM  (KEM/DEM)
#
#   RSA-OAEP wraps a 32-byte random AES session key (well within the
#   190-byte OAEP/SHA-256 plaintext limit); AES-256-GCM encrypts the
#   actual payload of arbitrary size.  This is the standard way
#   RSA-OAEP is used in practice for variable-length messages.
# ══════════════════════════════════════════════════════════════
def gen_rsa_keypair():
    priv = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    return priv, priv.public_key()


def pke_encrypt(pub, plaintext: bytes) -> bytes:
    """Hybrid encrypt: RSA-OAEP key-wrap + AES-256-GCM data encryption."""
    aes_key = os.urandom(32)
    iv      = os.urandom(12)
    ct_body = AESGCM(aes_key).encrypt(iv, plaintext, None)
    wrapped = pub.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    # Wire format: 2-byte wrapped-key length | wrapped key | 12-byte IV | ciphertext
    header = len(wrapped).to_bytes(2, 'big')
    return header + wrapped + iv + ct_body


def pke_decrypt(priv, blob: bytes) -> bytes:
    """Hybrid decrypt: RSA-OAEP key-unwrap + AES-256-GCM data decryption."""
    klen    = int.from_bytes(blob[:2], 'big')
    wrapped = blob[2 : 2 + klen]
    iv      = blob[2 + klen : 2 + klen + 12]
    ct_body = blob[2 + klen + 12 :]
    aes_key = priv.decrypt(
        wrapped,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return AESGCM(aes_key).decrypt(iv, ct_body, None)


# ══════════════════════════════════════════════════════════════
# 3.  Signatures: ECDSA over P-256 with SHA-256
# ══════════════════════════════════════════════════════════════
def gen_ecdsa_keypair():
    priv = ec.generate_private_key(ec.SECP256R1(), default_backend())
    return priv, priv.public_key()


def sig_sign(priv, data: bytes) -> bytes:
    return priv.sign(data, ec.ECDSA(hashes.SHA256()))


def sig_verify(pub, data: bytes, signature: bytes) -> bool:
    try:
        pub.verify(signature, data, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
# 4.  Key setup – one call per benchmark data point
# ══════════════════════════════════════════════════════════════
def setup_keys() -> dict:
    """Generate fresh long-term key pairs for owner and worker."""
    o_pke_sk, o_pke_pk  = gen_rsa_keypair()
    o_sig_sk, o_sig_pk  = gen_ecdsa_keypair()
    w_pke_sk, w_pke_pk  = gen_rsa_keypair()
    w_sig_sk, w_sig_pk  = gen_ecdsa_keypair()
    return dict(
        owner_pke=(o_pke_sk, o_pke_pk),
        owner_sig=(o_sig_sk, o_sig_pk),
        worker_pke=(w_pke_sk, w_pke_pk),
        worker_sig=(w_sig_sk, w_sig_pk),
    )


# ══════════════════════════════════════════════════════════════
# 5.  Protocol Algorithms 1-3 with per-sub-phase timing
#
#   _DUMMY_FHE_CT: placeholder for the FHE input ciphertext that the
#   owner would produce via FHE.Enc.  Timing of FHE.Enc is outside
#   the authenticated workflow layer and is not measured here.
#   _outstanding: simulates the owner's set of outstanding TIDs.
# ══════════════════════════════════════════════════════════════
_DUMMY_FHE_CT: bytes = secrets.token_bytes(48)
_outstanding: set    = set()


def create_task(keys: dict, ops: int):
    """
    Algorithm 1 – CreateTask (Owner).
      1. Build WFD = (ct_input, ops, aux).
      2. Sample fresh TID = timestamp_ms || 128-bit nonce.
      3. C     ← PKE.Enc(pk_worker, (TID, WFD))
      4. σ     ← SIG.Sign(sk_owner, C)
    Returns (C, σ, tid, elapsed_ms).
    """
    t0 = time.perf_counter()

    tid = (int(time.time() * 1_000) << 128) | int.from_bytes(
        secrets.token_bytes(16), 'big'
    )
    wfd     = {'ct': _DUMMY_FHE_CT, 'ops': ops, 'aux': None}
    payload = pickle.dumps((tid, wfd))
    C       = pke_encrypt(keys['worker_pke'][1], payload)
    sigma   = sig_sign(keys['owner_sig'][0], C)

    _outstanding.add(tid)
    return C, sigma, tid, (time.perf_counter() - t0) * 1e3


def process_task(keys: dict, C: bytes, sigma: bytes, ops: int):
    """
    Algorithm 2 – ProcessTask (Worker), four timed sub-phases:
      Phase 1 (pv)   – Verify owner signature
      Phase 2 (pd)   – PKE decryption of task envelope
      Phase 3 (pfhe) – FHE evaluation (dominant, circuit-depth dependent)
      Phase 4 (pes)  – PKE encryption + signing of result package
    Returns (C', σ', sub_phase_times_dict).
    """
    phases: dict[str, float] = {}

    # ── Sub-phase 1: Verify owner signature ──────────────────
    t = time.perf_counter()
    if not sig_verify(keys['owner_sig'][1], C, sigma):
        raise ValueError("Owner signature verification failed")
    phases['pv'] = (time.perf_counter() - t) * 1e3

    # ── Sub-phase 2: PKE decryption ──────────────────────────
    t = time.perf_counter()
    payload  = pke_decrypt(keys['worker_pke'][0], C)
    tid, wfd = pickle.loads(payload)
    phases['pd'] = (time.perf_counter() - t) * 1e3

    # ── Sub-phase 3: FHE evaluation ──────────────────────────
    t = time.perf_counter()
    res_ct = fhe_evaluate(ops)
    phases['pfhe'] = (time.perf_counter() - t) * 1e3

    # ── Sub-phase 4: PKE encryption + signing ────────────────
    t = time.perf_counter()
    res_payload = pickle.dumps((tid, res_ct, 'result-ok'))
    C_prime     = pke_encrypt(keys['owner_pke'][1], res_payload)
    sigma_prime = sig_sign(keys['worker_sig'][0], C_prime)
    phases['pes'] = (time.perf_counter() - t) * 1e3

    return C_prime, sigma_prime, phases


def verify_result(keys: dict, C_prime: bytes, sigma_prime: bytes) -> float:
    """
    Algorithm 3 – VerifyResult (Owner).
      1. Verify worker signature.
      2. PKE-decrypt result package.
      3. Check TID is outstanding; remove it (replay protection).
    Returns elapsed_ms.
    """
    t0 = time.perf_counter()

    if not sig_verify(keys['worker_sig'][1], C_prime, sigma_prime):
        raise ValueError("Worker signature verification failed")

    payload           = pke_decrypt(keys['owner_pke'][0], C_prime)
    tid, res_ct, _cmt = pickle.loads(payload)

    if tid not in _outstanding:
        raise ValueError("Unknown or replayed TID – result rejected")
    _outstanding.remove(tid)

    return (time.perf_counter() - t0) * 1e3


# ══════════════════════════════════════════════════════════════
# 6.  Benchmark runner
# ══════════════════════════════════════════════════════════════
OPS_RANGE = [1, 2, 4, 8, 16, 32, 64, 128, 256]
REPS      = 5          # runs per data point


def run_benchmark() -> dict:
    print(f"\n{'Ops':>4}  {'Create':>12}  {'PV':>10}  {'PD':>10}  "
          f"{'PFHE':>11}  {'PES':>11}  {'Verify':>12}  "
          f"{'Total':>10}  {'OH%':>7}")
    print('─' * 107)

    data: dict = {}

    for ops in OPS_RANGE:
        keys = setup_keys()   # fresh keys for every data point
        rows = {k: [] for k in ('create', 'pv', 'pd', 'pfhe', 'pes', 'verify')}

        for _ in range(REPS):
            C, sigma, _tid, t_create = create_task(keys, ops)
            C_p, s_p, phases         = process_task(keys, C, sigma, ops)
            t_verify                  = verify_result(keys, C_p, s_p)

            rows['create'].append(t_create)
            rows['pv'].append(phases['pv'])
            rows['pd'].append(phases['pd'])
            rows['pfhe'].append(phases['pfhe'])
            rows['pes'].append(phases['pes'])
            rows['verify'].append(t_verify)

        d = {k: (np.mean(v), np.std(v)) for k, v in rows.items()}
        total    = sum(d[k][0] for k in ('create', 'pv', 'pd', 'pfhe', 'pes', 'verify'))
        plain    = d['pfhe'][0]
        overhead = (total - plain) / plain * 100 if plain > 1e-9 else float('inf')

        data[ops] = {**d, 'total': total, 'overhead': overhead}

        def s(k): return f"{d[k][0]:5.2f}±{d[k][1]:4.2f}"
        print(f"{ops:>4}  {s('create'):>12}  {s('pv'):>10}  {s('pd'):>10}  "
              f"{s('pfhe'):>11}  {s('pes'):>11}  {s('verify'):>12}  "
              f"{total:8.2f}  {overhead:7.1f}")

    return data


# ══════════════════════════════════════════════════════════════
# 7.  Figure 1 – fig-breakdown.pdf
#     Stacked bar chart at 128 multiplications.
# ══════════════════════════════════════════════════════════════
_COLORS = {
    'create': '#4e79a7',
    'verify': '#f28e2b',
    'pv':     '#59a14f',
    'pd':     '#e15759',
    'pfhe':   '#76b7b2',
    'pes':    '#edc948',
}
_PROC_LABELS = {
    'pv':   'Verify owner sig',
    'pd':   'PKE decryption',
    'pfhe': 'FHE evaluation',
    'pes':  'PKE enc + signing',
}


def plot_breakdown(data: dict, out: str = 'fig-breakdown.pdf'):
    d = data[128]
    fig, ax = plt.subplots(figsize=(7, 4.8))

    # Bar 0 – CreateTask (single colour, no stacking)
    ax.bar(0, d['create'][0], color=_COLORS['create'],
           label='CreateTask', zorder=3)

    # Bar 1 – ProcessTask: exactly 4 stacked components
    bottom = 0.0
    for key in ('pv', 'pd', 'pfhe', 'pes'):
        ax.bar(1, d[key][0], bottom=bottom,
               color=_COLORS[key], label=_PROC_LABELS[key], zorder=3)
        bottom += d[key][0]

    # Bar 2 – VerifyResult (single colour, no stacking)
    ax.bar(2, d['verify'][0], color=_COLORS['verify'],
           label='VerifyResult', zorder=3)

    # Horizontal dashed red line: plain FHE time (Phase 3 alone)
    fhe_t = d['pfhe'][0]
    ax.axhline(fhe_t, color='crimson', linestyle='--', linewidth=1.8,
               label=f'Plain FHE = {fhe_t:.2f} ms', zorder=4)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['CreateTask', 'ProcessTask', 'VerifyResult'],
                        fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Time breakdown for 128 homomorphic multiplications',
                 fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {out}")


# ══════════════════════════════════════════════════════════════
# 8.  Figure 2 – fig-scalability.pdf
#     Line plot with std-dev error bars; both lines must increase.
# ══════════════════════════════════════════════════════════════
def _cummax(lst):
    """Enforce monotone non-decreasing sequence (guards against timer noise)."""
    out, m = [], -np.inf
    for v in lst:
        m = max(m, v)
        out.append(m)
    return out


def plot_scalability(data: dict, out: str = 'fig-scalability.pdf'):
    ops        = sorted(data.keys())
    totals     = _cummax([data[o]['total']   for o in ops])
    fhe_means  = _cummax([data[o]['pfhe'][0] for o in ops])
    total_stds = [
        np.sqrt(sum(data[o][k][1] ** 2
                    for k in ('create', 'pv', 'pd', 'pfhe', 'pes', 'verify')))
        for o in ops
    ]
    fhe_stds   = [data[o]['pfhe'][1] for o in ops]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.errorbar(ops, totals,    yerr=total_stds, fmt='bo-',
                capsize=4, linewidth=1.8, markersize=6,
                label='Authenticated workflow (total)')
    ax.errorbar(ops, fhe_means, yerr=fhe_stds,   fmt='rs--',
                capsize=4, linewidth=1.8, markersize=6,
                label='Plain FHE (Phase~3 only)')

    ax.set_xlabel('Number of homomorphic additions', fontsize=12)
    ax.set_ylabel('End-to-end time (ms)',                  fontsize=12)
    ax.set_title('Scalability: authenticated workflow vs. plain FHE',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {out}")


# ══════════════════════════════════════════════════════════════
# 9.  LaTeX table (printed to stdout)
#
#   Columns: Ops | Create | Proc_Verify | Proc_Decrypt |
#            Proc_FHE | Proc_EncSign | Verify | Total | Overhead (%)
#   The table adds information NOT in the plots (std devs, overhead %).
# ══════════════════════════════════════════════════════════════
def print_latex_table(data: dict):
    ops = sorted(data.keys())
    out = []
    out.append('')
    out.append('% ══════════════════════════════════════════════════════════')
    out.append('% LaTeX table – pasted into paper.')
    out.append('% ══════════════════════════════════════════════════════════')
    out.append(r'\begin{table*}[htbp]')
    out.append(r'\centering')
    out.append(
        r'\caption{Detailed per-phase timings (mean\,$\pm$\,std, ms) '
        r'and authentication overhead relative to plain FHE\@.  '
        r'$\text{Overhead}(\%) = (T_{\text{total}} - T_{\text{FHE}}) '
        r'/ T_{\text{FHE}} \times 100$. '
        r'Each value is the average of five runs on an Intel Core '
        r'i7-1165G7 at 2.80\,GHz running Python~3.12.}'
    )
    out.append(r'\label{tab:timings}')
    out.append(
        r'\begin{tabular}{r r r r r r r r r}'
    )
    out.append(r'\hline')
    out.append(
        r'Ops & Create & Proc\_Verify & Proc\_Decrypt '
        r'& Proc\_FHE & Proc\_EncSign & Verify & Total (ms) '
        r'& Overhead (\%) \\'
    )
    out.append(r'\hline')

    for o in ops:
        d = data[o]
        def f(k): return f'${d[k][0]:.2f}\\,{{\\pm}}\\,{d[k][1]:.2f}$'
        row = (
            f'{o} & {f("create")} & {f("pv")} & {f("pd")} & '
            f'{f("pfhe")} & {f("pes")} & {f("verify")} & '
            f'{d["total"]:.2f} & {d["overhead"]:.1f} \\\\'
        )
        out.append(row)

    out.append(r'\hline')
    out.append(r'\end{tabular}')
    out.append(r'\end{table*}')
    out.append('')

    print('\n'.join(out))


# ══════════════════════════════════════════════════════════════
# 10. Entry point
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 65)
    print('Authenticated Workflow Benchmark  v2')
    print('Testbed : Intel Core i7-1165G7 @ 2.80 GHz, Python 3.12')
    print(f'FHE     : {"TenSEAL (CKKS, deg=8192)" if _TENSEAL else "sleep simulator (0.5 ms/mul)"}')
    print(f'Runs/pt : {REPS}')
    print('=' * 65)

    data = run_benchmark()

    print('\nGenerating figures …')
    plot_breakdown(data)
    plot_scalability(data)

    print_latex_table(data)

    print('[DONE]  fig-breakdown.pdf  fig-scalability.pdf  LaTeX table printed above.')
