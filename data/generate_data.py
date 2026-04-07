"""
Generates synthetic transactions with a graduated risk spectrum.

Account pools:
  - Pure normal     (ACC/EMP/MER)  → fraud ratio ~0.00  → score LOW
  - Suspicious      (SUS)          → normal but odd behaviour → score LOW-MEDIUM
  - Lightly mixed   (MIX_L)        → ~10% fraud touches  → score MEDIUM-LOW
  - Moderately mixed(MIX_M)        → ~35% fraud touches  → score MEDIUM
  - Heavily mixed   (MIX_H)        → ~65% fraud touches  → score MEDIUM-HIGH
  - Core fraud      (MUL/SRC/LAY…) → ~100% fraud         → score HIGH-CRITICAL

This creates a natural 0→1 continuum instead of binary 0/1.
"""
import json, sys, os, random, uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation.normal_patterns import generate_mixed_normal
from src.simulation.fraud_patterns  import generate_mixed_fraud

BASE_TS = 1_700_000_000.0
random.seed(7)

CHANNELS = ["ACH", "WIRE", "CARD", "P2P", "ATM"]

def txn(src, dst, amount, channel, ts, label):
    return {"txn_id": str(uuid.uuid4()), "from_account": src, "to_account": dst,
            "amount": round(amount, 2), "channel": channel,
            "timestamp": float(ts), "label": label}

# ── Base datasets ─────────────────────────────────────────────────────────────
normal_base = generate_mixed_normal(total=500, seed=42)
fraud_base  = generate_mixed_fraud(seed=99, start_ts=BASE_TS + 3600)

# Account pools to use as targets in mixed transactions
normal_pool = list({t["from_account"] for t in normal_base if t["from_account"].startswith("ACC")}
                   | {t["to_account"]   for t in normal_base if t["to_account"].startswith("ACC")})
fraud_pool  = list({t["from_account"] for t in fraud_base}
                   | {t["to_account"]   for t in fraud_base})
random.shuffle(normal_pool); random.shuffle(fraud_pool)

ts = BASE_TS + 10_000
extra_txns = []

# ── Suspicious-but-clean accounts (SUS) ─────────────────────────────────────
# High velocity, multiple channels — but all normal transactions.
# GNN should score these MEDIUM-LOW because of unusual topology.
for i in range(15):
    acc = f"SUS{10000+i}"
    targets = random.sample(normal_pool[:80], random.randint(10, 20))
    chs = random.sample(CHANNELS, random.randint(3, 5))  # many channels
    for tgt in targets:
        ts += random.uniform(2, 20)   # fast-ish
        extra_txns.append(txn(acc, tgt, random.uniform(300, 1200),
                               random.choice(chs), ts, "normal"))

# ── Lightly mixed accounts (MIX_L) — ~10% fraud involvement ─────────────────
for i in range(30):
    acc = f"MXL{10000+i}"
    n_normal = random.randint(15, 25)
    n_fraud  = random.randint(1, 3)            # 1–3 fraud touches
    for _ in range(n_normal):
        ts += random.uniform(60, 600)
        tgt = random.choice(normal_pool[:100])
        extra_txns.append(txn(acc, tgt, random.uniform(50, 500),
                               random.choice(["ACH","CARD","P2P"]), ts, "normal"))
    for _ in range(n_fraud):
        ts += random.uniform(30, 300)
        tgt = random.choice(fraud_pool[:30])
        extra_txns.append(txn(acc, tgt, random.uniform(200, 1500),
                               random.choice(CHANNELS), ts, "fraud"))

# ── Moderately mixed accounts (MIX_M) — ~35% fraud involvement ──────────────
for i in range(25):
    acc = f"MXM{10000+i}"
    n_normal = random.randint(8, 14)
    n_fraud  = random.randint(4, 8)
    for _ in range(n_normal):
        ts += random.uniform(30, 400)
        tgt = random.choice(normal_pool[:80])
        extra_txns.append(txn(acc, tgt, random.uniform(100, 800),
                               random.choice(["ACH","CARD"]), ts, "normal"))
    for _ in range(n_fraud):
        ts += random.uniform(10, 200)
        tgt = random.choice(fraud_pool[:40])
        extra_txns.append(txn(acc, tgt, random.uniform(400, 3000),
                               random.choice(CHANNELS), ts, "fraud"))

# ── Heavily mixed accounts (MIX_H) — ~65% fraud involvement ─────────────────
for i in range(20):
    acc = f"MXH{10000+i}"
    n_normal = random.randint(3, 7)
    n_fraud  = random.randint(8, 15)
    for _ in range(n_normal):
        ts += random.uniform(20, 200)
        tgt = random.choice(normal_pool[:60])
        extra_txns.append(txn(acc, tgt, random.uniform(50, 400),
                               random.choice(["ACH","CARD"]), ts, "normal"))
    for _ in range(n_fraud):
        ts += random.uniform(5, 100)
        tgt = random.choice(fraud_pool[:50])
        extra_txns.append(txn(acc, tgt, random.uniform(500, 5000),
                               random.choice(CHANNELS), ts, "fraud"))

# ── Write files ───────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "synthetic")
os.makedirs(out_dir, exist_ok=True)

all_normal = normal_base + [t for t in extra_txns if t["label"] == "normal"]
all_fraud  = fraud_base  + [t for t in extra_txns if t["label"] == "fraud"]
random.shuffle(all_normal); random.shuffle(all_fraud)

with open(os.path.join(out_dir, "normal_transactions.json"), "w") as f:
    json.dump(all_normal, f, indent=2)
with open(os.path.join(out_dir, "fraud_transactions.json"), "w") as f:
    json.dump(all_fraud, f, indent=2)

print(f"Normal transactions : {len(all_normal)}")
print(f"Fraud  transactions : {len(all_fraud)}")
print(f"Mixed accounts created: SUS=15  MXL=30  MXM=25  MXH=20")
