"""
Phase 1 pilot grid: 9 runs over (V, D).

Goal: validate the BPB metric, confirm that V/D affects loss at fixed compute,
and check that runs finish in a reasonable time on a single A100.

Architecture: small (n_layer=6, n_head=8) so the runs are quick.
Compute budget: 1e17 FLOPs each (small enough that 9 runs fit on 8xA100 in
roughly 1 hour wall time, large enough that loss curves are well-formed).
"""

GRID = []
VOCAB_SIZES = [1024, 8192, 32768]
HIDDEN_DIMS = [256, 512, 1024]

for V in VOCAB_SIZES:
    for D in HIDDEN_DIMS:
        # n_head must divide n_embd
        n_head = 8 if D >= 256 else 4
        GRID.append({
            "V": V,
            "D": D,
            "n_layer": 6,
            "n_head": n_head,
            "block_size": 1024,
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "compute_budget_flops": 1e17,
            "learning_rate": 6e-4,
            "warmup_iters": 100,
        })

if __name__ == "__main__":
    print(f"Total runs: {len(GRID)}")
    for r in GRID:
        print(f"  V={r['V']:>5}  D={r['D']:>5}  L={r['n_layer']}  V/D={r['V']/r['D']:.2f}")
