"""Step-1 table from quality/*.json, bench_r*.log and model sizes."""
import json, re, os, glob, statistics as st
Q = "/Volumes/T5/cache/mllm/g16/quality"; L = "/Volumes/T5/cache/mllm/g16/logs"; M = "/Volumes/T5/cache/mllm/models"
names = ["8bit", "4bit", "mixed_3_6", "3bit", "mixed_2_6", "2bit"]
ref = json.load(open(f"{Q}/8bit.json"))
bench = {}
for f in sorted(glob.glob(f"{L}/bench_r*.log")):
    cur = None
    for line in open(f):
        if line.startswith("=== "): cur = line.split()[1]
        m = re.search(r"Averages: prompt_tps=([\d.]+), generation_tps=([\d.]+), peak_memory=([\d.]+)", line)
        if m and cur: bench.setdefault(cur, []).append(tuple(map(float, m.groups())))
def size(n):
    if n == "4bit":
        d = glob.glob("/Volumes/T5/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/*/")[0]
        idx = json.load(open(d + "model.safetensors.index.json"))
        # text weights only (the 4-bit checkpoint also carries the vision tower)
        import struct
        tot = 0
        for f in glob.glob(d + "*.safetensors"):
            with open(f, "rb") as fh:
                h = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
            tot += sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in h.items() if k.startswith("language_model."))
        return tot / 1e9
    return sum(os.path.getsize(f) for f in glob.glob(f"{M}/Qwen3.6-27B-{n}/*.safetensors")) / 1e9
print("| variant | weights GB | KL mean | KL p99 | top-1 | ppl | greedy prefix vs 8-bit (3 prompts) | prefill p512 tok/s | decode tok/s | peak GB |")
print("|---|---|---|---|---|---|---|---|---|---|")
for n in names:
    if not os.path.exists(f"{Q}/{n}.json"): continue
    r = json.load(open(f"{Q}/{n}.json"))
    pref = []
    for a, b in zip(r["greedy"], ref["greedy"]):
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        pref.append(k)
    b = bench.get(n, [])
    bs = lambda i: " / ".join(f"{x[i]:.1f}" for x in b) if b else "-"
    if n == "8bit":
        print(f"| {n} (reference) | {size(n):.2f} | 0 | 0 | 100% | {r.get('ppl', 0) or json.load(open(f'{Q}/4bit.json'))['ppl_ref']:.3f} | 64/64/64 | {bs(0)} | {bs(1)} | {bs(2)} |")
        continue
    print(f"| {n} | {size(n):.2f} | {r['kl_mean']:.4f} | {r['kl_p99']:.3f} | {100 * r['top1']:.1f}% | {r['ppl']:.3f} | "
          f"{'/'.join(map(str, pref))} | {bs(0)} | {bs(1)} | {bs(2)} |")
