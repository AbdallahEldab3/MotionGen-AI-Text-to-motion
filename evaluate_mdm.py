import os, sys, json, types, argparse, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   required=True)
    p.add_argument("--args_path",    required=True)
    p.add_argument("--dataset_path", default="dataset/HumanML3D")
    p.add_argument("--output_dir",   default="eval_results")
    p.add_argument("--num_samples",  type=int, default=50)
    p.add_argument("--num_reps",     type=int, default=3)
    p.add_argument("--guidance",     type=float, default=2.5)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def build_args_namespace(args_path):
    with open(args_path) as f:
        saved = json.load(f)
    defaults = dict(
        arch="trans_enc", emb_trans_dec=False, ff_size=1024,
        num_heads=4, dropout=0.1, activation="gelu",
        clip_version="ViT-B/32", cond_mask_prob=0.1,
        lambda_rcxyz=0.0, lambda_vel=0.0, lambda_fc=0.0,
        lambda_target_loc=0.0, unconstrained=False,
        noise_schedule="cosine", sigma_small=True,
        pred_len=0, context_len=0, pos_embed_max_len=5000,
        mask_frames=False, text_encoder_type="clip",
        multi_target_cond=False, multi_encoder_type="single",
        target_enc_layers=1, use_ema=False,
    )
    defaults.update(saved)
    return types.SimpleNamespace(**defaults)


class _MockDataset:
    num_actions = 1

class _MockDataLoader:
    dataset = _MockDataset()


def load_model_and_diffusion(model_path, args_path, device):
    sys.path.insert(0, os.getcwd())
    from utils.model_util import create_model_and_diffusion
    from utils.sampler_util import ClassifierFreeSampleModel

    args  = build_args_namespace(args_path)
    model, diffusion = create_model_and_diffusion(args, _MockDataLoader())

    raw = torch.load(model_path, map_location="cpu")
    if isinstance(raw, dict):
        sd = raw.get("model_avg") or raw.get("model") or raw
    else:
        sd = raw
    for k in ["sequence_pos_encoder.pe", "embed_timestep.sequence_pos_encoder.pe"]:
        sd.pop(k, None)
    model.load_state_dict(sd, strict=False)

    model = ClassifierFreeSampleModel(model)
    model.eval().to(device)
    return model, diffusion


def load_test_data(dataset_path, num_samples):
    with open(os.path.join(dataset_path, "test.txt")) as f:
        ids = [l.strip() for l in f if l.strip()][:num_samples]

    texts_dir = os.path.join(dataset_path, "texts")
    samples   = []
    for sid in ids:
        text_file = os.path.join(texts_dir, f"{sid}.txt")
        text = ""
        if os.path.exists(text_file):
            with open(text_file, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                text = random.choice(lines).split("#")[0].strip()
        samples.append({
            "id":   sid,
            "text": text,
            "path": os.path.join(dataset_path, "new_joint_vecs", f"{sid}.npy"),
        })
    return samples


def load_motion_joints(path, mean, std):
    vec   = np.load(path).astype(np.float32) * std + mean
    T     = vec.shape[0]
    root  = np.zeros((T, 1, 3), dtype=np.float32)
    root[:, 0, 1] = vec[:, 0]
    local = vec[:, 4:67].reshape(T, 21, 3)
    return np.concatenate([root, local], axis=1)


def compute_ade(pred, gt):
    return float(np.linalg.norm(pred - gt, axis=-1).mean())


def sample_motion(model, diffusion, gt_joints, text, guidance, num_reps, device):
    T = gt_joints.shape[0]
    best_ade, best_pred = float("inf"), None
    for _ in range(num_reps):
        noise = torch.randn(1, 263, 1, T, device=device)
        with torch.no_grad():
            sample = diffusion.p_sample_loop(
                model, noise.shape, noise=noise, clip_denoised=False, progress=False,
                model_kwargs={"y": {
                    "mask":    torch.ones(1, 1, 1, T, device=device),
                    "lengths": torch.tensor([T], device=device),
                    "text":    [text],
                    "tokens":  [""],
                    "scale":   torch.tensor([guidance], device=device),
                }},
            )
        vec    = sample[0, :, 0, :].permute(1, 0).cpu().numpy()
        root_p = np.zeros((T, 1, 3), dtype=np.float32)
        root_p[:, 0, 1] = vec[:, 0]
        pred   = np.concatenate([root_p, vec[:, 4:67].reshape(T, 21, 3)], axis=1)
        ade    = compute_ade(pred, gt_joints)
        if ade < best_ade:
            best_ade, best_pred = ade, pred
    return best_pred


def save_bar_chart(scores, path):
    mean_v = np.mean(scores)
    fig, ax = plt.subplots(figsize=(max(8, len(scores)//3), 5))
    ax.bar(range(len(scores)), scores, color="steelblue", edgecolor="white", lw=0.4)
    ax.axhline(mean_v, color="tomato", lw=1.8, ls="--", label=f"Mean = {mean_v:.4f}")
    ax.set(xlabel="Sample", ylabel="ADE", title="Per-Sample ADE")
    ax.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)

def save_line_chart(scores, path):
    cum = np.cumsum(scores) / np.arange(1, len(scores)+1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(scores)+1), scores, alpha=0.35, color="steelblue", lw=1, label="ADE per sample")
    ax.plot(range(1, len(scores)+1), cum,    color="tomato", lw=2, label="Cumulative mean")
    ax.set(xlabel="Samples", ylabel="ADE", title="ADE Convergence")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close(fig)

def save_box_plot(scores, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(scores, patch_artist=True, boxprops=dict(facecolor="steelblue", alpha=0.7))
    ax.set(ylabel="ADE", title="ADE Distribution")
    ax.set_xticks([1]); ax.set_xticklabels(["Test Set"])
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)

def save_text_log(samples, ade_scores, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'ID':<20} {'ADE':>8}  Prompt\n")
        f.write("-"*90 + "\n")
        for s, ade in zip(samples, ade_scores):
            f.write(f"{s['id']:<20} {ade:>8.4f}  {s['text'][:60]}\n")


def main():
    cfg = parse_args()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"
    elif cfg.device.startswith("cuda"):
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} ({p.total_memory/1e9:.1f} GB) | CUDA {torch.version.cuda} | PyTorch {torch.__version__}")

    mean = np.load(os.path.join(cfg.dataset_path, "Mean.npy"))
    std  = np.where((s := np.load(os.path.join(cfg.dataset_path, "Std.npy"))) == 0, 1.0, s)

    model, diffusion = load_model_and_diffusion(cfg.model_path, cfg.args_path, cfg.device)
    samples          = load_test_data(cfg.dataset_path, cfg.num_samples)

    ade_scores, evaluated, failed = [], [], []
    for s in tqdm(samples, desc="Evaluating"):
        try:
            gt   = load_motion_joints(s["path"], mean, std)
            pred = sample_motion(model, diffusion, gt, s["text"], cfg.guidance, cfg.num_reps, cfg.device)
            ade_scores.append(compute_ade(pred, gt))
            evaluated.append(s)
        except Exception as e:
            print(f"\nSkipping {s['id']}: {e}")
            failed.append(s["id"])

    if not ade_scores:
        print("No samples evaluated."); return

    print(f"\n{'='*48}")
    print(f"  Evaluated  : {len(ade_scores)}")
    print(f"  Mean ADE   : {np.mean(ade_scores):.4f}")
    print(f"  Median ADE : {np.median(ade_scores):.4f}")
    print(f"  Std        : {np.std(ade_scores):.4f}")
    print(f"  Min / Max  : {np.min(ade_scores):.4f} / {np.max(ade_scores):.4f}")
    print(f"{'='*48}")

    json.dump({
        "mean_ade": float(np.mean(ade_scores)), "median_ade": float(np.median(ade_scores)),
        "std_ade":  float(np.std(ade_scores)),  "min_ade":    float(np.min(ade_scores)),
        "max_ade":  float(np.max(ade_scores)),  "n_evaluated": len(ade_scores),
        "per_sample": [{"id": s["id"], "text": s["text"], "ade": a} for s, a in zip(evaluated, ade_scores)],
    }, open(os.path.join(cfg.output_dir, "ade_results.json"), "w", encoding="utf-8"), indent=2)

    save_bar_chart( ade_scores, os.path.join(cfg.output_dir, "ade_bar_chart.png"))
    save_line_chart(ade_scores, os.path.join(cfg.output_dir, "ade_line_chart.png"))
    save_box_plot(  ade_scores, os.path.join(cfg.output_dir, "ade_box_plot.png"))
    save_text_log(  evaluated,  ade_scores, os.path.join(cfg.output_dir, "ade_per_prompt.txt"))
    print(f"Results in: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
