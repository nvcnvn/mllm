"""ModelSpec: the shapes every other component is sized from. Plain data, read from config.json."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    path: Path
    kinds: tuple          # per layer: "full" | "linear"
    hidden: int
    vocab: int
    n_heads: int          # full attention
    n_kv_heads: int
    head_dim: int
    lin_k_heads: int      # Gated DeltaNet
    lin_v_heads: int
    lin_k_dim: int
    lin_v_dim: int
    conv_kernel: int

    @property
    def n_layers(self):
        return len(self.kinds)

    @property
    def full_layers(self):
        return [i for i, k in enumerate(self.kinds) if k == "full"]

    @property
    def linear_layers(self):
        return [i for i, k in enumerate(self.kinds) if k == "linear"]

    @property
    def conv_dim(self):
        return 2 * self.lin_k_heads * self.lin_k_dim + self.lin_v_heads * self.lin_v_dim


def resolve(model):
    """Repo id or local dir -> local dir. Downloads go to $HF_HOME (keep it on /Volumes/T5)."""
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model))


def load(model) -> ModelSpec:
    path = resolve(model)
    c = json.loads((path / "config.json").read_text())
    t = c.get("text_config", c)
    if t["model_type"] not in ("qwen3_5", "qwen3_5_text"):
        raise ValueError(f"unsupported model_type {t['model_type']}")  # ponytail: one family until a second one runs
    kinds = tuple("full" if k == "full_attention" else "linear" for k in t["layer_types"])
    return ModelSpec(
        path=path, kinds=kinds, hidden=t["hidden_size"], vocab=t["vocab_size"],
        n_heads=t["num_attention_heads"], n_kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"],
        lin_k_heads=t["linear_num_key_heads"], lin_v_heads=t["linear_num_value_heads"],
        lin_k_dim=t["linear_key_head_dim"], lin_v_dim=t["linear_value_head_dim"],
        conv_kernel=t["linear_conv_kernel_dim"],
    )
