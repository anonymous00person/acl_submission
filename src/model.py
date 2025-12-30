# src/model.py
# -*- coding: utf-8 -*-

import os
from typing import Optional

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from .graph_features import FEATURE_DIM, FEATURE_KEYS


def build_bnb_config(bf16: bool) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
    )


def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


class Qwen2p5CoderQLoRAWithGraph(nn.Module):
    """
    Wrapper = (Qwen2.5-Coder backbone w/ LoRA) + (graph feature MLP) + (fusion head)

    Train time: backbone = get_peft_model(base, lora_config)
    Eval time : backbone = PeftModel.from_pretrained(base, lora_adapter_dir)
    """

    def __init__(
        self,
        model_name: str,
        bnb_config: BitsAndBytesConfig,
        lora_config: LoraConfig,
        graph_feat_dim: int,
        bf16: bool,
        dropout: float = 0.1,
        backbone: Optional[nn.Module] = None,   # allow passing a ready backbone (eval)
        device: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.bf16 = bf16
        self.graph_feat_dim = int(graph_feat_dim)
        self.dropout = float(dropout)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        if backbone is None:
            # ---- build base model (4-bit) ----
            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map={"": 0} if device.startswith("cuda") else None,
                torch_dtype=torch.bfloat16 if bf16 else torch.float16,
                trust_remote_code=True,
            )
            base = prepare_model_for_kbit_training(base)

            # enable checkpointing on backbone (Trainer may call wrapper method too)
            if hasattr(base, "gradient_checkpointing_enable"):
                base.gradient_checkpointing_enable()
            if hasattr(base.config, "use_cache"):
                base.config.use_cache = False

            # ---- inject LoRA (trainable) ----
            self.backbone = get_peft_model(base, lora_config)
        else:
            self.backbone = backbone

        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            raise ValueError("Cannot infer hidden_size from backbone config.")

        # ---- symbolic branch ----
        self.graph_norm = nn.LayerNorm(self.graph_feat_dim)
        self.graph_mlp = nn.Sequential(
            nn.Linear(self.graph_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # ---- fusion head ----
        self.fuse = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden + 64, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

        # IMPORTANT: do NOT call self.to(device) on quantized backbone.
        # Only move graph modules.
        self._move_graph_modules(device)

    # ----------------- device helpers -----------------
    def _move_graph_modules(self, device: str):
        self.graph_norm.to(device)
        self.graph_mlp.to(device)
        self.fuse.to(device)

    @property
    def graph_device(self) -> torch.device:
        return next(self.graph_norm.parameters()).device

    # ----------------- Trainer hooks -----------------
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            try:
                if gradient_checkpointing_kwargs is not None:
                    self.backbone.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                    )
                else:
                    self.backbone.gradient_checkpointing_enable()
            except TypeError:
                self.backbone.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    # ----------------- forward -----------------
    def forward(self, input_ids, attention_mask, graph_feats, labels=None, pos_weight: Optional[torch.Tensor] = None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = out.hidden_states[-1]     # [B,L,H]
        cls = last_hidden[:, -1, :]            # last token repr [B,H]

        # ensure graph feats on same device as graph modules
        if graph_feats.device != self.graph_device:
            graph_feats = graph_feats.to(self.graph_device)

        g = self.graph_norm(graph_feats)
        g = self.graph_mlp(g)

        fused = torch.cat([cls.to(self.graph_device), g], dim=-1)
        logits = self.fuse(fused).squeeze(-1)

        loss = None
        if labels is not None:
            labels = labels.view(-1).to(logits.device)
            if pos_weight is not None:
                pos_weight = pos_weight.to(logits.device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    # ----------------- save/load artifacts -----------------
    def save_artifacts(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        # save LoRA adapter
        adapter_dir = os.path.join(out_dir, "lora_adapter")
        self.backbone.save_pretrained(adapter_dir)

        # save head weights
        head = {
            "graph_norm": self.graph_norm.state_dict(),
            "graph_mlp": self.graph_mlp.state_dict(),
            "fuse": self.fuse.state_dict(),
            "feature_dim": FEATURE_DIM,
            "feature_keys": FEATURE_KEYS,
        }
        torch.save(head, os.path.join(out_dir, "fusion_head.pt"))

    def load_fusion_head(self, artifact_dir: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        head_path = os.path.join(artifact_dir, "fusion_head.pt")
        head = torch.load(head_path, map_location="cpu")

        self.graph_norm.load_state_dict(head["graph_norm"])
        self.graph_mlp.load_state_dict(head["graph_mlp"])
        self.fuse.load_state_dict(head["fuse"])

        self._move_graph_modules(device)

    @classmethod
    def from_artifacts(
        cls,
        model_name: str,
        artifact_dir: str,
        bf16: bool,
        graph_feat_dim: int,
        dropout: float,
        lora_cfg: dict,
        device: Optional[str] = None,
    ):
        """
        Build inference model = base (4-bit) + load LoRA adapter + load fusion_head.pt
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        bnb = build_bnb_config(bf16=bf16)
        lora = build_lora_config(
            r=int(lora_cfg.get("r", 16)),
            alpha=int(lora_cfg.get("alpha", 32)),
            dropout=float(lora_cfg.get("dropout", 0.05)),
        )

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map={"": 0} if device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            trust_remote_code=True,
        )
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = False

        adapter_dir = os.path.join(artifact_dir, "lora_adapter")
        backbone = PeftModel.from_pretrained(
            base,
            adapter_dir,
            is_trainable=False,
        )

        model = cls(
            model_name=model_name,
            bnb_config=bnb,
            lora_config=lora,
            graph_feat_dim=graph_feat_dim,
            bf16=bf16,
            dropout=dropout,
            backbone=backbone,
            device=device,
        )
        model.load_fusion_head(artifact_dir, device=device)
        model.eval()
        return model
