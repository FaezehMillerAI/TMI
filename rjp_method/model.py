import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        base = resnet18(weights="DEFAULT")
        in_feats = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Linear(in_feats, embed_dim)

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.proj(feat)
        return F.normalize(emb, dim=-1)


class AdaptiveFusionDecoder(nn.Module):
    """
    RAFS-inspired lightweight decoder.
    At each step, a scalar gate mixes image-conditioned and retrieval-conditioned context.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pad_idx: int,
        use_adaptive_fusion: bool = True,
        fixed_fusion_alpha: float = 0.5,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.use_adaptive_fusion = use_adaptive_fusion
        self.fixed_fusion_alpha = fixed_fusion_alpha
        self.tok_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim + embed_dim, hidden_dim, batch_first=True)
        self.init_h = nn.Linear(embed_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim + embed_dim * 2, 1)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_emb, retrieved_emb, tokens_in):
        tok_emb = self.tok_embed(tokens_in)
        bsz, seq_len, _ = tok_emb.shape

        h = torch.tanh(self.init_h(image_emb)).unsqueeze(0)
        logits = []

        for t in range(seq_len):
            step_tok = tok_emb[:, t, :]
            step_h = h.squeeze(0)
            if self.use_adaptive_fusion:
                alpha = torch.sigmoid(
                    self.gate(torch.cat([step_h, image_emb, retrieved_emb], dim=-1))
                )
            else:
                alpha = torch.full(
                    (image_emb.size(0), 1),
                    fill_value=self.fixed_fusion_alpha,
                    dtype=image_emb.dtype,
                    device=image_emb.device,
                )
            fused = alpha * image_emb + (1.0 - alpha) * retrieved_emb
            gru_in = torch.cat([step_tok, fused], dim=-1).unsqueeze(1)
            out, h = self.gru(gru_in, h)
            logits.append(self.out(out.squeeze(1)))

        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def greedy_decode(self, image_emb, retrieved_emb, bos_idx: int, eos_idx: int, max_len: int):
        bsz = image_emb.size(0)
        h = torch.tanh(self.init_h(image_emb)).unsqueeze(0)
        token = torch.full((bsz,), bos_idx, dtype=torch.long, device=image_emb.device)
        outputs = []

        for _ in range(max_len):
            step_tok = self.tok_embed(token)
            step_h = h.squeeze(0)
            if self.use_adaptive_fusion:
                alpha = torch.sigmoid(
                    self.gate(torch.cat([step_h, image_emb, retrieved_emb], dim=-1))
                )
            else:
                alpha = torch.full(
                    (image_emb.size(0), 1),
                    fill_value=self.fixed_fusion_alpha,
                    dtype=image_emb.dtype,
                    device=image_emb.device,
                )
            fused = alpha * image_emb + (1.0 - alpha) * retrieved_emb
            gru_in = torch.cat([step_tok, fused], dim=-1).unsqueeze(1)
            out, h = self.gru(gru_in, h)
            logit = self.out(out.squeeze(1))
            token = logit.argmax(dim=-1)
            outputs.append(token)

        seq = torch.stack(outputs, dim=1)
        for i in range(bsz):
            eos_pos = (seq[i] == eos_idx).nonzero(as_tuple=False)
            if len(eos_pos) > 0:
                first = eos_pos[0, 0].item()
                seq[i, first + 1 :] = eos_idx
        return seq


class ResidualAlignmentModule(nn.Module):
    """
    RAM: align visual embedding to textual manifold using a residual adapter.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gate = nn.Linear(embed_dim * 2, 1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image_emb, retrieved_emb):
        residual = self.adapter(
            torch.cat([image_emb, retrieved_emb, image_emb - retrieved_emb], dim=-1)
        )
        beta = torch.sigmoid(self.gate(torch.cat([image_emb, retrieved_emb], dim=-1)))
        aligned = self.norm(image_emb + beta * residual)
        return F.normalize(aligned, dim=-1), beta


class RJPModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pad_idx: int,
        use_ram: bool = True,
        use_adaptive_fusion: bool = True,
        fixed_fusion_alpha: float = 0.5,
    ):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim)
        self.decoder = AdaptiveFusionDecoder(
            vocab_size,
            embed_dim,
            hidden_dim,
            pad_idx,
            use_adaptive_fusion=use_adaptive_fusion,
            fixed_fusion_alpha=fixed_fusion_alpha,
        )
        self.report_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.ram = ResidualAlignmentModule(embed_dim)
        self.use_ram = use_ram
        self.pad_idx = pad_idx

    def encode_retrieved(self, retrieved_tokens):
        emb = self.report_embed(retrieved_tokens)
        mask = (retrieved_tokens != self.pad_idx).unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(summed / denom, dim=-1)

    def forward(self, images, retrieved_tokens, tokens_in):
        retrieved_emb = self.encode_retrieved(retrieved_tokens)
        image_emb = self.encoder(images)
        if self.use_ram:
            aligned_image_emb, ram_gate = self.ram(image_emb, retrieved_emb)
        else:
            aligned_image_emb = image_emb
            ram_gate = torch.zeros(image_emb.size(0), 1, device=image_emb.device)
        logits = self.decoder(aligned_image_emb, retrieved_emb, tokens_in)
        aux = {
            "raw_image_emb": image_emb,
            "aligned_image_emb": aligned_image_emb,
            "retrieved_emb": retrieved_emb,
            "ram_gate": ram_gate,
        }
        return logits, aux
