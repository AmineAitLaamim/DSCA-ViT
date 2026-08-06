# DSCA-ViT — Module Sanity Check
# Run this in Colab BEFORE starting training to verify
# all modules are importable and produce correct tensor shapes.

# ============================================================
# Cell 0 — Clone / Pull Repository
# ============================================================

import subprocess
import os

REPO_URL = "https://github.com/AmineAitLaamim/DSCA-ViT.git"
REPO_DIR = "/content/DSCA-ViT"

if not os.path.exists(REPO_DIR):
    print("Cloning repository...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print("✅ Repository cloned.")
else:
    print("Pulling latest changes...")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
    print("✅ Repository updated.")

# Add repository to Python path
import sys
sys.path.insert(0, REPO_DIR)

print(f"REPO_DIR: {REPO_DIR}")


# ============================================================
# Cell 0b — Imports
# ============================================================

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

B = 2  # Batch size for testing

# ============================================================
# Test 1 — Color Deconvolution
# ============================================================

print("=" * 50)
print("Test 1: ColorDeconvolution")
print("=" * 50)

from models.color_deconv import ColorDeconvolution

deconv = ColorDeconvolution().to(device)

x_rgb = torch.rand(B, 3, 224, 224).to(device)  # [0, 1] range
h_ch, d_ch = deconv(x_rgb)

assert h_ch.shape == (B, 1, 224, 224), f"H shape wrong: {h_ch.shape}"
assert d_ch.shape == (B, 1, 224, 224), f"DAB shape wrong: {d_ch.shape}"
assert h_ch.min() >= 0.0, "H channel has negative values"
assert d_ch.min() >= 0.0, "DAB channel has negative values"
assert sum(p.numel() for p in deconv.parameters()) == 0, "Deconv has learnable params!"

print(f"  H shape  : {h_ch.shape}  ✅")
print(f"  DAB shape: {d_ch.shape}  ✅")
print(f"  H range  : [{h_ch.min():.3f}, {h_ch.max():.3f}]")
print(f"  DAB range: [{d_ch.min():.3f}, {d_ch.max():.3f}]")
print(f"  Learnable params: 0  ✅")


# ============================================================
# Test 2 — Stain Channel Projection (1->3)
# ============================================================

print("\n" + "=" * 50)
print("Test 2: StainChannelProjection")
print("=" * 50)

from models.shared_vit import StainChannelProjection

proj = StainChannelProjection(init_mode="repeat").to(device)

x_1ch = torch.rand(B, 1, 224, 224).to(device)
x_3ch = proj(x_1ch)

assert x_3ch.shape == (B, 3, 224, 224), f"Proj output wrong: {x_3ch.shape}"

print(f"  Input  : {x_1ch.shape}  ✅")
print(f"  Output : {x_3ch.shape}  ✅")
print(f"  Params : {sum(p.numel() for p in proj.parameters())}  (expected ~12)")


# ============================================================
# Test 3 — Shared ViT Encoder
# ============================================================

print("\n" + "=" * 50)
print("Test 3: SharedViTEncoder")
print("=" * 50)

from models.shared_vit import SharedViTEncoder

encoder = SharedViTEncoder(pretrained=False, split_after=9).to(device)

x_in = torch.rand(B, 3, 224, 224).to(device)
tokens = encoder.embed(x_in)
assert tokens.shape == (B, 197, 768), f"Embed shape wrong: {tokens.shape}"

tokens_before = encoder.forward_before(tokens)
assert tokens_before.shape == (B, 197, 768)

tokens_after = encoder.forward_after(tokens_before)
assert tokens_after.shape == (B, 197, 768)

print(f"  embed()         : {tokens.shape}  ✅")
print(f"  forward_before(): {tokens_before.shape}  ✅")
print(f"  forward_after() : {tokens_after.shape}  ✅")


# ============================================================
# Test 4 — Bidirectional Cross-Attention
# ============================================================

print("\n" + "=" * 50)
print("Test 4: BidirectionalCrossAttention")
print("=" * 50)

from models.cross_attention import BidirectionalCrossAttention

bca = BidirectionalCrossAttention(embed_dim=768, num_heads=12).to(device)

h_tokens = torch.rand(B, 197, 768).to(device)
d_tokens = torch.rand(B, 197, 768).to(device)

h_out, d_out = bca(h_tokens, d_tokens)

assert h_out.shape == (B, 197, 768), f"H out shape wrong: {h_out.shape}"
assert d_out.shape == (B, 197, 768), f"D out shape wrong: {d_out.shape}"

print(f"  H input  : {h_tokens.shape}")
print(f"  D input  : {d_tokens.shape}")
print(f"  H output : {h_out.shape}  ✅")
print(f"  D output : {d_out.shape}  ✅")
print(f"  Params   : {sum(p.numel() for p in bca.parameters()):,}")


# ============================================================
# Test 5 — Gated Fusion
# ============================================================

print("\n" + "=" * 50)
print("Test 5: GatedFusion")
print("=" * 50)

from models.fusion import GatedFusion

fusion = GatedFusion(embed_dim=768).to(device)

fused, gates = fusion(h_out, d_out)

assert fused.shape == (B, 197, 768), f"Fused shape wrong: {fused.shape}"
assert gates.shape == (B, 196, 768), f"Gates shape wrong: {gates.shape}"
assert gates.min() >= 0.0 and gates.max() <= 1.0, "Gate values outside [0,1]!"

print(f"  Fused tokens: {fused.shape}  ✅")
print(f"  Gate values : {gates.shape}  ✅")
print(f"  Gate range  : [{gates.min():.3f}, {gates.max():.3f}]  ✅")


# ============================================================
# Test 6 — Refinement Block
# ============================================================

print("\n" + "=" * 50)
print("Test 6: RefinementBlock")
print("=" * 50)

from models.fusion import RefinementBlock

refinement = RefinementBlock(embed_dim=768, num_heads=12).to(device)

refined = refinement(fused)
assert refined.shape == (B, 197, 768), f"Refined shape wrong: {refined.shape}"

print(f"  Input  : {fused.shape}")
print(f"  Output : {refined.shape}  ✅")


# ============================================================
# Test 7 — Classification Head
# ============================================================

print("\n" + "=" * 50)
print("Test 7: ClassificationHead")
print("=" * 50)

from models.fusion import ClassificationHead

head = ClassificationHead(embed_dim=768, num_classes=4).to(device)

logits = head(refined)
assert logits.shape == (B, 4), f"Logits shape wrong: {logits.shape}"

print(f"  Input  : {refined.shape}")
print(f"  Logits : {logits.shape}  ✅")


# ============================================================
# Test 8 — Full DSCA-ViT End-to-End
# ============================================================

print("\n" + "=" * 50)
print("Test 8: Full DSCAViT (end-to-end)")
print("=" * 50)

from models import DSCAViT

model = DSCAViT(
    num_classes=4,
    pretrained=False,   # Fast test, no download
    split_after=9,
).to(device)

x_test = torch.rand(B, 3, 224, 224).to(device)
logits = model(x_test)

assert logits.shape == (B, 4), f"Final logits wrong: {logits.shape}"

gates = model.get_gate_values()
assert gates is not None
assert gates.shape == (B, 196, 768)

counts = model.count_parameters()

print(f"  Input  : {x_test.shape}")
print(f"  Logits : {logits.shape}  ✅")
print(f"  Gates  : {gates.shape}  ✅")
print()
print("  Parameter Summary:")
for name, count in counts.items():
    print(f"    {name:<20}: {count:>12,}")

print()
print("=" * 50)
print("ALL TESTS PASSED ✅")
print("=" * 50)
