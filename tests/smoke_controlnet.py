"""
CPU smoke test for the ControlNet-on-consistency-model integration.

Runs WITHOUT installing anything: it stubs out ``mpi4py`` / ``blobfile`` (only
needed for distributed IO, not for the math here), uses a tiny attention-free
UNet, fp32, and the l2 consistency loss (no ``piq``). It validates the load-
bearing properties of the design:

  1. forward / denoise produce correct shapes;
  2. at zero-init the controlled model == the bare backbone (edge has no effect);
  3. the boundary condition f(x, sigma_min, c) == x holds for any edge map;
  4. gradients reach ONLY the ControlNet branch; the backbone stays frozen;
  5. the conditional consistency-training loss runs and backprops.

Run:  /usr/local/anaconda3/envs/pytorch/bin/python tests/smoke_controlnet.py
"""
import os
import sys
import types

# --- hermetic stubs (import-time deps we don't exercise here) ----------------
_mpi = types.ModuleType("mpi4py")


class _Comm:
    rank = 0
    size = 1

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, x, root=0):
        return x

    def Barrier(self):
        pass


_mpi.MPI = types.SimpleNamespace(COMM_WORLD=_Comm())
sys.modules["mpi4py"] = _mpi

_bf = types.ModuleType("blobfile")
_bf.BlobFile = open
_bf.join = os.path.join
_bf.dirname = os.path.dirname
_bf.basename = os.path.basename
_bf.exists = os.path.exists
_bf.isdir = os.path.isdir
_bf.listdir = os.listdir
sys.modules["blobfile"] = _bf

# Make sure we import the in-repo cm package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy

import torch as th

from cm.unet import UNetModel
from cm.controlnet import ControlNet, ControlledModel, sync_controlnet_from_unet
from cm.karras_diffusion import KarrasDenoiser

th.manual_seed(0)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{PASS if cond else FAIL}] {name}")


def build():
    # Tiny, attention-free, unconditional UNet (no einops / flash needed).
    unet = UNetModel(
        image_size=16,
        in_channels=3,
        model_channels=32,  # must be divisible by GroupNorm's 32 groups
        out_channels=3,
        num_res_blocks=1,
        attention_resolutions=(),
        channel_mult=(1, 2),
        num_classes=None,
        use_fp16=False,
        num_heads=1,
        use_scale_shift_norm=True,
        resblock_updown=False,
    )
    controlnet = ControlNet(unet, hint_channels=1, hint_hidden=4)
    sync_controlnet_from_unet(controlnet, unet)
    model = ControlledModel(unet, controlnet, freeze_unet=True)

    # ADM zero-inits the final output conv; with a *random* backbone that would
    # block all gradient on step 0. A real run loads a pretrained checkpoint
    # whose output conv is nonzero -- emulate that so the gradient-flow checks
    # below are meaningful.
    with th.no_grad():
        for p in unet.out.parameters():
            p.data.normal_(0.0, 0.1)
    diffusion = KarrasDenoiser(
        sigma_data=0.5, sigma_max=80.0, sigma_min=0.002,
        distillation=True, loss_norm="l2",
    )
    return unet, controlnet, model, diffusion


def main():
    unet, controlnet, model, diffusion = build()
    B = 2
    x = th.randn(B, 3, 16, 16)
    edge = (th.rand(B, 1, 16, 16) > 0.5).float()
    sigmas = th.rand(B) * 10 + 0.1

    # 1. shapes
    model_out, denoised = diffusion.denoise(model, x, sigmas, edge=edge)
    check("denoise output shape", model_out.shape == (B, 3, 16, 16))
    check("denoise denoised shape", denoised.shape == (B, 3, 16, 16))

    # 2. zero-init: controlled == backbone, and edge has no effect yet
    model.eval()
    with th.no_grad():
        t = sigmas
        y_base = unet(x, t)
        y_edge = model(x, t, edge=edge)
        y_none = model(x, t, edge=None)
    check("zero-init: controlled(edge) == backbone", th.allclose(y_edge, y_base, atol=1e-5))
    check("zero-init: controlled(None) == backbone", th.allclose(y_none, y_base, atol=1e-5))

    # 3. boundary condition: at sigma_min, denoised == x for any edge
    sig_min = th.full((B,), 0.002)
    with th.no_grad():
        _, den_min = diffusion.denoise(model, x, sig_min, edge=edge)
    check("boundary condition denoised(sigma_min) == x", th.allclose(den_min, x, atol=1e-4))

    # 4 & 5. conditional CT loss backprops; grads only on ControlNet
    target = copy.deepcopy(model)
    target.requires_grad_(False)
    model.train()
    terms = diffusion.consistency_losses(
        model, x, num_scales=10, target_model=target, model_kwargs={"edge": edge}
    )
    loss = terms["loss"].mean()
    loss.backward()

    unet_grads = [p.grad for p in model.unet.parameters() if p.grad is not None]
    cnet_grads = [p.grad for p in model.controlnet.parameters() if p.grad is not None]
    check("CT loss is finite", th.isfinite(loss).item())
    check("backbone receives NO gradient", len(unet_grads) == 0)
    check("ControlNet receives gradient", len(cnet_grads) > 0)

    # zero-convs specifically should get gradient even though they output 0
    zc_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.controlnet.zero_convs.parameters()
    )
    check("zero-convs receive non-zero gradient", zc_grad)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nparams: total={n_total:,}  trainable(ControlNet)={n_train:,} "
          f"({100.0 * n_train / n_total:.1f}%)")

    test_script_util_path()

    ok = all(c for _, c in results)
    print("\n=== SMOKE TEST:", "ALL PASS ===" if ok else "FAILURES PRESENT ===")
    sys.exit(0 if ok else 1)


def test_script_util_path():
    """Validate the script_util construction path used by the real scripts."""
    from cm.script_util import create_model_and_diffusion, model_and_diffusion_defaults
    from cm.controlnet import ControlledModel

    cfg = model_and_diffusion_defaults()
    cfg.update(
        dict(
            image_size=64,
            num_channels=32,         # divisible by 32 groups
            num_res_blocks=1,
            class_cond=True,         # ImageNet-64 backbone is class-conditional
            use_controlnet=True,
            hint_channels=1,
            loss_norm="l2",          # avoid piq in this no-deps environment
            use_fp16=False,
        )
    )
    cfg["distillation"] = True
    model, diffusion = create_model_and_diffusion(**cfg)
    check("script_util builds ControlledModel", isinstance(model, ControlledModel))

    # Emulate a loaded (nonzero output conv) pretrained backbone so gradients
    # can flow into the ControlNet on step 0 (see note in build()).
    with th.no_grad():
        for p in model.unet.out.parameters():
            p.data.normal_(0.0, 0.1)

    B = 2
    x = th.randn(B, 3, 64, 64)
    edge = (th.rand(B, 1, 64, 64) > 0.5).float()
    y = th.randint(0, 1000, (B,))
    sigmas = th.rand(B) * 10 + 0.1
    with th.no_grad():
        _, denoised = diffusion.denoise(model, x, sigmas, edge=edge, y=y)
    check("script_util class-cond denoise shape", denoised.shape == (B, 3, 64, 64))

    # Mirror CMTrainLoop.forward_backward for consistency_training: same call
    # signature, with BOTH conditioning keys (class label + edge map) in
    # model_kwargs, then an EMA target update. This is the closest CPU proxy to
    # a real ImageNet-64 ControlNet training step.
    from cm.nn import update_ema

    target = copy.deepcopy(model)
    target.requires_grad_(False)
    model.train()
    terms = diffusion.consistency_losses(
        model, x, num_scales=40, target_model=target,
        model_kwargs={"edge": edge, "y": y},
    )
    loss = terms["loss"].mean()
    loss.backward()
    check("class+edge CT step: loss finite", bool(th.isfinite(loss).item()))
    cnet_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.controlnet.parameters()
    )
    check("class+edge CT step: ControlNet gets gradient", cnet_grad)
    backbone_grad = [p for p in model.unet.parameters() if p.grad is not None]
    check("class+edge CT step: backbone frozen", len(backbone_grad) == 0)
    # EMA target update must run over the ControlledModel params without error.
    update_ema(list(target.parameters()), list(model.parameters()), rate=0.99)
    check("EMA target update runs on ControlledModel", True)


if __name__ == "__main__":
    main()
