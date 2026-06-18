"""
ControlNet branch for the ADM ``UNetModel``, adapting Zhang et al. 2023
("Adding Conditional Control to Text-to-Image Diffusion Models") to the
consistency-model setting.

Design summary (the "why"):

* The frozen ``UNetModel`` is the pretrained consistency-model backbone. We do
  not touch its weights, so its one-step / multistep behaviour (and the
  parameterization f = c_skip * x + c_out * F) is preserved exactly.
* A *trainable copy* of the UNet encoder (``input_blocks`` + ``middle_block``)
  reads the same noisy input ``x`` and, additionally, an encoded edge map. Its
  per-block features are pushed through **zero-initialized 1x1 convolutions**
  and added back onto the frozen UNet's skip connections (and middle output).
* Because every injection point starts at exactly zero (zero-convs + the
  zero-initialized last layer of the hint encoder), at initialization the
  controlled model is *identical* to the frozen backbone. Training can only
  move away from that safe starting point gradually -> stable fine-tuning.

Interaction with the consistency-model boundary condition f(x, eps, c) = x:
the scalings c_skip(eps)=1 and c_out(eps)=0 are fixed functions of sigma in
``KarrasDenoiser`` and are *not* learned. Since c_out(eps)=0 multiplies the
*entire* network output F (backbone + control), the boundary condition holds
for every edge map c and at every point in training -- the control branch can
never break it. This is the key reason this design is sound for CMs.
"""

import copy

import torch as th
import torch.nn as nn

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import conv_nd, zero_module, timestep_embedding
from .unet import UNetModel


class ControlNet(nn.Module):
    """
    A ControlNet branch mirroring the encoder of a given :class:`UNetModel`.

    :param base_unet: the backbone UNet whose architecture (and, after the
        weights are loaded, parameters) this branch copies.
    :param hint_channels: number of channels in the conditioning image (1 for a
        single-channel binary edge map).
    :param hint_hidden: hidden width of the small input-hint encoder.
    """

    def __init__(self, base_unet: UNetModel, hint_channels: int = 1, hint_hidden: int = 16):
        super().__init__()
        dims = 2
        self.dtype = base_unet.dtype
        self.model_channels = base_unet.model_channels
        self.num_classes = base_unet.num_classes

        # --- Trainable copies of the encoder pathway. -----------------------
        # deepcopy preserves both architecture and (current) weights. The
        # caller is responsible for re-syncing these from the *loaded*
        # pretrained backbone (see scripts/cm_train init helper), since at
        # construction time the backbone is still randomly initialized.
        self.time_embed = copy.deepcopy(base_unet.time_embed)
        if self.num_classes is not None:
            self.label_emb = copy.deepcopy(base_unet.label_emb)
        self.input_blocks = copy.deepcopy(base_unet.input_blocks)
        self.middle_block = copy.deepcopy(base_unet.middle_block)

        # --- Input-hint encoder: edge map -> first-skip feature space. ------
        # Same spatial resolution as input_blocks[0]'s output; its channel
        # count must match the first skip connection. Last conv is zeroed so
        # the injected hint starts at 0.
        first_ch = base_unet._input_block_chans[0]
        self.input_hint_block = nn.Sequential(
            conv_nd(dims, hint_channels, hint_hidden, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, hint_hidden, hint_hidden, 3, padding=1),
            nn.SiLU(),
            zero_module(conv_nd(dims, hint_hidden, first_ch, 3, padding=1)),
        )

        # --- Zero convolutions, one per encoder skip + one for the middle. --
        # Each maps a feature map to a residual of identical shape, initialized
        # to zero so it contributes nothing until trained.
        self.zero_convs = nn.ModuleList(
            [zero_module(conv_nd(dims, ch, ch, 1)) for ch in base_unet._input_block_chans]
        )
        mid_ch = base_unet._middle_block_channels
        self.middle_block_out = zero_module(conv_nd(dims, mid_ch, mid_ch, 1))

    def forward(self, x, hint, timesteps, y=None):
        """
        :param x: the (scaled) noisy input fed to the backbone, [N, C, H, W].
        :param hint: the conditioning edge map, [N, hint_channels, H, W].
        :param timesteps: 1-D batch of (rescaled) timesteps.
        :param y: optional class labels (if the backbone is class-conditional).
        :return: a list ``[*skip_residuals, middle_residual]`` aligned with the
            backbone's ``hs`` skip connections plus the middle-block output.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y is not None, "class-conditional ControlNet requires y"
            emb = emb + self.label_emb(y)

        guided_hint = self.input_hint_block(hint.type(self.dtype))

        outs = []
        h = x.type(self.dtype)
        for i, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            h = module(h, emb)
            if i == 0:
                # Inject the encoded edge map right after the stem conv, exactly
                # where ControlNet adds it in the original formulation.
                h = h + guided_hint
            outs.append(zero_conv(h))
        h = self.middle_block(h, emb)
        outs.append(self.middle_block_out(h))
        return outs

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.input_hint_block.apply(convert_module_to_f16)
        self.zero_convs.apply(convert_module_to_f16)
        self.middle_block_out.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.input_hint_block.apply(convert_module_to_f32)
        self.zero_convs.apply(convert_module_to_f32)
        self.middle_block_out.apply(convert_module_to_f32)


class ControlledModel(nn.Module):
    """
    Wraps a (optionally frozen) backbone :class:`UNetModel` together with a
    :class:`ControlNet` branch into a single module with the same call
    signature the rest of the codebase expects.

    Its ``forward(x, timesteps, edge=None, y=None)`` is invoked by
    ``KarrasDenoiser.denoise`` as ``model(c_in * x_t, rescaled_t, **model_kwargs)``,
    so the edge map simply rides along inside ``model_kwargs`` -- which means it
    flows, unchanged, through training_losses / consistency_losses (CD & CT) /
    karras_sample without any further plumbing.

    When ``edge`` is None the control branch is skipped entirely and the call is
    identical to the bare backbone (useful for unconditional sanity checks and
    classifier-free-style training).
    """

    def __init__(self, unet: UNetModel, controlnet: ControlNet, freeze_unet: bool = True,
                 only_mid_control: bool = False):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.only_mid_control = only_mid_control
        self.freeze_unet = freeze_unet
        if freeze_unet:
            self.unet.requires_grad_(False)

        # Mirror a few attributes that external code reads off the model.
        self.dtype = unet.dtype
        self.num_classes = unet.num_classes
        self.image_size = unet.image_size

    def forward(self, x, timesteps, edge=None, y=None):
        if edge is None:
            control = None
        else:
            control = self.controlnet(x, edge, timesteps, y=y)
        return self.unet(
            x, timesteps, y=y, control=control, only_mid_control=self.only_mid_control
        )

    def convert_to_fp16(self):
        self.unet.convert_to_fp16()
        self.controlnet.convert_to_fp16()
        self.dtype = th.float16

    def convert_to_fp32(self):
        self.unet.convert_to_fp32()
        self.controlnet.convert_to_fp32()
        self.dtype = th.float32


def sync_controlnet_from_unet(controlnet: ControlNet, unet: UNetModel):
    """
    Copy the (loaded) backbone encoder weights into the ControlNet's trainable
    copy. Call this *after* loading a pretrained checkpoint into ``unet`` so the
    control branch starts as a faithful, trainable replica of the pretrained
    encoder (zero-convs and the hint encoder stay at their zero/random init).
    """
    controlnet.time_embed.load_state_dict(unet.time_embed.state_dict())
    if controlnet.num_classes is not None:
        controlnet.label_emb.load_state_dict(unet.label_emb.state_dict())
    controlnet.input_blocks.load_state_dict(unet.input_blocks.state_dict())
    controlnet.middle_block.load_state_dict(unet.middle_block.state_dict())
