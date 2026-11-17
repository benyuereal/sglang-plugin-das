from __future__ import annotations

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.hyperconnection import GroupedGemmaRMSNorm
from sglang.srt.layers.linear import ReplicatedLinear


def _init_pre_fc_norms(self, config):
    self.pre_fc_norm_embedding = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    if self.hc_count > 1:
        self.pre_fc_norm_hidden = GroupedGemmaRMSNorm(
            self.hc_count * config.hidden_size, eps=config.rms_norm_eps,
            group_size=config.hidden_size,
        )
    else:
        self.pre_fc_norm_hidden = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def _init_linear_projections(self, config):
    self.fc_embedding = ReplicatedLinear(
        config.hidden_size, config.hidden_size, bias=False,
        quant_config=self.quant_config, prefix="mtp.fc_embedding",
    )
    self.fc_hidden = ReplicatedLinear(
        config.hidden_size, config.hidden_size, bias=False,
        quant_config=self.quant_config, prefix="mtp.fc_hidden",
    )


def _fuse_residual_linear_shared(self, input_embeds, hidden_states):
    input_embeds, _ = self.fc_embedding(self.pre_fc_norm_embedding(input_embeds))
    orig_shape = hidden_states.shape
    hidden_states = self.pre_fc_norm_hidden(hidden_states)
    decoder_view = hidden_states.view(*hidden_states.shape[:-1], self.hc_count, self.hidden_size)
    encoder_inputs, _ = self.fc_hidden(decoder_view.reshape(-1, self.hidden_size))
    encoder_inputs = encoder_inputs.view_as(decoder_view)
    return (input_embeds.unsqueeze(-2) + encoder_inputs).view(orig_shape)
