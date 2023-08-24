import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaForCausalLM


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def find_relative_keys(self, output):
    # output: (batch, time, 3 * heads * attention_dim)
    if self.is_input_encoding_pass or self.is_first_test_decoding_step:
        return
    with torch.no_grad():
        prompt_size = self.prompt_input_ids.shape[1]
        generated_size = self.input_ids_size - prompt_size
        window_size = self.actual_model_window_size
        # topk = min(self.actual_model_window_size, attn_weights.shape[-1])
        topk = min(prompt_size, window_size)
        if not self.is_encoder_decoder:
            topk = min(topk, window_size - generated_size + 1)
        if self.gpu_index:
            topk = min(topk, 2048)

        query = self.process_query(output)[:, -1]  # (batch * beam, head, dim)
        query = query[:, self.head_nums]  # (batch * beam, head, dim)

        # query: (batch, beam, head, dim)
        # need to multiply by key vector
        # query.view(query.shape[0], query.shape[1] * query.shape[2])
        # k_proj in attention?
        datastore_index = 0 if self.is_encoder_decoder else self.cur_decoder_layer_index
        attention_layer_list = self.get_kv_projections(self.layer_begin, self.layer_end)
        k_proj_layer = [layers[0] for layers in attention_layer_list][
            self.cur_decoder_layer_index
        ]
        v_proj_layer = [layers[1] for layers in attention_layer_list][
            self.cur_decoder_layer_index
        ]

        # modify query by k_projs
        k_proj = k_proj_layer.weight
        datastore_query = self.preprocess_query(
            query, k_proj
        )  # (batch * beam, num_heads, embed_dim)
        batch_size = self.datastore[datastore_index].batch_size
        datastore_query = datastore_query.view(
            (batch_size, -1, datastore_query.shape[2])
        )  # (batch, beam * num_heads, embed_dim)
        # then search
        _, top_search_key_indices = self.datastore[datastore_index].search(
            datastore_query, k=topk
        )
        # self.embeddings: (batch,              src_len, dim)
        # indices:         (batch, beam * head, actual_model_window_size)
        # embeddings: (batch, beam * head, actual_model_window_size, dim)
        embeddings = torch.take_along_dim(
            input=self.hidden_states[datastore_index].unsqueeze(1),
            indices=top_search_key_indices.unsqueeze(-1).to(
                self.hidden_states[datastore_index].device
            ),
            dim=-2,
        )
        # embeddings = embeddings.to(self.device)
        # (batch, beam, head, actual_model_window_size)
        # top_search_key_scores = top_search_key_scores.reshape(batch_size, -1, *top_search_key_scores.shape[1:])
        top_search_key_indices = top_search_key_indices.reshape(
            batch_size, -1, *top_search_key_indices.shape[1:]
        )
        # embeddings: (batch, beam, head, actual_model_window_size, dim)
        embeddings = embeddings.reshape(
            batch_size, -1, self.num_heads, *embeddings.shape[2:]
        )

    # k_proj_layer.weight, v_proj_layer.weight: (embed_dim, embed_dim)
    # embeddings: (batch, beam, head, encoder_len, embed_dim)
    retrieved_keys, retrieved_values = self.post_process_retrieved(
        embeddings, k_proj_layer, v_proj_layer, top_search_key_indices
    )

    # retrieved_keys, retrieved_values: (batch * beam, head, encoder_len, attn_dim)
    retrieved_keys = retrieved_keys.flatten(0, 1)[:, :, :topk]
    retrieved_values = retrieved_values.flatten(0, 1)[:, :, :topk]

    return [retrieved_keys, retrieved_values]


def patch_llama_attn(attn, unlimiformer):
    self = attn

    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        past_key_value = find_relative_keys(unlimiformer, hidden_states)

        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            past_key_value[0] = past_key_value[0].to(hidden_states.device)
            past_key_value[1] = past_key_value[1].to(hidden_states.device)

            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                # raise ValueError(
                #     f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                # )
                # ! TODO: Check this again

                attention_mask = torch.ones_like(attn_weights)
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, None

    self.forward = forward

    return


def patch_llama(model: LlamaForCausalLM):
    for layer in model.model.layers:
        patch_llama_attn(layer.self_attn, model.unlimiformer)


if __name__ == "__main__":
    llama = LlamaForCausalLM.from_pretrained("/data/llama/Llama-7b-hf")

    patch_llama(llama)
