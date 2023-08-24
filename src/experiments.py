import json
import os
import sys

import matplotlib.pyplot as plt
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

from patch_attn import patch_llama
from unlimiformer import UnlimiformerLLaMa
from usage import UnlimiformerArguments

base_name = "meta-llama/Llama-2-7b-hf"

dtype = torch.float16
short_name = base_name.split("/")[-1]

tokenizer = AutoTokenizer.from_pretrained(base_name)

config = AutoConfig.from_pretrained(base_name)
with init_empty_weights():
    model = LlamaForCausalLM(config)

# Tie weights
model.tie_weights()

model = load_checkpoint_and_dispatch(
    model,
    base_name,
    dtype=dtype,
    device_map="auto",
    no_split_module_classes=["LlamaDecoderLayer"],
)

defaults = UnlimiformerArguments()
unlimiformer_kwargs = {
    "layer_begin": 16,
    "tokenizer": tokenizer,
    "use_datastore": True,
    "gpu_datastore": True,
    "gpu_index": True,
    "datastore_device": 4,
    "index_devices": [1, 2, 3, 4],
    "flat_index": True,
    # "verbose": True,
}

model = UnlimiformerLLaMa.convert_model(model, **unlimiformer_kwargs)

patch_llama(model)
model.eval()

print("Model loaded")

dataset = load_dataset("tau/scrolls", "gov_report")["validation"]

# fmt: off
long_inputs = dataset[[1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 17, 18, 20, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34, 35, 37, 38, 41, 42, 45, 48, 49, 50, 52, 57, 59, 61, 62, 64]]["input"]
print("Datatset loaded")

def get_logits(logits, input_ids):
    input_ids = input_ids[0, 1 : logits.size(1) + 1].unsqueeze(-1)
    logits = logits.log_softmax(dim=-1)[0].gather(1, input_ids).squeeze()
    return logits


def get_accumulated_perplexity(logits):
    accumulated_logits = logits.cumsum(dim=0)
    size = torch.ones_like(accumulated_logits).int().cumsum(dim=0)
    return (-accumulated_logits / size).exp().tolist()


max_len = 6000
max_chunk_len = 4000
extended_segment_len = 1000
num_chunks = int((max_len - max_chunk_len) / extended_segment_len)

all_logits = []


with torch.no_grad():
    for context in tqdm(long_inputs[:2]):
        feature = tokenizer(context, return_tensors="pt")
        input_ids = feature["input_ids"]
        attention_mask = feature["attention_mask"]

        logits = []

        model.unlimiformer.reset_memory(
            input_ids=input_ids[:, :max_chunk_len],
            attention_mask=attention_mask[:, :max_chunk_len],
        )

        out = model(
            input_ids=input_ids[:, :max_chunk_len],
            attention_mask=attention_mask[:, :max_chunk_len],
        )
        logits_first = get_logits(out.logits.float(), input_ids)

        logits.append(logits_first)

        for i in trange(num_chunks):
            idx_start = max_chunk_len + extended_segment_len * i
            idx_end = max_chunk_len + extended_segment_len * (i + 1)

            tmp_input_ids = input_ids[:, :idx_end]
            tmp_attn_mask = attention_mask[:, :idx_end]

            model.unlimiformer.reset_memory(
                input_ids=tmp_input_ids, attention_mask=tmp_attn_mask
            )

            out = model(input_ids=tmp_input_ids, attention_mask=tmp_attn_mask)

            logits_extended = get_logits(
                out.logits.float()[:, -extended_segment_len:],
                tmp_input_ids[:, idx_start:],
            )

            logits.append(logits_extended)

        all_logits.append(torch.cat(logits, dim=0))

mean_logits = torch.stack(all_logits, dim=0).mean(dim=0)
if mean_logits.isnan().any() or mean_logits.isinf().any():
    print("[WARNING] NaNs or Infs in logits")


logits_accum = get_accumulated_perplexity(mean_logits)

with open("./result.json", "w") as f:
    f.write(str(logits_accum))



plt.plot(
    logits_accum,
    label="Unlimiformer",
)

# plt.ylim([2, 6])
plt.title("Language modelling peformance of Unlimiformer")
plt.legend(loc="upper right")
plt.xlabel("Number of tokens")
plt.ylabel("Perplexity")
plt.show()
