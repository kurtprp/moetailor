# GGUF — why repack is not in v0.1

GGUF expert tensors are packed with quantisation blocks
(e.g. Q4_K, IQ1_S). Deleting an expert is not just deleting
rows — you have to rewrite the packed blocks, the metadata
(`llama.expert_count`, etc.), and the tensor offsets, or
llama.cpp will refuse to load the file.

moetailor v0.1 therefore:
- Writes `keep_indices.json` alongside the pruned safetensors
  checkpoint, so a GGUF converter can use it later.
- Makes the safetensors checkpoint the primary output.
  That checkpoint loads in colibri (DeepSeek REAP does
  exactly this) and in vLLM / transformers.

If you want GGUF output today: prune in safetensors, then
convert the *pruned* checkpoint to GGUF with your usual
converter. The expert count in the converted GGUF will
already be the pruned count, because the config was rewritten.
