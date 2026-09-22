# llama.cpp: fix slow Vulkan readback on AMD iGPUs (UMA)

A 4-line patch for [llama.cpp](https://github.com/ggml-org/llama.cpp). On AMD integrated GPUs, reading a tensor back from the GPU goes through a `memcpy` from **write-combined** memory at roughly **190 MB/s**. Models with a recurrent state (hybrid attention + Gated DeltaNet, e.g. Qwen3.6-A3B) save that state whenever llama-server creates a context checkpoint, so a single checkpoint of 62.8 MiB costs **324 ms** while the actual GPU work for the request is ~14 ms.

With the patch, follow-up turns of a chat go from **873 ms to 258 ms** of prompt-processing time on a Radeon 8060S, and `llama-bench` throughput is unchanged.

Not in upstream. See [Upstream status](#upstream-status).

## Symptoms

You are on an AMD APU (Strix Halo / Strix Point / Phoenix / Hawk Point) with the Vulkan backend, running a model that has a recurrent state, and:

- short requests take 300-700 ms even though the GPU graph is fast,
- `--verbose` shows `slot create_check: created context checkpoint ... size = NN MiB` right before a `prompt eval time` of a few hundred ms for a handful of tokens,
- latency jumps as soon as a request appends to a cached prefix, or processes 5+ new tokens, but 1-4 fresh tokens are fast,
- `--ctx-checkpoints 0 --cache-ram 0` makes it go away (at the cost of losing prefix reuse).

Dense models (no recurrent state) are unaffected. Discrete GPUs are unaffected.

## Cause

`ggml_vk_buffer_read_2d` (the Vulkan implementation behind `ggml_backend_tensor_get`) has two paths:

1. **UMA path** - a pipeline barrier, a fence wait, then `memcpy` straight from the mapped buffer.
2. **Staging path** - `vkCmdCopyBuffer` into `sync_staging` (which is allocated `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`), fence wait, then `memcpy` from there.

On UMA devices, `ggml_vk_create_buffer_device` prefers `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` and never asks for `HOST_CACHED`. On RADV that memory type is write-combined: writes are fast, but CPU reads bypass the cache and crawl. The UMA path is taken whenever the buffer is host-visible, so every readback from such a buffer runs at write-combined read speed.

This rarely mattered because llama.cpp mostly writes to device buffers. The exception is `llama_state_seq_get_data`, which walks every recurrent-state tensor through `ggml_backend_tensor_get`. llama-server calls it for context checkpoints (`--ctx-checkpoints`, 32 per slot by default) and for the RAM prompt cache (`--cache-ram`, 8 GiB by default).

Evidence on a Ryzen AI Max+ 395 (Radeon 8060S, RADV GFX1151) with Qwen3.6-35B-A3B Q4_K:

```
perf record -g during one checkpoint:
  91.93%  llama-server  libc.so.6  [.] __memmove_avx512_unaligned_erms
          --91.33%--ggml_backend_vk_buffer_get_tensor
                    llama_io_write_host::~llama_io_write_host()
```

`GGML_VK_PERF_LOGGER=1` reports ~20 ms of GPU work per graph in both the fast and the slow case, and `strace -e trace=ioctl` shows fence waits summing to under 80 ms across several requests. The time is neither GPU compute nor synchronization - it is the CPU-side `memcpy`.

Server log for a request that appends one token to a cached 8-token prefix:

```
05.208  slot create_check: erasing context checkpoint too close to an earlier one (... size = 62.813 MiB)
05.532  slot create_check: created context checkpoint 2 of 32 (pos_min = 7, pos_max = 7, n_tokens = 8, size = 62.813 MiB)
05.546  prompt eval time = 340.54 ms / 1 tokens
```

324 ms between the two `create_check` lines, 14 ms of actual decode.

## The patch

Take the direct-memcpy UMA path only when the source memory is `HOST_CACHED`; otherwise fall through to the staging path that already exists for discrete GPUs.

```diff
-    if(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma) {
+    // On UMA, only read through the mapping if the memory is host-cached. Reading uncached
+    // (write-combined) memory with memcpy is very slow, use the staging copy instead.
+    if(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma &&
+       src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached) {
```

Allocation policy is untouched, so nothing about GPU-side access changes. It relies on `memory_property_flags` reflecting the memory type that was actually allocated, which upstream [#24326](https://github.com/ggml-org/llama.cpp/pull/24326) fixed.

`0001-vulkan-uma-read-hostcached.patch` applies to `ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp` on current master. On trees from before the Vulkan backend was split into several files, the same function lives in `ggml-vulkan.cpp`; the patch is one condition, so adjusting it by hand is easy.

## Results

Upstream master `7ab4ee7ba`, same tree and same flags for both columns, only the patch differs. Ryzen AI Max+ 395 / Radeon 8060S / 64 GB / Mesa 25.2.8 / kernel 6.17 / Ubuntu 24.04. Qwen3.6-35B-A3B Q4_K (20.2 GiB, 30 recurrent layers, 62.8 MiB of recurrent state per sequence). Server: `-c 8192 -ngl 999 --flash-attn on -np 1`.

| | master | patched |
|---|---|---|
| one 62.8 MiB context checkpoint | 324 ms | **23 ms** |
| cached prefix + 1 new token (`prompt_ms`) | 343 ms | **40 ms** |
| 5 new tokens, no cache | 361 ms | **54 ms** |
| 1-4 new tokens, no cache (no checkpoint) | 14-20 ms | 14-20 ms |
| chat turn 1 (1821 tokens, cold) | 1973 ms | **1356 ms** |
| chat turn 2 (+139 tokens on the cached conversation) | 873 ms | **258 ms** |
| chat turn 3 (+139 tokens) | 874 ms | **263 ms** |
| 600 short single-token requests (~115-token prompts, repeated prefixes), p50 | 738 ms | **100 ms** |

No throughput regression:

| `llama-bench -ngl 999 -fa 1 -r 3` | master | patched |
|---|---|---|
| pp128 | 743.21 +/- 20.17 | 743.33 +/- 20.04 |
| pp512 | 1269.38 +/- 7.65 | 1266.96 +/- 5.89 |
| tg64 (alternating runs) | 81.42, 81.38 | 81.44, 79.92 |

Correctness:

- `test-backend-ops test -b Vulkan0`: 2/2 backends passed, both builds.
- `tests/test-save-load-state` with the 35B: all tests pass, identical generated tokens.
- `tests/test-state-restore-fragmented`: SUCCESS on both.
- After a checkpoint rollback and restore, the next-token top-10 probabilities match the unpatched build to 4 decimals across 6 trials.

Other hardware:

- **Radeon 780M (RADV PHOENIX)**, same model: cached prefix + 1 token 296 ms -> 88 ms; one checkpoint 242 ms -> 35 ms.
- **NVIDIA A40 (discrete)**: unchanged, as expected - `uma == false` never took the direct path. Measured before ~= after.

## Build

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git apply /path/to/0001-vulkan-uma-read-hostcached.patch
cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server
```

## Reproducing

`bench/checkpoint_cost.py` shows the cliff: latency versus new-token count, and versus cached-prefix length.

```sh
llama-server -m <hybrid-model>.gguf -c 8192 -ngl 999 --flash-attn on --verbose &
python3 bench/checkpoint_cost.py
```

Unpatched, you get a step from ~20 ms to ~350 ms at 5 new tokens, and a flat ~343 ms for "cached prefix + 1 token" regardless of prefix length. Patched, those become ~54 ms and ~40 ms.

`bench/chat_turns.py` measures what a multi-turn chat actually pays:

```sh
python3 bench/chat_turns.py
```

Any model with a recurrent state reproduces this. Dense models do not - useful as a control.

## Workaround without patching

`--ctx-checkpoints 0 --cache-ram 0` stops llama-server from saving state, so no readback happens. Fine if every request is a fresh short prompt (measured 180 ms versus 237 ms for the patched build with checkpoints on, since checkpoints still cost ~23 ms each).

It is not fine for chat: without checkpoints, a recurrent model cannot roll back to a cached prefix, so every turn reprocesses the whole conversation. Turn 2 of the example above goes from 258 ms to 1632 ms.

## Upstream status

Not submitted as of 2026-09-22. [#23762](https://github.com/ggml-org/llama.cpp/pull/23762) addresses the same symptom by making UMA device buffers prefer `HOST_CACHED` at allocation time. A maintainer pushed back that cached memory is intended for outputs, that using it everywhere may reduce GPU access performance, and that only one iGPU had been tested; the author moved it back to draft in September 2026 and it has not landed. This patch changes only the read path, so that objection does not apply to it, but it overlaps in purpose - if you want it upstream, commenting on #23762 is probably the right move rather than opening a duplicate.

## License

The patch is a modification of llama.cpp, which is MIT-licensed; it carries llama.cpp's license. The scripts in `bench/` are MIT (see `LICENSE`).
