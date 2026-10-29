# Hardware, time and cost

What each model size needs, and what the commands took when measured.

## On your own GPU

An NVIDIA GPU with 8 GB runs the 2B, 16 GB runs the 4B, and 24 GB runs the 9B. The model you study must fit on the GPU. The 9B judge (about 19 GB) doesn't have to: what doesn't fit stays in CPU memory and moves to the GPU a layer at a time as it scores. On an 8 GB card that is about 15 GB of RAM for the judge (an estimate).

Measured on 2026-10-27 on a 24 GB A10G, with the kit installed as above (with flash-linear-attention) and the weights already downloaded. `PERSONAKIT_GPU_GB` held some commands to 16 or 8 GB, as on a smaller card. Minutes are each command's wall time, loading included.

| Command | GPU | Minutes | Peak GPU memory |
|---|---|---|---|
| `build --model 4b` | 16 GB | 4.4 | 13.2 GB |
| `make "a wizard" --model 4b` (atlas built) | 16 GB | 3.2 | 16.1 GB |
| `make "a wizard" --model 4b` (atlas built) | 24 GB | 1.9 | 19.2 GB |
| `explain 22 2742 --sign + --model 4b --measure --concept "an alien"` | 24 GB | 0.6 | 18.9 GB |
| `twins 22 2742 --sign + --from 4b --to 9b` | 24 GB | 0.6 | 18.6 GB |
| `make "a wizard" --model 2b` (including the 2B's atlas) | 8 GB | 5.0 | 7.4 GB |
| `port wizard.json --to 9b` (including the 9B's atlas) | 24 GB | 14.8 | 22.0 GB |

The peaks come from nvidia-smi, which counts the CUDA context outside the cap. On 16 GB the judge partly sits in CPU memory, and its labels came out identical to those on 24 GB.

When a batch doesn't fit, the kit halves it until it does, and says so. On 24 GB, the 9B generated 50 replies at a time instead of 100, and scored MMLU 16 items at a time instead of 32.

`PERSONAKIT_GPU_GB=<n>` keeps the kit within n GB of the GPU, if you share it.

## On Modal

Measured on Modal on 2026-10-27, with the weights already cached. Minutes are each function's own time. Starting a container and loading the weights add a minute or two.

| Command | GPU | Minutes | About |
|---|---|---|---|
| `build --model 4b` (the atlas: 580 replies, 1,000 Pile passages) | A10G | 4.8 | $0.14 |
| `make "a wizard" --model 4b` (atlas built) | A10G | 3.6 | $0.10 |
| `port wizard.json --to 2b` (including the 2B's atlas) | A10G | 5.0 | $0.14 |
| `port wizard.json --to 9b` (including the 9B's atlas) | L40S | 5.6 | $0.24 |

The whole check cost $1.33, including the 4B's atlas, `explain --measure` on the alien neuron, those three recipes and the 9B twins search. The first command on a model also downloads its weights, and every judged command needs the 9B judge's.

Every reply is judged by Qwen3.5-9B, which reads the label off the digit with the highest probability. It agreed with blind hand labels on 96% of identity judgements (κ 0.92).
