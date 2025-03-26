# MLX.zig

A [Zig](https://ziglang.org/) language binding for [MLX](https://github.com/ml-explore/mlx), Apple's array framework for machine learning on Apple Silicon.

## Features

1. **Build System**: Compiles MLX (C++), MLX-C, and PCRE2 from source using Zig's build system (no CMake or other build tools needed)
2. **Transformer**: Implements a Llama-3.2 language model with attention mechanisms and key-value caching
3. **Tokenizer**: Uses PCRE2 for efficient regex-based text processing, handling complex patterns and special tokens

## Prerequisites

- Apple Silicon Mac
- Zig v0.13.0

## Getting Started

1. Clone the repository:
```
git clone https://github.com/jaco-bro/MLX.zig.git
cd MLX.zig
```

2. Build and run the interactive demo:
```
zig build run
```

This will compile MLX from source and launch an interactive prompt where you can chat with the model:

```text
Enter your message:
```

<details>
<summary>Click to expand</summary>

```text
Enter your message: Hi, how have you been?

Input IDs: { 128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 271, 2675, 527, 264, 11190, 18328, 13, 128009, 128006, 882, 128007, 271, 13347, 11, 1268, 617, 499, 1027, 30, 128009, 128006, 78191, 128007, 271 }

Generated token 1/100: 40
Generated token 2/100: 2846
Generated token 3/100: 1120
Generated token 4/100: 264
Generated token 5/100: 4221
Generated token 6/100: 1646
Generated token 7/100: 11
Generated token 8/100: 358
Generated token 9/100: 1541
Generated token 10/100: 956
Generated token 11/100: 617
Generated token 12/100: 21958
Generated token 13/100: 477
Generated token 14/100: 16024
Generated token 15/100: 1093
Generated token 16/100: 12966
Generated token 17/100: 656
Generated token 18/100: 11
Generated token 19/100: 719
Generated token 20/100: 358
Generated token 21/100: 2846
Generated token 22/100: 31301
Generated token 23/100: 10489
Generated token 24/100: 323
Generated token 25/100: 5644
Generated token 26/100: 311
Generated token 27/100: 1520
Generated token 28/100: 449
Generated token 29/100: 904
Generated token 30/100: 4860
Generated token 31/100: 477
Generated token 32/100: 9256
Generated token 33/100: 499
Generated token 34/100: 1253
Generated token 35/100: 617
Generated token 36/100: 13
Generated token 37/100: 1102
Generated token 38/100: 596
Generated token 39/100: 2294
Generated token 40/100: 311
Generated token 41/100: 6865
Generated token 42/100: 430
Generated token 43/100: 499
Generated token 44/100: 2351
Generated token 45/100: 10371
Generated token 46/100: 1268
Generated token 47/100: 358
Generated token 48/100: 3077
Generated token 49/100: 1027
Generated token 50/100: 0
Generated token 51/100: 358
Generated token 52/100: 2846
Generated token 53/100: 3815
Generated token 54/100: 1664
Generated token 55/100: 11
Generated token 56/100: 9523
Generated token 57/100: 369
Generated token 58/100: 10371
Generated token 59/100: 13
Generated token 60/100: 358
Generated token 61/100: 3077
Generated token 62/100: 1027
Generated token 63/100: 16572
Generated token 64/100: 389
Generated token 65/100: 264
Generated token 66/100: 13057
Generated token 67/100: 3392
Generated token 68/100: 315
Generated token 69/100: 1495
Generated token 70/100: 828
Generated token 71/100: 11
Generated token 72/100: 902
Generated token 73/100: 6276
Generated token 74/100: 757
Generated token 75/100: 311
Generated token 76/100: 3493
Generated token 77/100: 11190
Generated token 78/100: 323
Generated token 79/100: 39319
Generated token 80/100: 14847
Generated token 81/100: 311
Generated token 82/100: 701
Generated token 83/100: 20126
Generated token 84/100: 13
Generated token 85/100: 2650
Generated token 86/100: 922
Generated token 87/100: 499
Generated token 88/100: 30
Generated token 89/100: 2650
Generated token 90/100: 596
Generated token 91/100: 701
Generated token 92/100: 1938
Generated token 93/100: 2133
Generated token 94/100: 779
Generated token 95/100: 3117
Generated token 96/100: 30
Generated token 97/100: 128009
EOS token reached after 97 tokens

Output IDs: { 40, 2846, 1120, 264, 4221, 1646, 11, 358, 1541, 956, 617, 21958, 477, 16024, 1093, 12966, 656, 11, 719, 358, 2846, 31301, 10489, 323, 5644, 311, 1520, 449, 904, 4860, 477, 9256, 499, 1253, 617, 13, 1102, 596, 2294, 311, 6865, 430, 499, 2351, 10371, 1268, 358, 3077, 1027, 0, 358, 2846, 3815, 1664, 11, 9523, 369, 10371, 13, 358, 3077, 1027, 16572, 389, 264, 13057, 3392, 315, 1495, 828, 11, 902, 6276, 757, 311, 3493, 11190, 323, 39319, 14847, 311, 701, 20126, 13, 2650, 922, 499, 30, 2650, 596, 701, 1938, 2133, 779, 3117, 30, 128009 }

Input: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi, how have you been?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Output: I'm just a language model, I don't have emotions or feelings like humans do, but I'm functioning properly and ready to help with any questions or tasks you may have. It's great to hear that you're asking how I've been! I'm doing well, thanks for asking. I've been trained on a vast amount of text data, which allows me to provide helpful and informative responses to your queries. How about you? How's your day going so far?<|eot_id|>⏎   
```

</details>

*The example uses 4-bit quantization to reduce the Llama 3.2 model size from 2.7GB (bfloat16) to under 0.7GB.*

## Examples

```zig
// Load tokenizer
var tokenizer = try Tokenizer.init(allocator, model_name);
defer tokenizer.deinit();

// Load transformer
var transformer = try Transformer.init(allocator, model_name);
defer transformer.deinit();

// Encode input string to token IDs (chat format)
const input_ids = try tokenizer.encodeChat(null, sys_prompt, user_input);
defer allocator.free(input_ids);

// Generate new tokens
const output_ids = try transformer.generate(input_ids, n_toks);
defer allocator.free(output_ids);
```

## Acknowledgements

This project's build system is based on Erik Kaunismäki's [zig-build-mlx](https://github.com/ErikKaum/zig-build-mlx).

## License

This project is licensed under the [Apache License 2.0](LICENSE)
