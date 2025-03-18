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

2. Download the [Llama-3.2-1B-Instruct model weights](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/model.safetensors) (2.47GB) and place it in the project root directory.

3. Build and run the example demo:
```
zig build run
```

This will compile MLX from source and run a simple text generation demo:

<details>
<summary>Click to expand</summary>

```text
Input: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello world<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Output: Hello! It's nice to meet you. Is there something I can help you with or would you
```

</details>

## Examples

```zig
// Load tokenizer
var tokenizer = try Tokenizer.init(allocator, null);
defer tokenizer.deinit();

// Load transformer
var transformer = try Transformer.init(allocator, null);
defer transformer.deinit();

// Encode input string to token IDs (chat format)
const input_ids = try tokenizer.encodeChat(allocator, "You are a helpful assistant.", user_input);
defer allocator.free(input_ids);

// Generate new tokens
const output_ids = try transformer.generate(input_ids, num_tokens_to_generate);
defer allocator.free(output_ids);
```

## Acknowledgements

This project's build system is based on Erik Kaunismäki's [zig-build-mlx](https://github.com/ErikKaum/zig-build-mlx), which pioneered the approach of building MLX directly with Zig rather than using CMake. This project uses a condensed version of Erik's build configuration.

## License

This project is licensed under the [Apache License 2.0](LICENSE)
