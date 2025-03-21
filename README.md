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

2. Build and run the example demo:
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

Output: Hello World!

This is the classic first program written
```

</details>

*This example demonstrates the use of quantized models to significantly reduce memory usage. A Llama 3.2 model originally sized at 2.67GB (bfloat16) is compressed to under 700MB, enabling efficient performance on resource-constrained systems.*

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
