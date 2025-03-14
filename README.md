# MLX.zig

A [Zig](https://ziglang.org/) language binding for [MLX](https://github.com/ml-explore/mlx), Apple's array framework for machine learning on Apple Silicon.

## Overview

MLX.zig provides a Zig-native interface to MLX, allowing you to build and run machine learning models on Apple Silicon devices using the Zig programming language. This project demonstrates how to:

- Use MLX from Zig without any additional build tools
- Implement a transformer-based language model
- Handle tokenization and text generation

## Features

- **Pure Zig Build System**: No CMake or other external build tools needed
- **Complete Dependency Management**: All C/C++ dependencies (MLX, MLX-C, PCRE2, etc.) resolved through Zig's build system
- **Working LLM Example**: Includes a tokenizer and transformer implementation capable of text generation
- **Efficient Tokenization**: Uses [PCRE2](https://github.com/PCRE2Project/pcre2) (Perl Compatible Regular Expressions) for fast and reliable text processing
- **Low-Level MLX Access**: Direct bindings to MLX's C API for maximum performance

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

This will compile MLX from source and run a simple text generation demo.

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

## How it Works

MLX.zig integrates Zig with Apple's ML framework through three key components:

1. **Zig Build System**: Compiles MLX (C++), MLX-C, and PCRE2 from source with zero external dependencies
2. **Transformer**: Implements a Llama-style language model with attention mechanisms and key-value caching
3. **Tokenizer**: Uses PCRE2 for efficient regex-based text processing, handling complex patterns and special tokens

The system works by encoding text to tokens, processing them through MLX tensor operations optimized for Apple Silicon, and decoding the generated output back to text—all managed through a clean Zig interface.

## Acknowledgements

This project's build system is based on Erik Kaunismäki's [zig-build-mlx](https://github.com/ErikKaum/zig-build-mlx), which pioneered the approach of building MLX directly with Zig rather than using CMake. This project uses a condensed version of Erik's build configuration.

## License

This project is licensed under the [Apache License 2.0](LICENSE)
