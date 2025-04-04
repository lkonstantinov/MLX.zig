//! main.zig - Default entry point
//!
//! Copyright 2025 Joe

const std = @import("std");
pub fn main() !void {
    std.debug.print(
        \\
        \\MLX Zig ML Models
        \\================
        \\
        \\Zig Build System Usage:
        \\
        \\  zig build                           - Builds all executables and installs to zig-out/bin
        \\  zig build run-whisper -- [file]     - Builds and runs whisper with optional audio file
        \\  zig build run-phi                   - Builds and runs phi cli
        \\  zig build run-llama                 - Builds and runs llama chat
        \\  zig build run-llm                   - Builds and runs llm (qwen, olympic, phi, llama)
        \\  zig build run                       - Builds and runs main.zig (for development/testing)
        \\
        \\Exe Usage:
        \\
        \\  zig-out/bin/whisper [audio_file]    - Run whisper with optional audio file
        \\  zig-out/bin/phi [prompt]            - Run phi with optional user prompt
        \\  zig-out/bin/llama                   - Run llama chat
        \\  zig-out/bin/main                    - Run main app with build instructions
        \\  zig-out/bin/llm [options] [input]    - Run llm app (see below)
        \\
        \\LLM Usage: llm [options] [input]
        \\
        \\Options:
        \\  --model-type=TYPE       Model type: llama, phi, qwen, olympic (default: llama)
        \\  --model-name=NAME       Model name to download/use
        \\  --system-prompt=PROMPT  System prompt for the model
        \\  --num-tokens=N          Number of tokens to generate
        \\  --help                  Show this help
        \\
        \\Examples:
        \\  llm "Hi mom!"
        \\  llm --model-type=phi "How should I explain the Internet?"
        \\  llm --model-type=olympic "Write a python function to check if a number is prime"
        \\
    , .{});
}

test "Redirects" {
    std.debug.print("\n=== MAIN.ZIG ===\n\n", .{});
    _ = @import("phi_main.zig");
    _ = @import("whisper_main.zig");
    _ = @import("llama_main.zig");
    _ = @import("llm_main.zig");
}
