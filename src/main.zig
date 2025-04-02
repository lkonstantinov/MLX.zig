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
        \\  zig build run-phi                   - Builds and runs phi demo
        \\  zig build run-llama                 - Builds and runs llama chat
        \\  zig build run                       - Builds and runs main.zig (for development/testing)
        \\
        \\Direct executable usage:
        \\
        \\  zig-out/bin/whisper [audio_file]    - Run whisper with optional audio file
        \\  zig-out/bin/phi [prompt]            - Run phi with optional user prompt
        \\  zig-out/bin/llama                   - Run llama chat
        \\  zig-out/bin/main                    - Run main app with build instructions
        \\
        \\
    , .{});
}

test "Redirects" {
    std.debug.print("\n=== MAIN.ZIG ===\n\n", .{});
    _ = @import("phi_main.zig");
    _ = @import("whisper_main.zig");
    _ = @import("llama_main.zig");
}
