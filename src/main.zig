//! main.zig - Default entry point
//!
//! Nothing to see here yet
//!
//! Copyright 2025 Joe

const std = @import("std");

pub fn main() !void {
    std.debug.print(
        \\
        \\MLX Zig ML Models
        \\================
        \\
        \\This project contains multiple ML model interfaces:
        \\
        \\- For Whisper transcription: zig build run-whisper
        \\- For Llama 3.2 chat: zig build run-llama
        \\
        \\Please use the specific build command for your desired model.
        \\
        \\
    , .{});
}

test "Redirects" {
    std.debug.print("\n=== MAIN.ZIG ===\n\n", .{});
    _ = @import("whisper_main.zig");
    _ = @import("llama_main.zig");
}
