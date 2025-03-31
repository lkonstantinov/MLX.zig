//! whisper_main.zig - Entry point for the Whisper-Turbo-Large-v3 speech-to-text app
//!
//! Copyright 2025 Joe

const std = @import("std");
const download = @import("utils.zig").download;
const Transcriber = @import("whisper.zig").Transcriber;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();
    const model_name = "whisper-large-v3-turbo";
    const audio_file = "alive.mp3";
    try download(allocator, "openai", model_name);
    var transcriber = try Transcriber.init(allocator, model_name);
    defer transcriber.deinit();
    const transcription = try transcriber.transcribe(audio_file);
    defer allocator.free(transcription);
    std.debug.print("\nTranscription of {s}:{s}\n", .{ audio_file, transcription });
}

test "Whisper transcription" {
    std.debug.print("\n=== WHISPER_MAIN.ZIG ===\n\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();
    const model_name = "whisper-large-v3-turbo";
    const audio_file = "alive.mp3";
    try download(allocator, "openai", model_name);
    var transcriber = try Transcriber.init(allocator, model_name);
    defer transcriber.deinit();
    const transcription = try transcriber.transcribe(audio_file);
    defer allocator.free(transcription);
    std.debug.print("\nTranscription of {s}: {s}", .{ audio_file, transcription });
}
