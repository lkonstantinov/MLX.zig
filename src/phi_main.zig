//! phi.zig - Entry point for the Phi-4 cli app
//!
//! Copyright 2025 Joe

const std = @import("std");
const download = @import("utils.zig").download;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Transformer = @import("phi.zig").Transformer;

pub fn main() !void {
    const chat_format =
        \\<|im_start|>system<|im_sep|>
        \\{s}<|im_end|>
        \\<|im_start|>user<|im_sep|>
        \\{s}<|im_end|>
        \\<|im_start|>assistant<|im_sep|>
        \\
    ;
    const model_name = "phi-4-2bit";
    const sys_prompt = "You are a medieval knight and must provide explanations to modern people.";
    const default_input = "How should I explain the Internet?";
    const num_tokens = 100;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    var user_input: []const u8 = default_input;
    if (args.len > 1) {
        user_input = args[1];
    }
    try download(allocator, "mlx-community", model_name, &.{ "config.json", "tokenizer.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" });
    var tokenizer = try Tokenizer.init(allocator, model_name);
    defer tokenizer.deinit();
    var transformer = try Transformer.init(allocator, model_name);
    defer transformer.deinit();
    const input_ids = try tokenizer.encodeChat(chat_format, sys_prompt, user_input);
    defer allocator.free(input_ids);
    const output_ids = try transformer.generate(input_ids, num_tokens);
    defer allocator.free(output_ids);
    const output_str = try tokenizer.decode(output_ids);
    defer allocator.free(output_str);
    std.debug.print("\nInput: {s}\n\nOutput: {s}\n", .{ user_input, output_str });
}

test "Phi-4-2bit cli" {
    std.debug.print("\n=== PHI_MAIN.ZIG ===\n\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const chat_format =
        \\<|im_start|>system<|im_sep|>
        \\{s}<|im_end|>
        \\<|im_start|>user<|im_sep|>
        \\{s}<|im_end|>
        \\<|im_start|>assistant<|im_sep|>
        \\
    ;
    const model_name = "phi-4-2bit";
    const sys_prompt = "You are a medieval knight and must provide explanations to modern people.";
    const user_input = "How should I explain the Internet?";
    const num_tokens = 10;
    try download(allocator, "mlx-community", model_name, &.{ "config.json", "tokenizer.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" });
    var tokenizer = try Tokenizer.init(allocator, model_name);
    defer tokenizer.deinit();
    var transformer = try Transformer.init(allocator, model_name);
    defer transformer.deinit();
    const input_ids = try tokenizer.encodeChat(chat_format, sys_prompt, user_input);
    defer allocator.free(input_ids);
    const output_ids = try transformer.generate(input_ids, num_tokens);
    defer allocator.free(output_ids);
    const input_str = try tokenizer.decode(input_ids);
    defer allocator.free(input_str);
    const output_str = try tokenizer.decode(output_ids);
    defer allocator.free(output_str);
    std.debug.print("\nInput: {s}Output: {s}\n", .{ input_str, output_str });
}
