// Copyright 2025 Joe
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const std = @import("std");
const download = @import("utils.zig").download;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Transformer = @import("transformer.zig").DefaultTransformer;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const model_name = "Llama-3.2-1B-Instruct-4bit";
    const sys_prompt = "You are a helpful assistant.";
    const num_tokens = 100;

    // Download model files
    try download(allocator, model_name);

    // Load tokenizer
    var tokenizer = try Tokenizer.init(allocator, model_name);
    defer tokenizer.deinit();

    // Load transformer
    var transformer = try Transformer.init(allocator, model_name);
    defer transformer.deinit();

    // Prompt user for input
    const stdin = std.io.getStdIn().reader();
    std.debug.print("\n\n===============\n\nEnter your message: ", .{});
    var input_buffer: [1024]u8 = undefined;
    const input_slice = try stdin.readUntilDelimiterOrEof(&input_buffer, '\n') orelse "";
    const user_input = try allocator.dupe(u8, input_slice);
    defer allocator.free(user_input);

    // Encode input string to token IDs (chat format)
    const input_ids = try tokenizer.encodeChat(null, sys_prompt, user_input);
    defer allocator.free(input_ids);
    std.debug.print("\nInput IDs: {any}\n\n", .{input_ids});

    // Generate new token IDs
    const output_ids = try transformer.generate(input_ids, num_tokens);
    defer allocator.free(output_ids);

    // Decode input and output token IDs
    const input_str = try tokenizer.decode(input_ids);
    defer allocator.free(input_str);
    const output_str = try tokenizer.decode(output_ids);
    defer allocator.free(output_str);

    std.debug.print("\nOutput IDs: {any}\n\nInput: {s}Output: {s}\n", .{ output_ids, input_str, output_str });
}

test "Llama-3.2-1B-Instruct-4bit chat" {
    std.debug.print("\n=== MAIN.ZIG ===\n\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const model_name = "Llama-3.2-1B-Instruct-4bit";
    const sys_prompt = "You are a helpful assistant.";
    const user_input = "Hello world";
    const num_tokens = 10;
    try download(allocator, model_name);
    var tokenizer = try Tokenizer.init(allocator, model_name);
    defer tokenizer.deinit();
    var transformer = try Transformer.init(allocator, model_name);
    defer transformer.deinit();
    const input_ids = try tokenizer.encodeChat(null, sys_prompt, user_input);
    defer allocator.free(input_ids);
    std.debug.print("Input IDs: {any}\n\n", .{input_ids});
    const output_ids = try transformer.generate(input_ids, num_tokens);
    defer allocator.free(output_ids);
    const input_str = try tokenizer.decode(input_ids);
    defer allocator.free(input_str);
    const output_str = try tokenizer.decode(output_ids);
    defer allocator.free(output_str);
    std.debug.print("\nOutput IDs: {any}\n\nInput: {s}Output: {s}", .{ output_ids, input_str, output_str });
}

test "Llama-3.2-1B-Instruct-4bit raw" {
    std.debug.print("\n=== MAIN.ZIG ===\n\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tokenizer = try Tokenizer.init(allocator, "Llama-3.2-1B-Instruct-4bit");
    defer tokenizer.deinit();
    var transformer = try Transformer.init(allocator, "Llama-3.2-1B-Instruct-4bit");
    defer transformer.deinit();
    const initial_tokens = [_]u32{ 9906, 1917 };
    const num_tokens_to_generate = 10;
    const generated_tokens = try transformer.generate(&initial_tokens, num_tokens_to_generate);
    defer allocator.free(generated_tokens);
    std.debug.print("\nOutput IDs: ", .{});
    for (generated_tokens) |token| {
        std.debug.print("{d} ", .{token});
    }
    std.debug.print("\n", .{});
    const input_str = try tokenizer.decode(&initial_tokens);
    defer allocator.free(input_str);
    std.debug.print("Prompt: {s}\n", .{input_str});
    const output_str = try tokenizer.decode(generated_tokens);
    defer allocator.free(output_str);
    std.debug.print("Output: {s}\n", .{output_str});
}
