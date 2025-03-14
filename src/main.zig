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
const Tokenizer = @import("tokenizer.zig").DefaultTokenizer;
const Transformer = @import("transformer.zig").DefaultTransformer;
const c = @cImport({
    @cInclude("mlx/c/mlx.h");
    @cInclude("stdio.h");
});

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const user_input = "Hello world";
    const num_tokens_to_generate = 20;

    // Load tokenizer
    var tokenizer = try Tokenizer.init(allocator, null);
    defer tokenizer.deinit();

    // Load transformer
    var transformer = try Transformer.init(allocator, null);
    defer transformer.deinit();

    // Encode input string to token IDs (chat format)
    const input_ids = try tokenizer.encodeChat(allocator, "You are a helpful assistant.", user_input);
    defer allocator.free(input_ids);
    std.debug.print("Input IDs: {any}\n\n", .{input_ids});

    // Generate new token IDs
    const output_ids = try transformer.generate(input_ids, num_tokens_to_generate);
    defer allocator.free(output_ids);

    // Decode input and output token IDs
    const input_str = try tokenizer.decode(input_ids);
    defer allocator.free(input_str);
    const output_str = try tokenizer.decode(output_ids);
    defer allocator.free(output_str);

    std.debug.print("\nOutput IDs: {any}\n\nInput: {s}Output: {s}", .{ output_ids, input_str, output_str });

    // --- RESULT --- //
    // Input IDs: { 128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 271, 2675, 527, 264, 11190, 18328, 13, 128009, 128006, 882, 128007, 271, 9906, 1917, 128009, 128006, 78191, 128007, 271 }

    // Generated token 1/20: 9906
    // Generated token 2/20: 0
    // Generated token 3/20: 1102
    // Generated token 4/20: 596
    // Generated token 5/20: 6555
    // Generated token 6/20: 311
    // Generated token 7/20: 3449
    // Generated token 8/20: 499
    // Generated token 9/20: 13
    // Generated token 10/20: 2209
    // Generated token 11/20: 1070
    // Generated token 12/20: 2555
    // Generated token 13/20: 358
    // Generated token 14/20: 649
    // Generated token 15/20: 1520
    // Generated token 16/20: 499
    // Generated token 17/20: 449
    // Generated token 18/20: 477
    // Generated token 19/20: 1053
    // Generated token 20/20: 499

    // Output IDs: { 9906, 0, 1102, 596, 6555, 311, 3449, 499, 13, 2209, 1070, 2555, 358, 649, 1520, 499, 449, 477, 1053, 499 }

    // Input: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    // Cutting Knowledge Date: December 2023
    // Today Date: 26 Jul 2024

    // You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

    // Hello world<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    // Output: Hello! It's nice to meet you. Is there something I can help you with or would you
}

test "Transformer chatting" {
    std.debug.print("\n=== MAIN.ZIG ===\n\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const user_input = "Hello world";
    const num_tokens_to_generate = 20;
    var tokenizer = try Tokenizer.init(allocator, null);
    defer tokenizer.deinit();
    var transformer = try Transformer.init(allocator, null);
    defer transformer.deinit();
    const input_ids = try tokenizer.encodeChat(allocator, "You are a helpful assistant.", user_input);
    defer allocator.free(input_ids);
    std.debug.print("Input IDs: {any}\n", .{input_ids});
    const output_ids = try transformer.generate(input_ids, num_tokens_to_generate);
    defer allocator.free(output_ids);
    const input_str = try tokenizer.decode(input_ids);
    defer allocator.free(input_str);
    const output_str = try tokenizer.decode(output_ids);
    defer allocator.free(output_str);
    std.debug.print("\nOutput IDs: {any}\n\nInput: {s}Output: {s}", .{ output_ids, input_str, output_str });
}

test "Transformer decoding" {
    std.debug.print("\n=== MAIN.ZIG ===\n\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tokenizer = try Tokenizer.init(allocator, null);
    defer tokenizer.deinit();
    var transformer = try Transformer.init(allocator, null);
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
