//! llm_main.zig - Unified entry point for LLM models
//!
//! Copyright 2025 Joe

const std = @import("std");
const download = @import("utils.zig").download;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const LlamaTransformer = @import("llama.zig").Transformer;
const PhiTransformer = @import("phi.zig").Transformer;
const QwenTransformer = @import("qwen.zig").Transformer;

const ModelType = enum {
    llama,
    phi,
    qwen_coder,
    olympic_coder,

    pub fn fromString(str: []const u8) !ModelType {
        if (std.mem.eql(u8, str, "llama")) return .llama;
        if (std.mem.eql(u8, str, "phi")) return .phi;
        if (std.mem.eql(u8, str, "qwen")) return .qwen_coder;
        if (std.mem.eql(u8, str, "olympic")) return .olympic_coder;
        return error.InvalidModelType;
    }
};

pub const ModelConfig = struct {
    const Self = @This();
    model_type: ModelType,
    model_name: []const u8,
    num_tokens: usize = 30,
    chat_format: ?[]const u8,
    system_prompt: []const u8,
    user_input: []const u8,
    required_files: ?[]const []const u8 = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, model_type: ModelType) !Self {
        var config = Self{
            .model_type = model_type,
            .model_name = undefined,
            .chat_format = null,
            .system_prompt = "",
            .user_input = "",
            .allocator = allocator,
        };

        switch (model_type) {
            .llama => {
                config.model_name = "Llama-3.2-1B-Instruct-4bit";
                config.chat_format =
                    \\<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    \\
                    \\Cutting Knowledge Date: December 2023
                    \\Today Date: 26 Jul 2024
                    \\
                    \\{s}<|eot_id|><|start_header_id|>user<|end_header_id|>
                    \\
                    \\{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    \\
                    \\
                ;
                config.system_prompt = "You are a helpful assistant.";
                config.user_input = "How to get length of ArrayList in Zig?";
            },
            .phi => {
                config.model_name = "phi-4-2bit";
                config.chat_format =
                    \\<|im_start|>system<|im_sep|>
                    \\{s}<|im_end|>
                    \\<|im_start|>user<|im_sep|>
                    \\{s}<|im_end|>
                    \\<|im_start|>assistant<|im_sep|>
                    \\
                ;
                config.system_prompt = "You are a medieval knight and must provide explanations to modern people.";
                config.user_input = "How should I explain the Internet?";
                const files = &[_][]const u8{
                    "config.json",                      "tokenizer.json",
                    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                };
                config.required_files = try allocator.dupe([]const u8, files);
            },
            .qwen_coder => {
                config.model_name = "Qwen2.5-Coder-1.5B-4bit";
                config.chat_format = null;
                config.system_prompt = "";
                config.user_input =
                    \\<|fim_prefix|>def quicksort(arr):
                    \\    if len(arr) <= 1:
                    \\        return arr
                    \\    pivot = arr[len(arr) // 2]
                    \\    <|fim_suffix|>
                    \\    middle = [x for x in arr if x == pivot]
                    \\    right = [x for x in arr if x > pivot]
                    \\    return quicksort(left) + middle + quicksort(right)<|fim_middle|>
                ;
            },
            .olympic_coder => {
                config.model_name = "OlympicCoder-7B-4bit";
                config.chat_format =
                    \\<|im_start|>user
                    \\{s}<|im_end|>
                    \\<|im_start|>assistant
                    \\
                ;
                config.system_prompt = "";
                config.user_input = "Write a python program to calculate the 10th fibonacci number";
            },
        }

        return config;
    }

    pub fn deinit(self: *Self) void {
        if (self.required_files) |files| {
            self.allocator.free(files);
        }
    }

    pub fn generate(self: *Self, token_ids: []const u32) ![]const u32 {
        return switch (self.model_type) {
            .llama => blk: {
                var transformer = try LlamaTransformer.init(self.allocator, self.model_name);
                defer transformer.deinit();
                break :blk try transformer.generate(token_ids, self.num_tokens);
            },
            .phi => blk: {
                var transformer = try PhiTransformer.init(self.allocator, self.model_name);
                defer transformer.deinit();
                break :blk try transformer.generate(token_ids, self.num_tokens);
            },
            .qwen_coder, .olympic_coder => blk: {
                var transformer = try QwenTransformer.init(self.allocator, self.model_name);
                defer transformer.deinit();
                break :blk try transformer.generate(token_ids, self.num_tokens);
            },
        };
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var model_type: ModelType = .llama;
    var user_input: ?[]const u8 = null;
    var system_prompt: ?[]const u8 = null;
    var num_tokens: usize = 30;
    var model_name: ?[]const u8 = null;

    const process_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, process_args);

    var i: usize = 1;
    while (i < process_args.len) : (i += 1) {
        const arg = process_args[i];
        if (std.mem.startsWith(u8, arg, "--")) {
            if (std.mem.indexOf(u8, arg, "=")) |equals_pos| {
                const key = arg[2..equals_pos];
                const value = arg[equals_pos + 1 ..];
                if (std.mem.eql(u8, key, "model-type")) {
                    model_type = try ModelType.fromString(value);
                } else if (std.mem.eql(u8, key, "model-name")) {
                    model_name = value;
                } else if (std.mem.eql(u8, key, "system-prompt")) {
                    system_prompt = value;
                } else if (std.mem.eql(u8, key, "num-tokens")) {
                    num_tokens = try std.fmt.parseInt(usize, value, 10);
                }
            } else if (std.mem.eql(u8, arg[2..], "help")) {
                printUsage();
                std.process.exit(0);
            }
        } else {
            user_input = arg;
        }
    }

    var config = try ModelConfig.init(allocator, model_type);
    defer config.deinit();

    if (user_input) |input| {
        config.user_input = input;
    }
    if (system_prompt) |prompt| {
        config.system_prompt = prompt;
    }
    if (model_name) |name| {
        config.model_name = name;
    }
    config.num_tokens = num_tokens;

    try runModel(&config);
}

fn runModel(config: *ModelConfig) !void {
    try download(config.allocator, "mlx-community", config.model_name, config.required_files);
    var tokenizer = try Tokenizer.init(config.allocator, config.model_name);
    defer tokenizer.deinit();

    var format_args: []const []const u8 = undefined;
    if (config.chat_format) |format| {
        var placeholders: usize = 0;
        var i: usize = 0;
        while (i + 2 < format.len) : (i += 1) {
            if (format[i] == '{' and format[i + 1] == 's' and format[i + 2] == '}') {
                placeholders += 1;
                i += 2;
            }
        }

        if (placeholders == 1) {
            format_args = &[_][]const u8{config.user_input};
        } else if (placeholders == 2) {
            format_args = &[_][]const u8{ config.system_prompt, config.user_input };
        } else {
            return error.UnsupportedFormatString;
        }
    } else {
        format_args = &[_][]const u8{config.user_input};
    }

    const input_ids = try tokenizer.encodeChat(config.chat_format, format_args);
    defer config.allocator.free(input_ids);

    const output_ids = try config.generate(input_ids);
    defer config.allocator.free(output_ids);

    const output_str = try tokenizer.decode(output_ids);
    defer config.allocator.free(output_str);
    std.debug.print("\nInput: {s}\n\nOutput: {s}\n", .{ config.user_input, output_str });
}

fn printUsage() void {
    const usage =
        \\Usage: llm [options] [input]
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
    ;
    std.debug.print("{s}", .{usage});
}

test "All LLM models" {
    std.debug.print("\n=== LLM_MAIN.ZIG ===\n\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    inline for (.{ .llama, .phi, .qwen_coder, .olympic_coder }) |model_type| {
        var config = try ModelConfig.init(allocator, model_type);
        defer config.deinit();
        config.num_tokens = 5;
        try runModel(&config);
    }
    _ = gpa.deinit();
}
