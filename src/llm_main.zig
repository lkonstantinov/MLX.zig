//! llm_main.zig - Entry point for LLM models
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

const TransformerUnion = union(ModelType) {
    llama: LlamaTransformer,
    phi: PhiTransformer,
    qwen_coder: QwenTransformer,
    olympic_coder: QwenTransformer,
};

pub const ModelConfig = struct {
    const Self = @This();
    model_type: ModelType,
    model_name: []const u8,
    num_tokens: usize = 30,
    chat_format: ?[]const u8 = null,
    chat_content: []const []const u8,
    required_files: ?[]const []const u8 = null,
    allocator: std.mem.Allocator,
    transformer: ?TransformerUnion = null,

    pub fn init(allocator: std.mem.Allocator, model_type: ModelType) !Self {
        var content = std.ArrayList([]const u8).init(allocator);
        errdefer content.deinit();

        var config = Self{
            .model_type = model_type,
            .model_name = undefined,
            .chat_content = undefined,
            .allocator = allocator,
            .transformer = null,
        };

        switch (model_type) {
            .llama => {
                config.model_name = "Llama-3.2-1B-Instruct-4bit";
                config.chat_format =
                    \\<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    \\
                    \\Cutting Knowledge Date: December 2023
                    \\Today Date: 26 Jul 2024
                    \\{s}<|eot_id|><|start_header_id|>user<|end_header_id|>
                    \\
                    \\{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    \\
                    \\
                ;

                try content.append("You are a helpful assistant.");
                try content.append("How to get length of ArrayList in Zig?");
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

                try content.append("You are a medieval knight and must provide explanations to modern people.");
                try content.append("How should I explain the Internet?");

                const files = &[_][]const u8{
                    "config.json",                      "tokenizer.json",
                    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                };
                config.required_files = try allocator.dupe([]const u8, files);
            },
            .qwen_coder => {
                config.model_name = "Qwen2.5-Coder-1.5B-4bit";
                config.chat_format = null;

                try content.append(
                    \\<|fim_prefix|>def quicksort(arr):
                    \\    if len(arr) <= 1:
                    \\        return arr
                    \\    pivot = arr[len(arr) // 2]
                    \\    <|fim_suffix|>
                    \\    middle = [x for x in arr if x == pivot]
                    \\    right = [x for x in arr if x > pivot]
                    \\    return quicksort(left) + middle + quicksort(right)<|fim_middle|>
                );
            },
            .olympic_coder => {
                config.model_name = "OlympicCoder-7B-4bit";
                config.chat_format =
                    \\<|im_start|>user
                    \\{s}<|im_end|>
                    \\<|im_start|>assistant
                    \\
                ;
                try content.append("Write a python program to calculate the 10th fibonacci number");
            },
        }
        config.chat_content = try content.toOwnedSlice();
        return config;
    }

    pub fn loadTransformer(self: *Self) !void {
        if (self.transformer != null) return;

        self.transformer = switch (self.model_type) {
            .llama => .{ .llama = try LlamaTransformer.init(self.allocator, self.model_name) },
            .phi => .{ .phi = try PhiTransformer.init(self.allocator, self.model_name) },
            .qwen_coder => .{ .qwen_coder = try QwenTransformer.init(self.allocator, self.model_name) },
            .olympic_coder => .{ .olympic_coder = try QwenTransformer.init(self.allocator, self.model_name) },
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.required_files) |files| {
            self.allocator.free(files);
        }
        self.allocator.free(self.chat_content);
        if (self.transformer) |*t| {
            switch (t.*) {
                .llama => |*transformer| transformer.deinit(),
                .phi => |*transformer| transformer.deinit(),
                .qwen_coder => |*transformer| transformer.deinit(),
                .olympic_coder => |*transformer| transformer.deinit(),
            }
        }
    }

    pub fn setContent(self: *Self, content: []const []const u8) !void {
        self.allocator.free(self.chat_content);
        self.chat_content = try self.allocator.dupe([]const u8, content);
    }

    pub fn generate(self: *Self, token_ids: []const u32) ![]u32 {
        if (self.transformer) |*t| {
            return switch (t.*) {
                .llama => |*transformer| transformer.generate(token_ids, self.num_tokens),
                .phi => |*transformer| transformer.generate(token_ids, self.num_tokens),
                .qwen_coder => |*transformer| transformer.generate(token_ids, self.num_tokens),
                .olympic_coder => |*transformer| transformer.generate(token_ids, self.num_tokens),
            };
        }
        return error.TransformerNotInitialized;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var model_type: ModelType = .llama;
    var num_tokens: usize = 30;
    var model_name: ?[]const u8 = null;
    var system_prompt: ?[]const u8 = null;

    var replacements = std.ArrayList([]const u8).init(allocator);
    defer replacements.deinit();

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
            try replacements.append(arg);
        }
    }

    if (system_prompt) |prompt| {
        try replacements.insert(0, prompt);
    } else if (replacements.items.len > 0) {
        if (model_type == .llama or model_type == .phi) {
            try replacements.insert(0, "You are a helpful assistant.");
        }
    }

    var config = try ModelConfig.init(allocator, model_type);
    defer config.deinit();

    if (model_name) |name| {
        config.model_name = name;
    }
    config.num_tokens = num_tokens;

    if (replacements.items.len > 0) {
        try config.setContent(replacements.items);
    }

    try runModel(&config);
}

fn runModel(config: *ModelConfig) !void {
    std.debug.print("Downloading model: {s}\n", .{config.model_name});
    try download(config.allocator, "mlx-community", config.model_name, config.required_files);

    std.debug.print("Initializing tokenizer\n", .{});
    var tokenizer = try Tokenizer.init(config.allocator, config.model_name);
    defer tokenizer.deinit();

    std.debug.print("Loading transformer\n", .{});
    try config.loadTransformer();

    std.debug.print("Encoding input\n", .{});
    const input_ids = try tokenizer.encodeChat(config.chat_format, config.chat_content);
    defer config.allocator.free(input_ids);

    std.debug.print("Starting model generation\n", .{});
    const output_ids = try config.generate(input_ids);
    defer config.allocator.free(output_ids);

    std.debug.print("Decoding output\n", .{});
    const output_str = try tokenizer.decode(output_ids);
    defer config.allocator.free(output_str);

    std.debug.print("\nInput: {s}\n\nOutput: {s}\n", .{
        if (config.chat_content.len > 0) config.chat_content[config.chat_content.len - 1] else "",
        output_str,
    });
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
        \\  llm --model-type=phi --system-prompt="You are a helpful assistant" "How should I explain the Internet?"
        \\  llm --model-type=olympic "Write a python function to check if a number is prime"
        \\
    ;
    std.debug.print("{s}", .{usage});
}

test "Llama model" {
    std.debug.print("\n=== TESTING LLAMA MODEL ===\n\n", .{});
    const allocator = std.testing.allocator;
    var config = try ModelConfig.init(allocator, .llama);
    defer config.deinit();
    config.num_tokens = 5;
    try runModel(&config);
}

test "Phi model" {
    std.debug.print("\n=== TESTING PHI MODEL ===\n\n", .{});
    const allocator = std.testing.allocator;
    var config = try ModelConfig.init(allocator, .phi);
    defer config.deinit();
    config.num_tokens = 5;
    try runModel(&config);
}

test "Qwen model" {
    std.debug.print("\n=== TESTING QWEN MODEL ===\n\n", .{});
    const allocator = std.testing.allocator;
    var config = try ModelConfig.init(allocator, .qwen_coder);
    defer config.deinit();
    config.num_tokens = 5;
    try runModel(&config);
}

test "Olympic model" {
    std.debug.print("\n=== TESTING OLYMPIC MODEL ===\n\n", .{});
    const allocator = std.testing.allocator;
    var config = try ModelConfig.init(allocator, .olympic_coder);
    defer config.deinit();
    config.num_tokens = 5;
    try runModel(&config);
}
