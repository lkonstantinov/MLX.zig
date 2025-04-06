//! llama.zig - Llama-3.2-Instruct
//!
//! Copyright 2025 Joe

const std = @import("std");
const mlx = @import("mlx.zig");
const loadJson = @import("utils.zig").loadJson;
const allocJoin = @import("utils.zig").allocJoin;

pub const MLP = struct {
    const Self = @This();
    gate_weight: *mlx.Linear,
    up_weight: *mlx.Linear,
    down_weight: *mlx.Linear,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,
    allocs_to_free: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, quant_config: ?mlx.QuantConfig, weights_hash: *std.StringHashMap(*mlx.Array), stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        var allocs_to_free = std.ArrayList([]const u8).init(allocator);
        try allocs_to_free.append(key);
        return .{
            .gate_weight = try mlx.Linear.init(allocator, key, "gate_proj", false, quant_config, weights_hash, stream),
            .up_weight = try mlx.Linear.init(allocator, key, "up_proj", false, quant_config, weights_hash, stream),
            .down_weight = try mlx.Linear.init(allocator, key, "down_proj", false, quant_config, weights_hash, stream),
            .stream = stream,
            .allocator = allocator,
            .allocs_to_free = allocs_to_free,
        };
    }

    pub fn deinit(self: *Self) void {
        self.gate_weight.deinit();
        self.up_weight.deinit();
        self.down_weight.deinit();
        for (self.allocs_to_free.items) |key| self.allocator.free(key);
        self.allocs_to_free.deinit();
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        var gate = mlx.arrayNew();
        var sigmoid = mlx.arrayNew();
        var up = mlx.arrayNew();
        defer {
            mlx.arrayFree(gate);
            mlx.arrayFree(sigmoid);
            mlx.arrayFree(up);
        }
        try self.gate_weight.forward(&gate, x);
        try mlx.sigmoid(&sigmoid, gate, self.stream);
        try mlx.multiply(&gate, gate, sigmoid, self.stream);
        try self.up_weight.forward(&up, x);
        try mlx.multiply(&up, gate, up, self.stream);
        try self.down_weight.forward(result, up);
    }
};

pub const Attention = struct {
    const Self = @This();
    n_heads: c_int,
    n_kv_heads: c_int,
    head_dim: c_int,
    n_repeat: c_int,
    scale: mlx.Array,
    q_weight: *mlx.Linear,
    k_weight: *mlx.Linear,
    v_weight: *mlx.Linear,
    o_weight: *mlx.Linear,
    rope: Llama3RoPE,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,
    allocs_to_free: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, n_heads: c_int, n_kv_heads: c_int, head_dim: c_int, rope_theta: f32, rope_scaling_config: LlamaConfig.RopeScalingConfig, quant_config: ?mlx.QuantConfig, weights_hash: *std.StringHashMap(*mlx.Array), stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        const scale = mlx.arrayNewFloat(1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))));
        const n_repeat = @divExact(n_heads, n_kv_heads);
        const rope = try Llama3RoPE.init(head_dim, rope_theta, rope_scaling_config, stream);
        var allocs_to_free = std.ArrayList([]const u8).init(allocator);
        try allocs_to_free.append(key);

        return .{
            .q_weight = try mlx.Linear.init(allocator, key, "q_proj", false, quant_config, weights_hash, stream),
            .k_weight = try mlx.Linear.init(allocator, key, "k_proj", false, quant_config, weights_hash, stream),
            .v_weight = try mlx.Linear.init(allocator, key, "v_proj", false, quant_config, weights_hash, stream),
            .o_weight = try mlx.Linear.init(allocator, key, "o_proj", false, quant_config, weights_hash, stream),
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .n_repeat = n_repeat,
            .scale = scale,
            .rope = rope,
            .stream = stream,
            .allocator = allocator,
            .allocs_to_free = allocs_to_free,
        };
    }

    pub fn deinit(self: *Self) void {
        self.q_weight.deinit();
        self.k_weight.deinit();
        self.v_weight.deinit();
        self.o_weight.deinit();
        self.rope.deinit();
        mlx.arrayFree(self.scale);
        for (self.allocs_to_free.items) |key| self.allocator.free(key);
        self.allocs_to_free.deinit();
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.KVCache, offset: c_int) !void {
        var q = mlx.arrayNew();
        var k = mlx.arrayNew();
        var v = mlx.arrayNew();
        var w = mlx.arrayNew();
        defer {
            mlx.arrayFree(q);
            mlx.arrayFree(k);
            mlx.arrayFree(v);
            mlx.arrayFree(w);
        }
        try self.q_weight.forward(&q, x);
        try self.k_weight.forward(&k, x);
        try self.v_weight.forward(&v, x);
        try mlx.rEshap(&q, q, "b l (h d) -> b h l d", .{ .h = self.n_heads, .d = self.head_dim }, self.stream);
        try mlx.rEshap(&k, k, "b l (h d) -> b h l d", .{ .h = self.n_kv_heads, .d = self.head_dim }, self.stream);
        try mlx.rEshap(&v, v, "b l (h d) -> b h l d", .{ .h = self.n_kv_heads, .d = self.head_dim }, self.stream);
        try self.rope.forward(&q, q, offset);
        try self.rope.forward(&k, k, offset);
        try mlx.multiply(&q, q, self.scale, self.stream);
        if (cache) |c| try c.update(&k, &v, offset, self.stream);
        try mlx.rEpeat(&k, k, "b h l d -> b (repeat h) l d", .{ .repeat = self.n_repeat }, self.stream);
        try mlx.rEpeat(&v, v, "b h l d -> b (repeat h) l d", .{ .repeat = self.n_repeat }, self.stream);
        try mlx.einsum(&w, .{ q, k }, "b h l d, b h k d -> b h l k", self.stream);
        if (mask) |m| try mlx.add(&w, w, m, self.stream);
        try mlx.softmax(&w, w, &.{3}, true, self.stream);
        try mlx.einsum(&w, .{ w, v }, "b h l k, b h k d -> b h l d", self.stream);
        try mlx.rEshap(&w, w, "b h l d -> b l (h d)", .{}, self.stream);
        try self.o_weight.forward(result, w);
    }
};

pub const Llama3RoPE = struct {
    const Self = @This();
    freqs: mlx.Array,
    rope_base: mlx.OptionalFloat,
    dims: c_int,
    stream: mlx.Stream,

    pub fn init(dims: c_int, theta: f32, scaling_config: LlamaConfig.RopeScalingConfig, stream: mlx.Stream) !Self {
        var freqs = mlx.arrayNew();
        var wavelens = mlx.arrayNew();
        var high_freq_mask = mlx.arrayNew();
        var mid_freq_mask = mlx.arrayNew();
        var high_freq = mlx.arrayNew();
        var smooth_factors = mlx.arrayNew();
        var mid_freq = mlx.arrayNew();
        defer {
            mlx.arrayFree(wavelens);
            mlx.arrayFree(high_freq_mask);
            mlx.arrayFree(mid_freq_mask);
            mlx.arrayFree(high_freq);
            mlx.arrayFree(smooth_factors);
            mlx.arrayFree(mid_freq);
        }
        try mlx.arange(&freqs, 0, @floatFromInt(dims), 2, mlx.DTYPE.FLOAT32, stream);
        try mlx.divide(&freqs, freqs, mlx.float(@floatFromInt(dims)), stream);
        try mlx.power(&freqs, mlx.float(theta), freqs, stream);
        try mlx.multiply(&wavelens, mlx.float(2.0 * std.math.pi), freqs, stream);
        try mlx.multiply(&high_freq, freqs, mlx.float(scaling_config.factor), stream);
        try mlx.greater(&high_freq_mask, wavelens, mlx.float(scaling_config.original_max_position_embeddings / scaling_config.low_freq_factor), stream);
        try mlx.where(&high_freq, high_freq_mask, high_freq, freqs, stream);
        try mlx.lessEqual(&mid_freq_mask, wavelens, mlx.float(scaling_config.original_max_position_embeddings / scaling_config.high_freq_factor), stream);
        try mlx.logicalOr(&mid_freq_mask, high_freq_mask, mid_freq_mask, stream);
        try mlx.logicalNot(&mid_freq_mask, mid_freq_mask, stream);
        try mlx.divide(&smooth_factors, mlx.float(scaling_config.original_max_position_embeddings), wavelens, stream);
        try mlx.subtract(&smooth_factors, smooth_factors, mlx.float(scaling_config.low_freq_factor), stream);
        try mlx.divide(&smooth_factors, smooth_factors, mlx.float(scaling_config.high_freq_factor - scaling_config.low_freq_factor), stream);
        try mlx.subtract(&mid_freq, mlx.float(1.0), smooth_factors, stream);
        try mlx.divide(&mid_freq, mid_freq, mlx.float(scaling_config.factor), stream);
        try mlx.add(&mid_freq, mid_freq, smooth_factors, stream);
        try mlx.divide(&mid_freq, freqs, mid_freq, stream);
        try mlx.where(&high_freq, high_freq_mask, high_freq, freqs, stream);
        return .{
            .freqs = freqs,
            .rope_base = mlx.OptionalFloat{ .has_value = false, .value = 0.0 },
            .dims = dims,
            .stream = stream,
        };
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, offset: c_int) !void {
        try mlx.fastRope(result, x, self.dims, false, self.rope_base, 1.0, offset, self.freqs, self.stream);
        try mlx.astype(result, result.*, mlx.DTYPE.BFLOAT16, self.stream);
    }

    pub fn deinit(self: *Self) void {
        mlx.arrayFree(self.freqs);
    }
};

pub const TransformerBlock = struct {
    const Self = @This();
    attention: Attention,
    mlp: MLP,
    input_layernorm: *mlx.RMSNorm,
    post_attention_layernorm: *mlx.RMSNorm,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,
    allocs_to_free: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, layer_idx: usize, config: *const LlamaConfig, weights_hash: *std.StringHashMap(*mlx.Array), stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, layer_idx);
        const attention = try Attention.init(allocator, key, "self_attn", config.num_attention_heads, config.num_key_value_heads, config.head_dim, config.rope_theta, config.rope_scaling, config.quantization, weights_hash, stream);
        const mlp = try MLP.init(allocator, key, "mlp", config.quantization, weights_hash, stream);
        const input_layernorm = try mlx.RMSNorm.init(allocator, key, "input_layernorm", config.rms_norm_eps, weights_hash, stream);
        const post_attention_layernorm = try mlx.RMSNorm.init(allocator, key, "post_attention_layernorm", config.rms_norm_eps, weights_hash, stream);
        var allocs_to_free = std.ArrayList([]const u8).init(allocator);
        try allocs_to_free.append(key);
        return .{
            .attention = attention,
            .mlp = mlp,
            .input_layernorm = input_layernorm,
            .post_attention_layernorm = post_attention_layernorm,
            .stream = stream,
            .allocator = allocator,
            .allocs_to_free = allocs_to_free,
        };
    }

    pub fn deinit(self: *Self) void {
        self.attention.deinit();
        self.mlp.deinit();
        self.input_layernorm.deinit();
        self.post_attention_layernorm.deinit();
        for (self.allocs_to_free.items) |key| self.allocator.free(key);
        self.allocs_to_free.deinit();
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.KVCache, offset: c_int) !void {
        var attn = mlx.arrayNew();
        var mlp = mlx.arrayNew();
        defer {
            mlx.arrayFree(attn);
            mlx.arrayFree(mlp);
        }
        try self.input_layernorm.forward(&attn, x);
        try self.attention.forward(&attn, attn, mask, cache, offset);
        try mlx.add(&attn, attn, x, self.stream);
        try self.post_attention_layernorm.forward(&mlp, attn);
        try self.mlp.forward(&mlp, mlp);
        try mlx.add(result, mlp, attn, self.stream);
    }
};

pub const LlamaModel = struct {
    const Self = @This();
    embed_tokens: *mlx.Embedding,
    layers: []TransformerBlock,
    norm: *mlx.RMSNorm,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,
    allocs_to_free: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, config: *const LlamaConfig, weights_hash: *std.StringHashMap(*mlx.Array), stream: mlx.Stream) !Self {
        const key = try allocator.dupe(u8, parent);
        const embed_tokens = try mlx.Embedding.init(allocator, key, "embed_tokens", config.quantization, weights_hash, stream);
        const norm = try mlx.RMSNorm.init(allocator, key, "norm", config.rms_norm_eps, weights_hash, stream);
        const layers = try allocator.alloc(TransformerBlock, @intCast(config.num_hidden_layers));
        const layers_key = try allocJoin(allocator, key, "layers");
        var allocs_to_free = std.ArrayList([]const u8).init(allocator);
        try allocs_to_free.append(key);
        try allocs_to_free.append(layers_key);
        for (0..@intCast(config.num_hidden_layers)) |i| {
            layers[i] = try TransformerBlock.init(allocator, layers_key, i, config, weights_hash, stream);
        }
        return .{
            .embed_tokens = embed_tokens,
            .layers = layers,
            .norm = norm,
            .stream = stream,
            .allocator = allocator,
            .allocs_to_free = allocs_to_free,
        };
    }

    pub fn deinit(self: *Self) void {
        self.embed_tokens.deinit();
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.norm.deinit();
        for (self.allocs_to_free.items) |key| self.allocator.free(key);
        self.allocs_to_free.deinit();
    }

    pub fn forward(self: *Self, result: *mlx.Array, toks: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.Cache) !void {
        const seq_len = mlx.arrayDim(toks, 1);
        const offset = if (cache) |c| c.offset else 0;
        var x = mlx.arrayNew();
        defer mlx.arrayFree(x);
        try self.embed_tokens.forward(&x, toks);
        for (self.layers, 0..) |*layer, i| {
            const layer_cache = if (cache) |c| &c.layers[i] else null;
            try layer.forward(&x, x, mask, layer_cache, offset);
        }
        try self.norm.forward(result, x);
        if (cache) |c| c.offset += seq_len;
    }
};

pub const Llama = struct {
    const Self = @This();
    model: LlamaModel,
    tie_word_embeddings: bool,
    lm_head: ?*mlx.Linear,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: *const LlamaConfig, weights_hash: *std.StringHashMap(*mlx.Array), stream: mlx.Stream) !Self {
        const model = try LlamaModel.init(allocator, "model", config, weights_hash, stream);
        const lm_head = if (!config.tie_word_embeddings) try mlx.Linear.init(allocator, "lm_head", "", false, config.quantization, weights_hash, stream) else null;
        return .{
            .model = model,
            .tie_word_embeddings = config.tie_word_embeddings,
            .lm_head = lm_head,
            .stream = stream,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        if (!self.tie_word_embeddings) self.lm_head.?.deinit();
    }

    pub fn forward(self: *Self, result: *mlx.Array, toks: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.Cache) !void {
        var x = mlx.arrayNew();
        defer mlx.arrayFree(x);
        try self.model.forward(&x, toks, mask, cache);
        if (self.tie_word_embeddings) {
            try self.model.embed_tokens.asLinear(result, x);
        } else {
            try self.lm_head.?.forward(result, x);
        }
    }
};

pub const Transformer = struct {
    const Self = @This();
    allocator: std.mem.Allocator,
    stream: mlx.Stream,
    model: Llama,
    eos_token_id: []u32,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        var buf: [1024]u8 = undefined;
        const stream = mlx.defaultGpuStreamNew();
        const path_config = try std.fmt.bufPrintZ(&buf, "{s}/config.json", .{model_path});
        const config = try loadJson(LlamaConfig, allocator, path_config, true);
        defer config.deinit();
        const eos_token_id = try allocator.dupe(u32, config.value.eos_token_id);
        const path_weight = try std.fmt.bufPrintZ(&buf, "{s}/model.safetensors", .{model_path});
        var safetensors = try mlx.Safetensors.load(path_weight, stream);
        defer safetensors.deinit();
        var weights_hash = std.StringHashMap(*mlx.Array).init(allocator);
        defer weights_hash.deinit();
        var model = try Llama.init(allocator, &config.value, &weights_hash, stream);
        errdefer model.deinit();
        try safetensors.unload(&weights_hash);
        return .{
            .allocator = allocator,
            .stream = stream,
            .model = model,
            .eos_token_id = eos_token_id,
        };
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        self.allocator.free(self.eos_token_id);
        mlx.streamFree(self.stream);
    }

    pub fn generate(self: *Self, initial_tokens: []const u32, num_tokens: usize) ![]u32 {
        std.debug.print("\nInput IDs: {any}\n\n", .{initial_tokens});
        var output_tokens = try self.allocator.alloc(u32, num_tokens);
        errdefer self.allocator.free(output_tokens);
        var cache = try mlx.Cache.init(self.allocator, self.model.model.layers.len, 2);
        var toks = try mlx.arrayNewData(initial_tokens.ptr, .{ 1, initial_tokens.len }, mlx.DTYPE.UINT32);
        var logits = mlx.arrayNew();
        var mask = mlx.arrayNew();
        defer {
            cache.deinit();
            mlx.arrayFree(toks);
            mlx.arrayFree(logits);
            mlx.arrayFree(mask);
        }
        var start_time = std.time.milliTimestamp();
        var prompt_ms: f16 = undefined;
        var i: usize = 0;
        while (i < num_tokens) : (i += 1) {
            try mlx.createCausalMask(&mask, mlx.arrayDim(toks, 1), cache.offset, mlx.DTYPE.BFLOAT16, self.stream);
            try self.model.forward(&logits, toks, mask, &cache);
            try mlx.take(&logits, logits, mlx.int(-1), 1, self.stream);
            try mlx.argmax(&logits, logits, 1, false, self.stream);
            try mlx.item(&output_tokens[i], logits);
            try mlx.arraySetData(&toks, &output_tokens[i], .{ 1, 1 }, mlx.DTYPE.UINT32);
            std.debug.print("Generated token {d}/{d}: {d}\n", .{ i + 1, num_tokens, output_tokens[i] });
            if (std.mem.indexOfScalar(u32, self.eos_token_id, output_tokens[i]) != null) {
                i += 1;
                break;
            }
            if (i == 0) {
                const current_time = std.time.milliTimestamp();
                prompt_ms = @floatFromInt(current_time - start_time);
                start_time = current_time;
            }
        }
        const end_time = std.time.milliTimestamp();
        if (i < num_tokens) {
            output_tokens = try self.allocator.realloc(output_tokens, i);
        }
        std.debug.print("\nOutput IDs: {any}\n", .{output_tokens});
        const prompt_tps = @as(f16, @floatFromInt(initial_tokens.len)) / (prompt_ms / 1000.0);
        std.debug.print("\nPrompt:     {d:.2} tokens-per-second ({d} tokens in {d:.2} ms)\n", .{ prompt_tps, initial_tokens.len, prompt_ms });
        if (i > 0) {
            const gen_ms = @as(f16, @floatFromInt(end_time - start_time));
            const gen_tps = @as(f16, @floatFromInt(i)) / (gen_ms / 1000.0);
            std.debug.print("Generation: {d:.2} tokens-per-second ({d} tokens in {d:.2} ms)\n", .{ gen_tps, i, gen_ms });
        }
        return output_tokens;
    }
};

const LlamaConfig = struct {
    eos_token_id: []u32,
    hidden_size: c_int = 2048,
    intermediate_size: c_int = 8192,
    num_attention_heads: c_int = 32,
    num_key_value_heads: c_int = 8,
    head_dim: c_int = 64,
    max_position_embeddings: c_int = 131072,
    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 500000.0,
    mlp_bias: bool = false,
    attention_bias: bool = false,
    tie_word_embeddings: bool = true,
    vocab_size: c_int = 128256,
    num_hidden_layers: c_int = 16,
    quantization: ?mlx.QuantConfig = null,
    rope_scaling: RopeScalingConfig,
    pub const RopeScalingConfig = struct {
        factor: f32 = 32.0,
        high_freq_factor: f32 = 4.0,
        low_freq_factor: f32 = 1.0,
        original_max_position_embeddings: f32 = 8192,
    };
};

test "Transformer generating" {
    std.debug.print("\n=== LLAMA.ZIG ===\n\n", .{});
    const allocator = std.testing.allocator;
    const initial_tokens = [_]u32{ 9906, 1917 };
    const num_tokens_to_generate = 10;
    var transformer = try Transformer.init(allocator, "Llama-3.2-1B-Instruct-4bit");
    defer transformer.deinit();
    const generated_tokens = try transformer.generate(&initial_tokens, num_tokens_to_generate);
    defer allocator.free(generated_tokens);
    std.debug.print("\nGenerated sequence: ", .{});
    for (generated_tokens) |token| {
        std.debug.print("{d} ", .{token});
    }
    std.debug.print("\n", .{});
}
