//! qwen.zig - Qwen 2.5 (Coder, Olympic)
//!
//! Copyright 2025

const std = @import("std");
const mlx = @import("mlx.zig");
const loadJson = @import("utils.zig").loadJson;
const allocJoin = @import("utils.zig").allocJoin;

pub const RoPE = struct {
    const Self = @This();
    dims: c_int,
    base: mlx.OptionalFloat,
    freqs: mlx.Array,
    stream: mlx.Stream,
    rope_scaling: ?QwenConfig.RopeScalingConfig,

    pub fn init(dims: c_int, base_: f32, scaling_config: ?QwenConfig.RopeScalingConfig, max_position_embeddings: c_int, stream: mlx.Stream) !Self {
        _ = max_position_embeddings;
        var freqs = mlx.arrayNew();
        try mlx.arange(&freqs, 0, @floatFromInt(dims), 2, mlx.DTYPE.FLOAT32, stream);
        try mlx.divide(&freqs, freqs, mlx.float(@floatFromInt(dims)), stream);
        try mlx.power(&freqs, mlx.float(base_), freqs, stream);
        const base = mlx.OptionalFloat{ .has_value = false, .value = 0.0 };
        return Self{
            .dims = dims,
            .base = base,
            .freqs = freqs,
            .stream = stream,
            .rope_scaling = scaling_config,
        };
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, offset: c_int) !void {
        try mlx.fastRope(result, x, self.dims, false, self.base, 1.0, offset, self.freqs, self.stream);
    }

    pub fn deinit(self: *Self) void {
        mlx.arrayFree(self.freqs);
    }
};

pub const Attention = struct {
    const Self = @This();
    key: []const u8,
    n_heads: c_int,
    n_kv_heads: c_int,
    head_dim: c_int,
    scale: f32,
    q_proj: mlx.Linear,
    k_proj: mlx.Linear,
    v_proj: mlx.Linear,
    o_proj: mlx.Linear,
    rope: RoPE,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, config: *const QwenConfig, stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        const n_heads = config.num_attention_heads;
        const n_kv_heads = config.num_key_value_heads;
        const head_dim = @divExact(config.hidden_size, n_heads);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        const q_proj = try mlx.Linear.init(allocator, key, "q_proj", true, config.quantization, stream);
        const k_proj = try mlx.Linear.init(allocator, key, "k_proj", true, config.quantization, stream);
        const v_proj = try mlx.Linear.init(allocator, key, "v_proj", true, config.quantization, stream);
        const o_proj = try mlx.Linear.init(allocator, key, "o_proj", false, config.quantization, stream);

        const rope = try RoPE.init(head_dim, config.rope_theta, config.rope_scaling, config.max_position_embeddings, stream);

        return Self{
            .key = key,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .scale = scale,
            .q_proj = q_proj,
            .k_proj = k_proj,
            .v_proj = v_proj,
            .o_proj = o_proj,
            .rope = rope,
            .stream = stream,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try self.q_proj.load(weights_map);
        try self.k_proj.load(weights_map);
        try self.v_proj.load(weights_map);
        try self.o_proj.load(weights_map);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.KVCache, offset: c_int) !void {
        var q = mlx.arrayNew();
        var k = mlx.arrayNew();
        var v = mlx.arrayNew();
        defer {
            mlx.arrayFree(q);
            mlx.arrayFree(k);
            mlx.arrayFree(v);
        }
        try self.q_proj.forward(&q, x);
        try self.k_proj.forward(&k, x);
        try self.v_proj.forward(&v, x);
        try mlx.rEshap(&q, q, "b l (h d) -> b h l d", .{ .h = self.n_heads, .d = self.head_dim }, self.stream);
        try mlx.rEshap(&k, k, "b l (h d) -> b h l d", .{ .h = self.n_kv_heads, .d = self.head_dim }, self.stream);
        try mlx.rEshap(&v, v, "b l (h d) -> b h l d", .{ .h = self.n_kv_heads, .d = self.head_dim }, self.stream);
        try self.rope.forward(&q, q, offset);
        try self.rope.forward(&k, k, offset);
        if (cache) |c| try c.update(&k, &v, null, self.stream);
        try mlx.fastScaledDotProductAttention(result, q, k, v, self.scale, mask, self.stream);
        try mlx.rEshap(result, result.*, "b h l d -> b l (h d)", .{}, self.stream);
        try self.o_proj.forward(result, result.*);
    }

    pub fn deinit(self: *Self) void {
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.o_proj.deinit();
        self.rope.deinit();
        self.allocator.free(self.key);
    }
};

pub const MLP = struct {
    const Self = @This();
    key: []const u8,
    gate_proj: mlx.Linear,
    down_proj: mlx.Linear,
    up_proj: mlx.Linear,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, quant_config: ?mlx.QuantConfig, stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        const gate_proj = try mlx.Linear.init(allocator, key, "gate_proj", false, quant_config, stream);
        const down_proj = try mlx.Linear.init(allocator, key, "down_proj", false, quant_config, stream);
        const up_proj = try mlx.Linear.init(allocator, key, "up_proj", false, quant_config, stream);
        return Self{
            .key = key,
            .gate_proj = gate_proj,
            .down_proj = down_proj,
            .up_proj = up_proj,
            .stream = stream,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try self.gate_proj.load(weights_map);
        try self.down_proj.load(weights_map);
        try self.up_proj.load(weights_map);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        var gate = mlx.arrayNew();
        var up = mlx.arrayNew();
        // var silu = mlx.arrayNew();
        defer {
            mlx.arrayFree(gate);
            mlx.arrayFree(up);
            // mlx.arrayFree(silu);
        }
        try self.gate_proj.forward(&gate, x);
        try self.up_proj.forward(&up, x);
        // try mlx.sigmoid(&silu, gate, self.stream);
        // try mlx.multiply(&gate, gate, silu, self.stream);
        try mlx.silu(&gate, gate, self.stream);
        try mlx.multiply(&gate, gate, up, self.stream);
        try self.down_proj.forward(result, gate);
    }

    pub fn deinit(self: *Self) void {
        self.gate_proj.deinit();
        self.down_proj.deinit();
        self.up_proj.deinit();
        self.allocator.free(self.key);
    }
};

pub const TransformerBlock = struct {
    const Self = @This();
    key: []const u8,
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: mlx.RMSNorm,
    post_attention_layernorm: mlx.RMSNorm,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, layer_idx: usize, config: *const QwenConfig, stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, layer_idx);
        errdefer allocator.free(key);
        const self_attn = try Attention.init(allocator, key, "self_attn", config, stream);
        const mlp = try MLP.init(allocator, key, "mlp", config.quantization, stream);
        const input_layernorm = try mlx.RMSNorm.init(allocator, key, "input_layernorm", config.rms_norm_eps, stream);
        const post_attention_layernorm = try mlx.RMSNorm.init(allocator, key, "post_attention_layernorm", config.rms_norm_eps, stream);
        return Self{
            .key = key,
            .self_attn = self_attn,
            .mlp = mlp,
            .input_layernorm = input_layernorm,
            .post_attention_layernorm = post_attention_layernorm,
            .stream = stream,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try self.self_attn.load(weights_map);
        try self.mlp.load(weights_map);
        try self.input_layernorm.load(weights_map);
        try self.post_attention_layernorm.load(weights_map);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.KVCache, offset: c_int) !void {
        var attn_input = mlx.arrayNew();
        var attn_output = mlx.arrayNew();
        var mlp_input = mlx.arrayNew();
        var mlp_output = mlx.arrayNew();
        defer {
            mlx.arrayFree(attn_input);
            mlx.arrayFree(attn_output);
            mlx.arrayFree(mlp_input);
            mlx.arrayFree(mlp_output);
        }
        try self.input_layernorm.forward(&attn_input, x);
        try self.self_attn.forward(&attn_output, attn_input, mask, cache, offset);
        try mlx.add(&attn_output, x, attn_output, self.stream);
        try self.post_attention_layernorm.forward(&mlp_input, attn_output);
        try self.mlp.forward(&mlp_output, mlp_input);
        try mlx.add(result, attn_output, mlp_output, self.stream);
    }

    pub fn deinit(self: *Self) void {
        self.self_attn.deinit();
        self.mlp.deinit();
        self.input_layernorm.deinit();
        self.post_attention_layernorm.deinit();
        self.allocator.free(self.key);
    }
};

pub const OlympicModel = struct {
    const Self = @This();
    key: []const u8,
    embed_tokens: mlx.Embedding,
    layers: []TransformerBlock,
    norm: mlx.RMSNorm,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, config: *const QwenConfig, stream: mlx.Stream) !Self {
        const key = try allocator.dupe(u8, parent);
        errdefer allocator.free(key);
        const embed_tokens = try mlx.Embedding.init(allocator, key, "embed_tokens", config.quantization, stream);
        const layers = try allocator.alloc(TransformerBlock, @intCast(config.num_hidden_layers));
        errdefer allocator.free(layers);
        const layers_key = try allocJoin(allocator, key, "layers");
        defer allocator.free(layers_key);
        for (0..@intCast(config.num_hidden_layers)) |i| {
            layers[i] = try TransformerBlock.init(allocator, layers_key, i, config, stream);
        }
        const norm = try mlx.RMSNorm.init(allocator, key, "norm", config.rms_norm_eps, stream);
        return Self{
            .key = key,
            .embed_tokens = embed_tokens,
            .layers = layers,
            .norm = norm,
            .stream = stream,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try self.embed_tokens.load(weights_map);
        for (self.layers) |*layer| {
            try layer.load(weights_map);
        }
        try self.norm.load(weights_map);
    }

    pub fn forward(self: *Self, result: *mlx.Array, inputs: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.Cache) !void {
        const offset = if (cache) |c| c.offset else 0;
        var h = mlx.arrayNew();
        defer mlx.arrayFree(h);
        try self.embed_tokens.forward(&h, inputs);
        if (cache) |c| {
            for (self.layers, 0..) |*layer, i| {
                const layer_cache = &c.layers[i];
                try layer.forward(&h, h, mask, layer_cache, offset);
            }
            c.offset += mlx.arrayDim(inputs, 1);
        } else {
            for (self.layers) |*layer| {
                try layer.forward(&h, h, mask, null, offset);
            }
        }
        try self.norm.forward(result, h);
    }

    pub fn deinit(self: *Self) void {
        self.embed_tokens.deinit();
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.norm.deinit();
        self.allocator.free(self.key);
    }
};

pub const Model = struct {
    const Self = @This();
    model: OlympicModel,
    tie_word_embeddings: bool,
    lm_head: ?mlx.Linear,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: *const QwenConfig, stream: mlx.Stream) !Self {
        const model = try OlympicModel.init(allocator, "model", config, stream);
        const lm_head = if (!config.tie_word_embeddings)
            try mlx.Linear.init(allocator, "lm_head", "", false, config.quantization, stream)
        else
            null;

        return Self{
            .model = model,
            .tie_word_embeddings = config.tie_word_embeddings,
            .lm_head = lm_head,
            .stream = stream,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try self.model.load(weights_map);
        if (!self.tie_word_embeddings and self.lm_head != null) {
            try self.lm_head.?.load(weights_map);
        }
    }

    pub fn forward(self: *Self, result: *mlx.Array, inputs: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.Cache) !void {
        var out = mlx.arrayNew();
        defer mlx.arrayFree(out);

        try self.model.forward(&out, inputs, mask, cache);

        if (self.tie_word_embeddings) {
            try self.model.embed_tokens.asLinear(result, out);
        } else {
            try self.lm_head.?.forward(result, out);
        }
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        if (!self.tie_word_embeddings and self.lm_head != null) {
            self.lm_head.?.deinit();
        }
    }
};

pub const Transformer = struct {
    const Self = @This();
    allocator: std.mem.Allocator,
    stream: mlx.Stream,
    model: Model,
    eos_token_id: []u32,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        var buf: [1024]u8 = undefined;
        const stream = mlx.defaultGpuStreamNew();
        const path_config = try std.fmt.bufPrintZ(&buf, "{s}/config.json", .{model_path});
        const config = try loadJson(QwenConfig, allocator, path_config, true);
        defer config.deinit();

        const eos_token_id = try allocator.dupe(u32, &[_]u32{config.value.eos_token_id});
        errdefer allocator.free(eos_token_id);

        var model = try Model.init(allocator, &config.value, stream);
        errdefer model.deinit();

        const path_weight = try std.fmt.bufPrintZ(&buf, "{s}/model.safetensors", .{model_path});
        var safetensors = try mlx.Safetensors.load(path_weight, stream);
        defer safetensors.deinit();

        try model.load(&safetensors.weights);

        return Self{
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
        defer cache.deinit();
        var toks = try mlx.arrayNewData(initial_tokens.ptr, .{ 1, initial_tokens.len }, mlx.DTYPE.UINT32);
        var logits = mlx.arrayNew();
        var mask = mlx.arrayNew();
        defer {
            mlx.arrayFree(toks);
            mlx.arrayFree(logits);
            mlx.arrayFree(mask);
        }
        var start_time = std.time.milliTimestamp();
        var prompt_ms: f16 = undefined;
        var i: usize = 0;
        while (i < num_tokens) : (i += 1) {
            try mlx.createCausalMask(&mask, mlx.arrayDim(toks, 1), cache.offset, mlx.DTYPE.FLOAT16, self.stream);
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

pub const QwenConfig = struct {
    bos_token_id: u32 = 151643,
    eos_token_id: u32 = 151645,
    hidden_size: c_int = 3584,
    num_hidden_layers: c_int = 28,
    intermediate_size: c_int = 18944,
    num_attention_heads: c_int = 28,
    num_key_value_heads: c_int = 4,
    max_position_embeddings: c_int = 32768,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 1000000.0,
    rope_traditional: bool = false,
    tie_word_embeddings: bool = false,
    vocab_size: c_int = 152064,
    quantization: ?mlx.QuantConfig = null,
    rope_scaling: ?RopeScalingConfig = null,

    pub const RopeScalingConfig = struct {
        factor: f32 = 1.0,
        type: []const u8 = "linear",
    };
};
