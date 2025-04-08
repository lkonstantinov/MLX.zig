//! phi.zig - Phi-4
//!
//! Copyright 2025

const std = @import("std");
const mlx = @import("mlx.zig");
const loadJson = @import("utils.zig").loadJson;
const allocJoin = @import("utils.zig").allocJoin;

pub const MLP = struct {
    const Self = @This();
    base: mlx.Module,
    gate_up_proj: *mlx.Linear,
    down_proj: *mlx.Linear,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, model_config: *const PhiConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        var self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .gate_up_proj = undefined,
            .down_proj = undefined,
        };
        const u_key = try self.base.allocJoin(key, "gate_up_proj");
        self.gate_up_proj = try mlx.Linear.init(mlx_config, u_key, false, model_config.quantization, weights_hash);
        const d_key = try self.base.allocJoin(key, "down_proj");
        self.down_proj = try mlx.Linear.init(mlx_config, d_key, false, model_config.quantization, weights_hash);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.gate_up_proj.deinit();
        self.down_proj.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        var gate_up = mlx.arrayNew();
        var gate = mlx.arrayNew();
        var up = mlx.arrayNew();
        defer {
            mlx.arrayFree(gate_up);
            mlx.arrayFree(gate);
            mlx.arrayFree(up);
        }
        try self.gate_up_proj.forward(&gate_up, x);
        try mlx.splitEqualParts(&.{ &gate, &up }, gate_up, 2, 2, self.base.stream);
        try mlx.silu(&gate, gate, self.base.stream);
        try mlx.multiply(&up, gate, up, self.base.stream);
        try self.down_proj.forward(result, up);
    }
};

pub const Attention = struct {
    const Self = @This();
    base: mlx.Module,
    n_heads: c_int,
    n_kv_heads: c_int,
    head_dim: c_int,
    scale: f32,
    q_pos: c_int,
    k_pos: c_int,
    qkv_proj: *mlx.Linear,
    o_proj: *mlx.Linear,
    rope: *RoPE,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, model_config: *const PhiConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        var self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .n_heads = model_config.num_attention_heads,
            .n_kv_heads = model_config.num_key_value_heads orelse model_config.num_attention_heads,
            .head_dim = @divExact(model_config.hidden_size, model_config.num_attention_heads),
            .scale = undefined,
            .q_pos = undefined,
            .k_pos = undefined,
            .qkv_proj = undefined,
            .o_proj = undefined,
            .rope = undefined,
        };

        self.scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        self.q_pos = self.n_heads * self.head_dim;
        self.k_pos = self.q_pos + self.n_kv_heads * self.head_dim;
        self.rope = try RoPE.init(mlx_config, self.head_dim, model_config.rope_theta);

        const qkv_key = try self.base.allocJoin(key, "qkv_proj");
        self.qkv_proj = try mlx.Linear.init(mlx_config, qkv_key, false, model_config.quantization, weights_hash);
        const o_key = try self.base.allocJoin(key, "o_proj");
        self.o_proj = try mlx.Linear.init(mlx_config, o_key, false, model_config.quantization, weights_hash);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.qkv_proj.deinit();
        self.o_proj.deinit();
        self.rope.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.KVCache, offset: c_int) !void {
        var qkv = mlx.arrayNew();
        var q = mlx.arrayNew();
        var k = mlx.arrayNew();
        var v = mlx.arrayNew();
        defer {
            mlx.arrayFree(qkv);
            mlx.arrayFree(q);
            mlx.arrayFree(k);
            mlx.arrayFree(v);
        }
        try self.qkv_proj.forward(&qkv, x);
        try mlx.split(&.{ &q, &k, &v }, qkv, &[_]c_int{ self.q_pos, self.k_pos }, 2, self.base.stream);
        try mlx.rEshap(&q, q, "b l (h d) -> b h l d", .{ .h = self.n_heads, .d = self.head_dim }, self.base.stream);
        try mlx.rEshap(&k, k, "b l (h d) -> b h l d", .{ .h = self.n_kv_heads, .d = self.head_dim }, self.base.stream);
        try mlx.rEshap(&v, v, "b l (h d) -> b h l d", .{ .h = self.n_kv_heads, .d = self.head_dim }, self.base.stream);
        try self.rope.forward(&q, q, offset);
        try self.rope.forward(&k, k, offset);
        if (cache) |c| try c.update(&k, &v, null, self.base.stream);
        try mlx.fastScaledDotProductAttention(result, q, k, v, self.scale, mask, self.base.stream);
        try mlx.rEshap(result, result.*, "b h l d -> b l (h d)", .{}, self.base.stream);
        try self.o_proj.forward(result, result.*);
    }
};

pub const RoPE = struct {
    const Self = @This();
    base: mlx.Module,
    dims: c_int,
    base_val: mlx.OptionalFloat,
    freqs: mlx.Array,

    pub fn init(mlx_config: mlx.MLXConfig, dims: c_int, base_: f32) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .dims = dims,
            .base_val = mlx.OptionalFloat{ .has_value = true, .value = base_ },
            .freqs = mlx.C.mlx_array_empty,
        };
        return self;
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, offset: c_int) !void {
        try mlx.fastRope(result, x, self.dims, false, self.base_val, 1.0, offset, self.freqs, self.base.stream);
    }

    pub fn deinit(self: *Self) void {
        self.base.deinit();
        self.base.allocator.destroy(self);
    }
};

pub const TransformerBlock = struct {
    const Self = @This();
    base: mlx.Module,
    self_attn: *Attention,
    mlp: *MLP,
    input_layernorm: *mlx.RMSNorm,
    post_attention_layernorm: *mlx.RMSNorm,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, model_config: *const PhiConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        var self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .self_attn = undefined,
            .mlp = undefined,
            .input_layernorm = undefined,
            .post_attention_layernorm = undefined,
        };
        const a_key = try self.base.allocJoin(key, "self_attn");
        self.self_attn = try Attention.init(mlx_config, a_key, model_config, weights_hash);
        const m_key = try self.base.allocJoin(key, "mlp");
        self.mlp = try MLP.init(mlx_config, m_key, model_config, weights_hash);
        const i_key = try self.base.allocJoin(key, "input_layernorm");
        self.input_layernorm = try mlx.RMSNorm.init(mlx_config, i_key, model_config.rms_norm_eps, weights_hash);
        const p_key = try self.base.allocJoin(key, "post_attention_layernorm");
        self.post_attention_layernorm = try mlx.RMSNorm.init(mlx_config, p_key, model_config.rms_norm_eps, weights_hash);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.self_attn.deinit();
        self.mlp.deinit();
        self.input_layernorm.deinit();
        self.post_attention_layernorm.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
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
        try mlx.add(&attn_output, x, attn_output, self.base.stream);
        try self.post_attention_layernorm.forward(&mlp_input, attn_output);
        try self.mlp.forward(&mlp_output, mlp_input);
        try mlx.add(result, attn_output, mlp_output, self.base.stream);
    }
};

pub const Phi4Model = struct {
    const Self = @This();
    base: mlx.Module,
    embed_tokens: *mlx.Embedding,
    layers: []*TransformerBlock,
    norm: *mlx.RMSNorm,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, model_config: *const PhiConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        var self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .embed_tokens = undefined,
            .layers = undefined,
            .norm = undefined,
        };
        const e_key = try self.base.allocJoin(key, "embed_tokens");
        self.embed_tokens = try mlx.Embedding.init(mlx_config, e_key, model_config.quantization, weights_hash);
        const l_key = try self.base.allocJoin(key, "layers");
        self.layers = try mlx_config.allocator.alloc(*TransformerBlock, @intCast(model_config.num_hidden_layers));
        for (0..@intCast(model_config.num_hidden_layers)) |i| {
            const i_key = try self.base.allocJoin(l_key, i);
            self.layers[i] = try TransformerBlock.init(mlx_config, i_key, model_config, weights_hash);
        }
        const n_key = try self.base.allocJoin(key, "norm");
        self.norm = try mlx.RMSNorm.init(mlx_config, n_key, model_config.rms_norm_eps, weights_hash);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.embed_tokens.deinit();
        for (self.layers) |layer| {
            layer.deinit();
        }
        self.base.allocator.free(self.layers);
        self.norm.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, inputs: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.Cache) !void {
        const offset = if (cache) |c| c.offset else 0;
        var h = mlx.arrayNew();
        defer mlx.arrayFree(h);
        try self.embed_tokens.forward(&h, inputs);
        if (cache) |c| {
            for (self.layers, 0..) |layer, i| {
                const layer_cache = &c.layers[i];
                try layer.forward(&h, h, mask, layer_cache, offset);
            }
            c.offset += mlx.arrayDim(inputs, 1);
        } else {
            for (self.layers) |layer| {
                try layer.forward(&h, h, mask, null, offset);
            }
        }
        try self.norm.forward(result, h);
    }
};

pub const Model = struct {
    const Self = @This();
    base: mlx.Module,
    model: *Phi4Model,
    tie_word_embeddings: bool,
    lm_head: ?*mlx.Linear,

    pub fn init(mlx_config: mlx.MLXConfig, model_config: *const PhiConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        var self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .tie_word_embeddings = model_config.tie_word_embeddings,
            .model = undefined,
            .lm_head = undefined,
        };
        self.model = try Phi4Model.init(mlx_config, "model", model_config, weights_hash);
        self.lm_head = if (!model_config.tie_word_embeddings) try mlx.Linear.init(mlx_config, "lm_head", false, model_config.quantization, weights_hash) else null;
        return self;
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
        self.base.deinit();
        self.base.allocator.destroy(self);
    }
};

pub const Transformer = struct {
    const Self = @This();
    mlx_config: mlx.MLXConfig,
    model: *Model,
    eos_token_id: []u32,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        var buf: [1024]u8 = undefined;
        var mlx_config = try mlx.MLXConfig.init(allocator);
        errdefer mlx_config.deinit();
        const path_config = try std.fmt.bufPrintZ(&buf, "{s}/config.json", .{model_path});
        const model_config = try loadJson(PhiConfig, allocator, path_config, true);
        defer model_config.deinit();
        const eos_token_id = try allocator.dupe(u32, &[_]u32{100265});
        errdefer allocator.free(eos_token_id);
        const path_weight_1 = try std.fmt.bufPrintZ(&buf, "{s}/model-00001-of-00002.safetensors", .{model_path});
        var safetensors = try mlx.Safetensors.load(path_weight_1, mlx_config.stream);
        defer safetensors.deinit();
        const path_weight_2 = try std.fmt.bufPrintZ(&buf, "{s}/model-00002-of-00002.safetensors", .{model_path});
        try safetensors.add(&[_][:0]const u8{path_weight_2}, allocator);
        var weights_hash = std.StringHashMap(*mlx.Array).init(allocator);
        defer weights_hash.deinit();
        var model = try Model.init(mlx_config, &model_config.value, &weights_hash);
        errdefer model.deinit();
        try safetensors.unload(&weights_hash);
        return .{
            .mlx_config = mlx_config,
            .model = model,
            .eos_token_id = eos_token_id,
        };
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        self.mlx_config.allocator.free(self.eos_token_id);
        self.mlx_config.deinit();
    }

    pub fn generate(self: *Self, initial_tokens: []const u32, num_tokens: usize) ![]u32 {
        std.debug.print("\nInput IDs: {any}\n\n", .{initial_tokens});
        var output_tokens = try self.mlx_config.allocator.alloc(u32, num_tokens);
        errdefer self.mlx_config.allocator.free(output_tokens);
        var cache = try mlx.Cache.init(self.mlx_config.allocator, self.model.model.layers.len, 2);
        defer cache.deinit();
        var toks = try mlx.arrayNewData(initial_tokens.ptr, .{ 1, initial_tokens.len }, mlx.UINT32);
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
            try mlx.createCausalMask(&mask, mlx.arrayDim(toks, 1), cache.offset, mlx.FLOAT16, self.mlx_config.stream);
            try self.model.forward(&logits, toks, mask, &cache);
            try mlx.take(&logits, logits, mlx.int(-1), 1, self.mlx_config.stream);
            try mlx.argmax(&logits, logits, 1, false, self.mlx_config.stream);
            try mlx.item(&output_tokens[i], logits);
            try mlx.arraySetData(&toks, &output_tokens[i], .{ 1, 1 }, mlx.UINT32);
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
            output_tokens = try self.mlx_config.allocator.realloc(output_tokens, i);
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

pub const PhiConfig = struct {
    bos_token_id: c_int = 100257,
    eos_token_id: c_int = 100265,
    hidden_size: c_int = 5120,
    num_hidden_layers: c_int = 40,
    intermediate_size: c_int = 17920,
    num_attention_heads: c_int = 40,
    num_key_value_heads: ?c_int = 10,
    rms_norm_eps: f32 = 1e-5,
    vocab_size: c_int = 100352,
    rope_theta: f32 = 250000.0,
    rope_traditional: bool = false,
    partial_rotary_factor: f32 = 1.0,
    max_position_embeddings: c_int = 16384,
    original_max_position_embeddings: c_int = 16384,
    tie_word_embeddings: bool = false,
    quantization: ?mlx.QuantConfig = null,
    rope_scaling: ?RopeScalingConfig = null,

    pub const RopeScalingConfig = struct {
        long_factor: f32 = 1.0,
        type: []const u8 = "linear",
    };
};
