//! qwen.zig - Qwen 2.5 (Coder, Olympic)
//!
//! Copyright 2025

const std = @import("std");
const mlx = @import("mlx.zig");
const utils = @import("utils.zig");

pub const MLP = struct {
    const Self = @This();
    base: mlx.Module,
    gate_proj: *mlx.Linear,
    down_proj: *mlx.Linear,
    up_proj: *mlx.Linear,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, quant_config: ?mlx.QuantConfig, weights: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .gate_proj = undefined,
            .down_proj = undefined,
            .up_proj = undefined,
        };
        const gate_key = try self.base.allocJoin(key, "gate_proj");
        self.gate_proj = try mlx.Linear.init(mlx_config, gate_key, false, quant_config, weights);
        const down_key = try self.base.allocJoin(key, "down_proj");
        self.down_proj = try mlx.Linear.init(mlx_config, down_key, false, quant_config, weights);
        const up_key = try self.base.allocJoin(key, "up_proj");
        self.up_proj = try mlx.Linear.init(mlx_config, up_key, false, quant_config, weights);
        return self;
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        var gate = mlx.arrayNew();
        var up = mlx.arrayNew();
        defer {
            mlx.arrayFree(gate);
            mlx.arrayFree(up);
        }
        try self.gate_proj.forward(&gate, x);
        try self.up_proj.forward(&up, x);
        try mlx.silu(&gate, gate, self.base.stream);
        try mlx.multiply(&gate, gate, up, self.base.stream);
        try self.down_proj.forward(result, gate);
    }

    pub fn deinit(self: *Self) void {
        self.gate_proj.deinit();
        self.down_proj.deinit();
        self.up_proj.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }
};

pub const Attention = struct {
    const Self = @This();
    base: mlx.Module,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    q_proj: *mlx.Linear,
    k_proj: *mlx.Linear,
    v_proj: *mlx.Linear,
    o_proj: *mlx.Linear,
    rope: *RoPE,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, model_config: *const Config, weights: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .n_heads = model_config.num_attention_heads,
            .n_kv_heads = model_config.num_key_value_heads,
            .head_dim = @divExact(model_config.hidden_size, model_config.num_attention_heads),
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(@divExact(model_config.hidden_size, model_config.num_attention_heads)))),
            .q_proj = undefined,
            .k_proj = undefined,
            .v_proj = undefined,
            .o_proj = undefined,
            .rope = undefined,
        };
        const q_key = try self.base.allocJoin(key, "q_proj");
        self.q_proj = try mlx.Linear.init(mlx_config, q_key, true, model_config.quantization, weights);
        const k_key = try self.base.allocJoin(key, "k_proj");
        self.k_proj = try mlx.Linear.init(mlx_config, k_key, true, model_config.quantization, weights);
        const v_key = try self.base.allocJoin(key, "v_proj");
        self.v_proj = try mlx.Linear.init(mlx_config, v_key, true, model_config.quantization, weights);
        const o_key = try self.base.allocJoin(key, "o_proj");
        self.o_proj = try mlx.Linear.init(mlx_config, o_key, false, model_config.quantization, weights);
        self.rope = try RoPE.init(mlx_config, self.head_dim, model_config.rope_theta, model_config.rope_scaling);
        return self;
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.KVCache, offset: i32) !void {
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

    pub fn deinit(self: *Self) void {
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.o_proj.deinit();
        self.rope.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }
};

pub const RoPE = struct {
    const Self = @This();
    base: mlx.Module,
    dims: i32,
    base_val: mlx.OptionalFloat,
    freqs: mlx.Array,
    rope_scaling: ?Config.RopeScalingConfig,

    pub fn init(mlx_config: mlx.MLXConfig, dims: i32, base_val: f32, scaling_config: ?Config.RopeScalingConfig) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .dims = dims,
            .base_val = .{ .has_value = false, .value = 0.0 },
            .freqs = mlx.arrayNew(),
            .rope_scaling = scaling_config,
        };
        try mlx.arange(&self.freqs, 0, @floatFromInt(dims), 2, mlx.FLOAT32, mlx_config.stream);
        try mlx.divide(&self.freqs, self.freqs, mlx.float(@floatFromInt(dims)), mlx_config.stream);
        try mlx.power(&self.freqs, mlx.float(base_val), self.freqs, mlx_config.stream);
        return self;
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, offset: i32) !void {
        try mlx.fastRope(result, x, self.dims, false, self.base_val, 1.0, offset, self.freqs, self.base.stream);
    }

    pub fn deinit(self: *Self) void {
        mlx.arrayFree(self.freqs);
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

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, model_config: *const Config, weights: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .self_attn = undefined,
            .mlp = undefined,
            .input_layernorm = undefined,
            .post_attention_layernorm = undefined,
        };
        const attn_key = try self.base.allocJoin(key, "self_attn");
        self.self_attn = try Attention.init(mlx_config, attn_key, model_config, weights);
        const mlp_key = try self.base.allocJoin(key, "mlp");
        self.mlp = try MLP.init(mlx_config, mlp_key, model_config.quantization, weights);
        const input_ln_key = try self.base.allocJoin(key, "input_layernorm");
        self.input_layernorm = try mlx.RMSNorm.init(mlx_config, input_ln_key, model_config.rms_norm_eps, weights);
        const post_ln_key = try self.base.allocJoin(key, "post_attention_layernorm");
        self.post_attention_layernorm = try mlx.RMSNorm.init(mlx_config, post_ln_key, model_config.rms_norm_eps, weights);
        return self;
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.KVCache, offset: i32) !void {
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

    pub fn deinit(self: *Self) void {
        self.self_attn.deinit();
        self.mlp.deinit();
        self.input_layernorm.deinit();
        self.post_attention_layernorm.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }
};

pub const QwenModel = struct {
    const Self = @This();
    base: mlx.Module,
    embed_tokens: *mlx.Embedding,
    layers: []*TransformerBlock,
    norm: *mlx.RMSNorm,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, model_config: *const Config, weights: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .embed_tokens = undefined,
            .layers = undefined,
            .norm = undefined,
        };
        const embed_key = try self.base.allocJoin(key, "embed_tokens");
        self.embed_tokens = try mlx.Embedding.init(mlx_config, embed_key, model_config.quantization, weights);
        const layers_key = try self.base.allocJoin(key, "layers");
        self.layers = try mlx_config.allocator.alloc(*TransformerBlock, @intCast(model_config.num_hidden_layers));
        for (0..@intCast(model_config.num_hidden_layers)) |i| {
            const layer_key = try self.base.allocJoin(layers_key, i);
            self.layers[i] = try TransformerBlock.init(mlx_config, layer_key, model_config, weights);
        }
        const norm_key = try self.base.allocJoin(key, "norm");
        self.norm = try mlx.RMSNorm.init(mlx_config, norm_key, model_config.rms_norm_eps, weights);
        return self;
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
};

pub const Model = struct {
    const Self = @This();
    base: mlx.Module,
    model: *QwenModel,
    tie_word_embeddings: bool,
    lm_head: ?*mlx.Linear,

    pub fn init(mlx_config: mlx.MLXConfig, model_config: *const Config, weights: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .tie_word_embeddings = model_config.tie_word_embeddings,
            .model = undefined,
            .lm_head = undefined,
        };
        self.model = try QwenModel.init(mlx_config, "model", model_config, weights);
        self.lm_head = if (!model_config.tie_word_embeddings)
            try mlx.Linear.init(mlx_config, "lm_head", false, model_config.quantization, weights)
        else
            null;
        return self;
    }

    pub fn forward(self: *Self, result: *mlx.Array, inputs: mlx.Array, mask: ?mlx.Array, cache: ?*mlx.Cache) !void {
        var out = mlx.arrayNew();
        defer mlx.arrayFree(out);
        try self.model.forward(&out, inputs, mask, cache);
        if (self.tie_word_embeddings) {
            try self.model.embed_tokens.asLinear(result, out);
        } else if (self.lm_head) |head| {
            try head.forward(result, out);
        }
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        if (self.lm_head) |head| {
            head.deinit();
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
        const config_json = try utils.loadJson(Config, allocator, path_config, true);
        defer config_json.deinit();
        const eos_token_id = try mlx_config.allocator.dupe(u32, &[_]u32{config_json.value.eos_token_id});
        errdefer mlx_config.allocator.free(eos_token_id);
        const path_weight = try std.fmt.bufPrintZ(&buf, "{s}/model.safetensors", .{model_path});
        var safetensors = try mlx.Safetensors.load(path_weight, mlx_config.stream);
        defer safetensors.deinit();
        var weights_hash = std.StringHashMap(*mlx.Array).init(allocator);
        defer weights_hash.deinit();
        var model = try Model.init(mlx_config, &config_json.value, &weights_hash);
        errdefer model.deinit();
        try safetensors.unload(&weights_hash);
        return .{
            .mlx_config = mlx_config,
            .model = model,
            .eos_token_id = eos_token_id,
        };
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
        var prompt_ms: f32 = undefined;
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
        const prompt_tps = @as(f32, @floatFromInt(initial_tokens.len)) / (prompt_ms / 1000.0);
        std.debug.print("\nPrompt:     {d:.2} tokens-per-second ({d} tokens in {d:.2} ms)\n", .{ prompt_tps, initial_tokens.len, prompt_ms });
        if (i > 0) {
            const gen_ms = @as(f32, @floatFromInt(end_time - start_time));
            const gen_tps = @as(f32, @floatFromInt(i)) / (gen_ms / 1000.0);
            std.debug.print("Generation: {d:.2} tokens-per-second ({d} tokens in {d:.2} ms)\n", .{ gen_tps, i, gen_ms });
        }
        return output_tokens;
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        self.mlx_config.allocator.free(self.eos_token_id);
        self.mlx_config.deinit();
        self.mlx_config.allocator.destroy(self);
    }
};

pub const Config = struct {
    bos_token_id: u32 = 151643,
    eos_token_id: u32 = 151645,
    hidden_size: i32 = 3584,
    num_hidden_layers: i32 = 28,
    intermediate_size: i32 = 18944,
    num_attention_heads: i32 = 28,
    num_key_value_heads: i32 = 4,
    max_position_embeddings: i32 = 32768,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 1000000.0,
    rope_traditional: bool = false,
    tie_word_embeddings: bool = false,
    vocab_size: i32 = 152064,
    quantization: ?mlx.QuantConfig = null,
    rope_scaling: ?RopeScalingConfig = null,
    pub const RopeScalingConfig = struct {
        factor: f32 = 1.0,
        type: []const u8 = "linear",
    };
};
