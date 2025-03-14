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
const c = @cImport({
    @cInclude("mlx/c/mlx.h");
    @cInclude("stdio.h");
});

pub const RopeScalingConfig = struct {
    factor: f32,
    high_freq_factor: f32,
    low_freq_factor: f32,
    original_max_position_embeddings: c_int,
};

pub const LlamaConfig = struct {
    hidden_size: c_int,
    intermediate_size: c_int,
    num_attention_heads: c_int,
    num_key_value_heads: c_int,
    head_dim: c_int,
    max_position_embeddings: c_int,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_scaling: RopeScalingConfig,
    mlp_bias: bool,
    attention_bias: bool,
    tie_word_embeddings: bool,
    vocab_size: c_int,
    num_hidden_layers: c_int,
};

pub const llamaConfig = LlamaConfig{
    .hidden_size = 2048,
    .intermediate_size = 8192,
    .num_attention_heads = 32,
    .num_key_value_heads = 8,
    .head_dim = 64,
    .max_position_embeddings = 131072,
    .rms_norm_eps = 1e-5,
    .rope_theta = 500000.0,
    .rope_scaling = .{
        .factor = 32.0,
        .high_freq_factor = 4.0,
        .low_freq_factor = 1.0,
        .original_max_position_embeddings = 8192,
    },
    .mlp_bias = false,
    .attention_bias = false,
    .tie_word_embeddings = true,
    .vocab_size = 128256,
    .num_hidden_layers = 16,
};

pub const Linear = struct {
    weight: c.mlx_array,
    stream: c.mlx_stream,

    pub fn init(weight: c.mlx_array, stream: c.mlx_stream) Linear {
        return Linear{
            .weight = weight,
            .stream = stream,
        };
    }

    pub fn load(loader: *WeightLoader, prefix: []const u8, stream: c.mlx_stream) !Linear {
        const weight = try loader.getComponentWeight(prefix, "weight");
        return Linear.init(weight, stream);
    }

    pub fn forward(self: *Linear, result: *c.mlx_array, x: c.mlx_array) !void {
        try einsumOp(result, "bsh,vh->bsv", &[_]c.mlx_array{ x, self.weight }, self.stream);
    }

    pub fn deinit(self: *Linear) void {
        _ = c.mlx_array_free(self.weight);
    }
};

pub const Embedding = struct {
    weight: c.mlx_array,
    stream: c.mlx_stream,

    pub fn init(weight: c.mlx_array, stream: c.mlx_stream) Embedding {
        return Embedding{
            .weight = weight,
            .stream = stream,
        };
    }

    pub fn load(loader: *WeightLoader, prefix: []const u8, stream: c.mlx_stream) !Embedding {
        const weight = try loader.getComponentWeight(prefix, "weight");
        return Embedding.init(weight, stream);
    }

    pub fn forward(self: *Embedding, result: *c.mlx_array, toks: c.mlx_array) !void {
        try mlxOp(c.mlx_take(result, self.weight, toks, 0, self.stream));
    }

    pub fn asLinear(self: *Embedding, result: *c.mlx_array, x: c.mlx_array) !void {
        try einsumOp(result, "blh,dh->bld", &[_]c.mlx_array{ x, self.weight }, self.stream);
    }

    pub fn deinit(self: *Embedding) void {
        _ = c.mlx_array_free(self.weight);
    }
};

pub const RMSNorm = struct {
    weight: c.mlx_array,
    eps: f32,
    stream: c.mlx_stream,

    pub fn init(weight: c.mlx_array, eps: f32, stream: c.mlx_stream) RMSNorm {
        return RMSNorm{
            .weight = weight,
            .eps = eps,
            .stream = stream,
        };
    }

    pub fn load(loader: *WeightLoader, prefix: []const u8, eps: f32, stream: c.mlx_stream) !RMSNorm {
        const weight = try loader.getComponentWeight(prefix, "weight");
        return RMSNorm.init(weight, eps, stream);
    }

    pub fn forward(self: *RMSNorm, result: *c.mlx_array, x: c.mlx_array) !void {
        try mlxOp(c.mlx_fast_rms_norm(result, x, self.weight, self.eps, self.stream));
    }

    pub fn deinit(self: *RMSNorm) void {
        _ = c.mlx_array_free(self.weight);
    }
};

pub const MLP = struct {
    gate_proj: c.mlx_array,
    up_proj: c.mlx_array,
    down_proj: c.mlx_array,
    stream: c.mlx_stream,

    pub fn init(gate_proj: c.mlx_array, up_proj: c.mlx_array, down_proj: c.mlx_array, stream: c.mlx_stream) MLP {
        return MLP{
            .gate_proj = gate_proj,
            .up_proj = up_proj,
            .down_proj = down_proj,
            .stream = stream,
        };
    }

    pub fn load(loader: *WeightLoader, prefix: []const u8, stream: c.mlx_stream) !MLP {
        const gate_proj = try loader.getComponentWeight(prefix, "gate_proj.weight");
        const up_proj = try loader.getComponentWeight(prefix, "up_proj.weight");
        const down_proj = try loader.getComponentWeight(prefix, "down_proj.weight");
        return MLP.init(gate_proj, up_proj, down_proj, stream);
    }

    pub fn forward(self: *MLP, result: *c.mlx_array, x: c.mlx_array) !void {
        var gate = c.mlx_array_new();
        var sigmoid = c.mlx_array_new();
        var up = c.mlx_array_new();
        defer {
            _ = c.mlx_array_free(gate);
            _ = c.mlx_array_free(sigmoid);
            _ = c.mlx_array_free(up);
        }
        try einsumOp(&gate, "bld,hd->blh", &[_]c.mlx_array{ x, self.gate_proj }, self.stream);
        try mlxOp(c.mlx_sigmoid(&sigmoid, gate, self.stream));
        try mlxOp(c.mlx_multiply(&gate, gate, sigmoid, self.stream));
        try einsumOp(&up, "bld,hd->blh", &[_]c.mlx_array{ x, self.up_proj }, self.stream);
        try mlxOp(c.mlx_multiply(result, gate, up, self.stream));
        try einsumOp(result, "blh,dh->bld", &[_]c.mlx_array{ result.*, self.down_proj }, self.stream);
    }

    pub fn deinit(self: *MLP) void {
        _ = c.mlx_array_free(self.gate_proj);
        _ = c.mlx_array_free(self.up_proj);
        _ = c.mlx_array_free(self.down_proj);
    }
};

pub const Attention = struct {
    q_proj: c.mlx_array,
    k_proj: c.mlx_array,
    v_proj: c.mlx_array,
    o_proj: c.mlx_array,
    n_heads: c_int,
    n_kv_heads: c_int,
    head_dim: c_int,
    n_repeat: c_int,
    scale_array: c.mlx_array,
    rope: Llama3RoPE,
    stream: c.mlx_stream,

    pub fn init(q_proj: c.mlx_array, k_proj: c.mlx_array, v_proj: c.mlx_array, o_proj: c.mlx_array, n_heads: c_int, n_kv_heads: c_int, head_dim: c_int, rope: Llama3RoPE, stream: c.mlx_stream) Attention {
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const scale_array = c.mlx_array_new_float(scale);
        const n_repeat = @divExact(n_heads, n_kv_heads);

        return Attention{
            .q_proj = q_proj,
            .k_proj = k_proj,
            .v_proj = v_proj,
            .o_proj = o_proj,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .n_repeat = n_repeat,
            .scale_array = scale_array,
            .rope = rope,
            .stream = stream,
        };
    }

    pub fn load(loader: *WeightLoader, prefix: []const u8, config: *const LlamaConfig, stream: c.mlx_stream) !Attention {
        const q_proj = try loader.getComponentWeight(prefix, "q_proj.weight");
        const k_proj = try loader.getComponentWeight(prefix, "k_proj.weight");
        const v_proj = try loader.getComponentWeight(prefix, "v_proj.weight");
        const o_proj = try loader.getComponentWeight(prefix, "o_proj.weight");
        const rope = try Llama3RoPE.init(config.head_dim, config.max_position_embeddings, false, config.rope_theta, config.rope_scaling, stream);
        const attention = Attention.init(q_proj, k_proj, v_proj, o_proj, config.num_attention_heads, config.num_key_value_heads, config.head_dim, rope, stream);
        return attention;
    }

    pub fn forward(self: *Attention, result: *c.mlx_array, x: c.mlx_array, mask: ?c.mlx_array, cache: ?*KVCache, offset: c_int) !void {
        var q = c.mlx_array_new();
        var k = c.mlx_array_new();
        var v = c.mlx_array_new();
        var w = c.mlx_array_new();
        defer {
            _ = c.mlx_array_free(q);
            _ = c.mlx_array_free(k);
            _ = c.mlx_array_free(v);
            _ = c.mlx_array_free(w);
        }
        const B = c.mlx_array_dim(x, 0);
        const L = c.mlx_array_dim(x, 1);
        const q_shape = [_]c_int{ B, L, self.n_heads, self.head_dim };
        const k_shape = [_]c_int{ B, L, self.n_kv_heads, self.head_dim };
        const input_dtype = c.mlx_array_dtype(x);
        try einsumOp(&q, "bld,fd->blf", &[_]c.mlx_array{ x, self.q_proj }, self.stream);
        try einsumOp(&k, "bld,fd->blf", &[_]c.mlx_array{ x, self.k_proj }, self.stream);
        try einsumOp(&v, "bld,fd->blf", &[_]c.mlx_array{ x, self.v_proj }, self.stream);
        try mlxOp(c.mlx_reshape(&q, q, &q_shape, 4, self.stream));
        try mlxOp(c.mlx_reshape(&k, k, &k_shape, 4, self.stream));
        try mlxOp(c.mlx_reshape(&v, v, &k_shape, 4, self.stream));
        try einsumOp(&q, "blhd->bhld", &[_]c.mlx_array{q}, self.stream);
        try einsumOp(&k, "blhd->bhld", &[_]c.mlx_array{k}, self.stream);
        try einsumOp(&v, "blhd->bhld", &[_]c.mlx_array{v}, self.stream);
        try self.rope.forward(&q, q, offset);
        try self.rope.forward(&k, k, offset);
        try mlxOp(c.mlx_multiply(&q, q, self.scale_array, self.stream));
        if (cache != null) {
            try cache.?.update(&k, &v, offset, self.stream);
        }
        if (self.n_repeat > 1) {
            try mlxOp(c.mlx_repeat(&k, k, self.n_repeat, 1, self.stream));
            try mlxOp(c.mlx_repeat(&v, v, self.n_repeat, 1, self.stream));
        }
        try einsumOp(&w, "bhld,bhkd->bhlk", &[_]c.mlx_array{ q, k }, self.stream);
        if (mask) |mask_val| {
            try mlxOp(c.mlx_add(&w, w, mask_val, self.stream));
        }
        try mlxOp(c.mlx_softmax(&w, w, &[_]c_int{3}, 1, true, self.stream));
        try einsumOp(&w, "bhlk,bhkd->bhld", &[_]c.mlx_array{ w, v }, self.stream);
        try einsumOp(&w, "bhld->blhd", &[_]c.mlx_array{w}, self.stream);
        try mlxOp(c.mlx_reshape(&w, w, &[_]c_int{ B, L, self.n_heads * self.head_dim }, 3, self.stream));
        try einsumOp(&w, "blf,df->bld", &[_]c.mlx_array{ w, self.o_proj }, self.stream);
        try mlxOp(c.mlx_astype(result, w, input_dtype, self.stream));
    }

    pub fn deinit(self: *Attention) void {
        _ = c.mlx_array_free(self.q_proj);
        _ = c.mlx_array_free(self.k_proj);
        _ = c.mlx_array_free(self.v_proj);
        _ = c.mlx_array_free(self.o_proj);
        _ = c.mlx_array_free(self.scale_array);
        self.rope.deinit();
    }
};

pub const Llama3RoPE = struct {
    freqs: c.mlx_array,
    dims: c_int,
    traditional: bool,
    max_position_embeddings: c_int,
    stream: c.mlx_stream,

    pub fn init(
        dims: c_int,
        max_position_embeddings: c_int,
        traditional: bool,
        base: f32,
        scaling_config: RopeScalingConfig,
        stream: c.mlx_stream,
    ) !Llama3RoPE {
        var freqs = c.mlx_array_new();
        var arange = c.mlx_array_new();
        var intermediate = c.mlx_array_new();
        var base_freqs = c.mlx_array_new();
        var wavelens = c.mlx_array_new();
        var high_freq_scaled = c.mlx_array_new();
        var high_freq_mask = c.mlx_array_new();
        var mid_freq_mask1 = c.mlx_array_new();
        var mid_freq_mask2 = c.mlx_array_new();
        var mid_freq_mask = c.mlx_array_new();
        var high_freq_result = c.mlx_array_new();
        var smooth_factors = c.mlx_array_new();
        var denominator = c.mlx_array_new();
        var mid_freq_scaled = c.mlx_array_new();
        const dims_array = c.mlx_array_new_float(@floatFromInt(dims));
        const base_array = c.mlx_array_new_float(base);
        const factor_array = c.mlx_array_new_float(scaling_config.factor);
        const low_freq_factor_array = c.mlx_array_new_float(scaling_config.low_freq_factor);
        const high_freq_factor_array = c.mlx_array_new_float(scaling_config.high_freq_factor);
        const old_ctx_array = c.mlx_array_new_float(@floatFromInt(scaling_config.original_max_position_embeddings));
        const two_pi = c.mlx_array_new_float(2.0 * std.math.pi);
        const one_array = c.mlx_array_new_float(1.0);
        const low_freq_wavelen = c.mlx_array_new_float(@as(f32, @floatFromInt(scaling_config.original_max_position_embeddings)) /
            scaling_config.low_freq_factor);
        const high_freq_wavelen = c.mlx_array_new_float(@as(f32, @floatFromInt(scaling_config.original_max_position_embeddings)) /
            scaling_config.high_freq_factor);
        defer {
            _ = c.mlx_array_free(arange);
            _ = c.mlx_array_free(intermediate);
            _ = c.mlx_array_free(base_freqs);
            _ = c.mlx_array_free(wavelens);
            _ = c.mlx_array_free(high_freq_scaled);
            _ = c.mlx_array_free(high_freq_mask);
            _ = c.mlx_array_free(mid_freq_mask1);
            _ = c.mlx_array_free(mid_freq_mask2);
            _ = c.mlx_array_free(mid_freq_mask);
            _ = c.mlx_array_free(high_freq_result);
            _ = c.mlx_array_free(smooth_factors);
            _ = c.mlx_array_free(denominator);
            _ = c.mlx_array_free(mid_freq_scaled);
            _ = c.mlx_array_free(dims_array);
            _ = c.mlx_array_free(base_array);
            _ = c.mlx_array_free(factor_array);
            _ = c.mlx_array_free(low_freq_factor_array);
            _ = c.mlx_array_free(high_freq_factor_array);
            _ = c.mlx_array_free(old_ctx_array);
            _ = c.mlx_array_free(two_pi);
            _ = c.mlx_array_free(one_array);
            _ = c.mlx_array_free(low_freq_wavelen);
            _ = c.mlx_array_free(high_freq_wavelen);
        }
        try mlxOp(c.mlx_arange(&arange, 0, @as(f64, @floatFromInt(dims)), 2, c.MLX_FLOAT32, stream));
        try mlxOp(c.mlx_divide(&intermediate, arange, dims_array, stream));
        try mlxOp(c.mlx_power(&base_freqs, base_array, intermediate, stream));
        try mlxOp(c.mlx_multiply(&wavelens, two_pi, base_freqs, stream));
        try mlxOp(c.mlx_multiply(&high_freq_scaled, base_freqs, factor_array, stream));
        try mlxOp(c.mlx_greater(&high_freq_mask, wavelens, low_freq_wavelen, stream));
        try mlxOp(c.mlx_greater(&mid_freq_mask1, wavelens, high_freq_wavelen, stream));
        try mlxOp(c.mlx_less_equal(&mid_freq_mask2, wavelens, low_freq_wavelen, stream));
        try mlxOp(c.mlx_logical_and(&mid_freq_mask, mid_freq_mask1, mid_freq_mask2, stream));
        try mlxOp(c.mlx_where(&high_freq_result, high_freq_mask, high_freq_scaled, base_freqs, stream));
        try mlxOp(c.mlx_divide(&intermediate, old_ctx_array, wavelens, stream));
        try mlxOp(c.mlx_subtract(&intermediate, intermediate, low_freq_factor_array, stream));
        try mlxOp(c.mlx_subtract(&denominator, high_freq_factor_array, low_freq_factor_array, stream));
        try mlxOp(c.mlx_divide(&smooth_factors, intermediate, denominator, stream));
        try mlxOp(c.mlx_subtract(&intermediate, one_array, smooth_factors, stream));
        try mlxOp(c.mlx_divide(&intermediate, intermediate, factor_array, stream));
        try mlxOp(c.mlx_add(&intermediate, intermediate, smooth_factors, stream));
        try mlxOp(c.mlx_divide(&mid_freq_scaled, base_freqs, intermediate, stream));
        try mlxOp(c.mlx_where(&freqs, mid_freq_mask, mid_freq_scaled, high_freq_result, stream));
        return Llama3RoPE{
            .freqs = freqs,
            .dims = dims,
            .traditional = traditional,
            .max_position_embeddings = max_position_embeddings,
            .stream = stream,
        };
    }

    pub fn forward(self: *Llama3RoPE, result: *c.mlx_array, x: c.mlx_array, offset: c_int) !void {
        const rope_base = c.mlx_optional_float{ .has_value = false, .value = 0.0 };
        try mlxOp(c.mlx_fast_rope(result, x, self.dims, self.traditional, rope_base, 1.0, offset, self.freqs, self.stream));
    }

    pub fn deinit(self: *Llama3RoPE) void {
        _ = c.mlx_array_free(self.freqs);
    }
};

pub const TransformerBlock = struct {
    attention: Attention,
    mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    stream: c.mlx_stream,

    pub fn init(attention: Attention, mlp: MLP, input_layernorm: RMSNorm, post_attention_layernorm: RMSNorm, stream: c.mlx_stream) TransformerBlock {
        return TransformerBlock{
            .attention = attention,
            .mlp = mlp,
            .input_layernorm = input_layernorm,
            .post_attention_layernorm = post_attention_layernorm,
            .stream = stream,
        };
    }

    pub fn load(loader: *WeightLoader, prefix: []const u8, config: *const LlamaConfig, stream: c.mlx_stream) !TransformerBlock {
        const attn_prefix = try std.fmt.allocPrint(loader.allocator, "{s}.self_attn", .{prefix});
        defer loader.allocator.free(attn_prefix);
        const attention = try Attention.load(loader, attn_prefix, config, stream);
        const mlp_prefix = try std.fmt.allocPrint(loader.allocator, "{s}.mlp", .{prefix});
        defer loader.allocator.free(mlp_prefix);
        const mlp = try MLP.load(loader, mlp_prefix, stream);
        const input_ln_prefix = try std.fmt.allocPrint(loader.allocator, "{s}.input_layernorm", .{prefix});
        defer loader.allocator.free(input_ln_prefix);
        const input_layernorm = try RMSNorm.load(loader, input_ln_prefix, config.rms_norm_eps, stream);
        const post_ln_prefix = try std.fmt.allocPrint(loader.allocator, "{s}.post_attention_layernorm", .{prefix});
        defer loader.allocator.free(post_ln_prefix);
        const post_attention_layernorm = try RMSNorm.load(loader, post_ln_prefix, config.rms_norm_eps, stream);
        return TransformerBlock.init(attention, mlp, input_layernorm, post_attention_layernorm, stream);
    }

    pub fn forward(self: *TransformerBlock, result: *c.mlx_array, x: c.mlx_array, mask: ?c.mlx_array, cache: ?*KVCache, offset: c_int) !void {
        var attn = c.mlx_array_new();
        var mlp = c.mlx_array_new();
        defer {
            _ = c.mlx_array_free(attn);
            _ = c.mlx_array_free(mlp);
        }
        try self.input_layernorm.forward(&attn, x);
        try self.attention.forward(&attn, attn, mask, cache, offset);
        try mlxOp(c.mlx_add(&attn, attn, x, self.stream));
        try self.post_attention_layernorm.forward(&mlp, attn);
        try self.mlp.forward(&mlp, mlp);
        try mlxOp(c.mlx_add(result, mlp, attn, self.stream));
    }

    pub fn deinit(self: *TransformerBlock) void {
        self.attention.deinit();
        self.mlp.deinit();
        self.input_layernorm.deinit();
        self.post_attention_layernorm.deinit();
    }
};

pub const LlamaModel = struct {
    embed_tokens: Embedding,
    layers: []TransformerBlock,
    norm: RMSNorm,
    stream: c.mlx_stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, embed_tokens: Embedding, layers: []TransformerBlock, norm: RMSNorm, stream: c.mlx_stream) LlamaModel {
        return LlamaModel{
            .embed_tokens = embed_tokens,
            .layers = layers,
            .norm = norm,
            .stream = stream,
            .allocator = allocator,
        };
    }

    pub fn load(loader: *WeightLoader, allocator: std.mem.Allocator, config: *const LlamaConfig, stream: c.mlx_stream) !LlamaModel {
        const embed_tokens = try Embedding.load(loader, "model.embed_tokens", stream);
        var layers = try allocator.alloc(TransformerBlock, @intCast(config.num_hidden_layers));
        errdefer {
            for (0..layers.len) |i| {
                if (i < layers.len) layers[i].deinit();
            }
            allocator.free(layers);
        }
        for (0..@intCast(config.num_hidden_layers)) |i| {
            const layer_prefix = try std.fmt.allocPrint(allocator, "model.layers.{d}", .{i});
            defer allocator.free(layer_prefix);

            layers[i] = try TransformerBlock.load(loader, layer_prefix, config, stream);
        }
        const norm = try RMSNorm.load(loader, "model.norm", config.rms_norm_eps, stream);
        return LlamaModel.init(allocator, embed_tokens, layers, norm, stream);
    }

    pub fn forward(self: *LlamaModel, result: *c.mlx_array, toks: c.mlx_array, mask: ?c.mlx_array, cache: ?*Cache) !void {
        const seq_len = c.mlx_array_dim(toks, 1);
        const offset = if (cache) |c_| c_.offset else 0;
        var current = c.mlx_array_new();
        defer _ = c.mlx_array_free(current);
        try self.embed_tokens.forward(&current, toks);
        for (self.layers, 0..) |*layer, i| {
            const layer_cache = if (cache) |c_| &c_.layers[i] else null;
            try layer.forward(&current, current, mask, layer_cache, offset);
        }
        try self.norm.forward(result, current);
        if (cache) |c_| {
            c_.offset += seq_len;
        }
    }

    pub fn deinit(self: *LlamaModel) void {
        self.embed_tokens.deinit();
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.norm.deinit();
    }
};

pub const Llama = struct {
    model: LlamaModel,
    tie_word_embeddings: bool,
    lm_head: ?Linear,
    stream: c.mlx_stream,

    pub fn init(model: LlamaModel, tie_word_embeddings: bool, lm_head: ?Linear, stream: c.mlx_stream) Llama {
        return Llama{
            .model = model,
            .tie_word_embeddings = tie_word_embeddings,
            .lm_head = lm_head,
            .stream = stream,
        };
    }

    pub fn load(loader: *WeightLoader, allocator: std.mem.Allocator, config: *const LlamaConfig, stream: c.mlx_stream) !Llama {
        const model = try LlamaModel.load(loader, allocator, config, stream);
        var lm_head: ?Linear = null;
        if (!config.tie_word_embeddings) {
            lm_head = try Linear.load(loader, "lm_head", stream);
        }
        return Llama.init(model, config.tie_word_embeddings, lm_head, stream);
    }

    pub fn forward(self: *Llama, result: *c.mlx_array, toks: c.mlx_array, mask: ?c.mlx_array, cache: ?*Cache) !void {
        var hidden = c.mlx_array_new();
        defer _ = c.mlx_array_free(hidden);
        try self.model.forward(&hidden, toks, mask, cache);
        if (self.tie_word_embeddings) {
            try self.model.embed_tokens.asLinear(result, hidden);
        } else {
            try self.lm_head.?.forward(result, hidden);
        }
    }

    pub fn deinit(self: *Llama) void {
        self.model.deinit();

        if (!self.tie_word_embeddings) {
            self.lm_head.?.deinit();
        }
    }
};

pub const DefaultTransformer = struct {
    allocator: std.mem.Allocator,
    stream: c.mlx_stream,
    model: Llama,

    pub fn init(allocator: std.mem.Allocator, model_path: ?[]const u8) !DefaultTransformer {
        const stream = c.mlx_default_cpu_stream_new();
        const filename = model_path orelse "model.safetensors";
        const file = c.fopen(filename.ptr, "rb");
        var weights = c.mlx_map_string_to_array_new();
        var meta = c.mlx_map_string_to_string_new();
        defer {
            _ = c.fclose(file);
            _ = c.mlx_map_string_to_array_free(weights);
            _ = c.mlx_map_string_to_string_free(meta);
        }
        _ = c.mlx_load_safetensors_file(&weights, &meta, file, stream);
        var loader = WeightLoader{
            .weights = weights,
            .allocator = allocator,
        };
        return DefaultTransformer{
            .model = try Llama.load(&loader, allocator, &llamaConfig, stream),
            .allocator = allocator,
            .stream = stream,
        };
    }

    pub fn deinit(self: *DefaultTransformer) void {
        self.model.deinit();
        _ = c.mlx_stream_free(self.stream);
    }

    pub fn generate(self: *DefaultTransformer, initial_tokens: []const u32, num_tokens: usize) ![]u32 {
        var output_tokens = try self.allocator.alloc(u32, num_tokens);
        errdefer self.allocator.free(output_tokens);
        var cache = try Cache.init(self.allocator, self.model.model.layers.len);
        var toks = c.mlx_array_new_data(@ptrCast(initial_tokens.ptr), &[_]c_int{ 1, @intCast(initial_tokens.len) }, 2, c.MLX_UINT32);
        var logits = c.mlx_array_new();
        var mask = c.mlx_array_new();
        defer {
            cache.deinit();
            _ = c.mlx_array_free(toks);
            _ = c.mlx_array_free(logits);
            _ = c.mlx_array_free(mask);
        }
        for (0..num_tokens) |i| {
            try createCausalMask(&mask, @intCast(c.mlx_array_dim(toks, 1)), cache.offset, self.stream);
            try self.model.forward(&logits, toks, mask, &cache);
            // try mlxOp(c.mlx_take(&logits, logits, c.mlx_array_new_int(c.mlx_array_dim(toks, 1) - 1), 1, self.stream)); // unnecessary
            try mlxOp(c.mlx_take(&logits, logits, c.mlx_array_new_int(-1), 1, self.stream));
            try mlxOp(c.mlx_argmax(&logits, logits, 1, false, self.stream));
            try mlxOp(c.mlx_array_item_uint32(&output_tokens[i], logits));
            try mlxOp(c.mlx_array_set_data(&toks, &output_tokens[i], &[_]c_int{ 1, 1 }, 2, c.MLX_UINT32));
            std.debug.print("Generated token {d}/{d}: {d}\n", .{ i + 1, num_tokens, output_tokens[i] });
        }
        return output_tokens;
    }
};

pub const WeightLoader = struct {
    weights: c.mlx_map_string_to_array,
    allocator: std.mem.Allocator,
    pub fn getTensor(self: *WeightLoader, key: []const u8) ?c.mlx_array {
        var result = c.mlx_array_new();
        if (c.mlx_map_string_to_array_get(&result, self.weights, key.ptr) == 0) {
            _ = c.mlx_array_eval(result);
            return result;
        }
        _ = c.mlx_array_free(result);
        return null;
    }
    pub fn getComponentWeight(self: *WeightLoader, prefix: []const u8, name: []const u8) !c.mlx_array {
        const key = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ prefix, name });
        defer self.allocator.free(key);
        return self.getTensor(key) orelse return error.MissingWeight;
    }
};

pub const KVCache = struct {
    k: c.mlx_array,
    v: c.mlx_array,

    pub fn init() KVCache {
        return KVCache{
            .k = c.mlx_array_new(),
            .v = c.mlx_array_new(),
        };
    }

    fn sliceCache(self: *KVCache, offset: c_int, stream: c.mlx_stream) !void {
        if (offset >= c.mlx_array_dim(self.k, 2)) {
            return;
        }
        const ndim = c.mlx_array_ndim(self.k);
        const start = [_]c_int{0} ** 4;
        const strides = [_]c_int{1} ** 4;
        var stop = [_]c_int{0} ** 4;
        for (0..ndim) |i| {
            stop[i] = c.mlx_array_dim(self.k, @intCast(i));
        }
        stop[2] = offset;
        try mlxOp(c.mlx_slice(&self.k, self.k, &start, ndim, &stop, ndim, &strides, ndim, stream));
        try mlxOp(c.mlx_slice(&self.k, self.v, &start, ndim, &stop, ndim, &strides, ndim, stream));
        std.debug.print("Cache offset set to {d}\n", .{offset});
    }

    pub fn update(self: *KVCache, k: *c.mlx_array, v: *c.mlx_array, offset: c_int, stream: c.mlx_stream) !void {
        if (offset > 0) {
            try self.sliceCache(offset, stream);
            var k_concat = [_]c.mlx_array{ self.k, k.* };
            const k_vec = c.mlx_vector_array_new_data(&k_concat[0], 2);
            defer _ = c.mlx_vector_array_free(k_vec);
            try mlxOp(c.mlx_concatenate(k, k_vec, 2, stream));
            var v_concat = [_]c.mlx_array{ self.v, v.* };
            const v_vec = c.mlx_vector_array_new_data(&v_concat[0], 2);
            defer _ = c.mlx_vector_array_free(v_vec);
            try mlxOp(c.mlx_concatenate(v, v_vec, 2, stream));
        }
        try mlxOp(c.mlx_array_set(&self.k, k.*));
        try mlxOp(c.mlx_array_set(&self.v, v.*));
    }

    pub fn deinit(self: *KVCache) void {
        _ = c.mlx_array_free(self.k);
        _ = c.mlx_array_free(self.v);
    }
};

pub const Cache = struct {
    layers: []KVCache,
    offset: c_int,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_layers: usize) !Cache {
        var layers = try allocator.alloc(KVCache, num_layers);

        for (0..num_layers) |i| {
            layers[i] = KVCache.init();
        }

        return Cache{
            .layers = layers,
            .offset = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Cache) void {
        for (self.layers) |*cache| {
            cache.deinit();
        }
        self.allocator.free(self.layers);
    }
};

pub fn mlxOp(result: c_int) !void {
    if (result != 0) return error.MLXOperationFailed;
}

pub fn einsumOp(result: *c.mlx_array, pattern: [*:0]const u8, arrays: []const c.mlx_array, stream: c.mlx_stream) !void {
    const operands = c.mlx_vector_array_new_data(@ptrCast(arrays.ptr), @intCast(arrays.len));
    defer _ = c.mlx_vector_array_free(operands);
    try mlxOp(c.mlx_einsum(result, pattern, operands, stream));
}

pub fn printArray(msg: []const u8, arr: c.mlx_array) void {
    var str = c.mlx_string_new();
    defer _ = c.mlx_string_free(str);
    const ndim = c.mlx_array_ndim(arr);
    const shape = c.mlx_array_shape(arr);
    _ = c.mlx_array_tostring(&str, arr);
    std.debug.print("{s}\n{s}\n", .{ msg, c.mlx_string_data(str) });
    std.debug.print("Shape: [", .{});
    for (0..ndim) |i| {
        if (i > 0) std.debug.print(",", .{});
        std.debug.print("{d}", .{shape[i]});
    }
    std.debug.print("]\n", .{});
}

pub fn createCausalMask(result: *c.mlx_array, seq_len: c_int, offset: c_int, stream: c.mlx_stream) !void {
    const zero = c.mlx_array_new_float(0.0);
    const neg_inf = c.mlx_array_new_float(-std.math.inf(f32));
    var mask = c.mlx_array_new();
    defer {
        _ = c.mlx_array_free(zero);
        _ = c.mlx_array_free(neg_inf);
        _ = c.mlx_array_free(mask);
    }
    const total_len = seq_len + offset;
    try mlxOp(c.mlx_ones(&mask, &[_]c_int{ seq_len, total_len }, 2, c.MLX_INT32, stream));
    try mlxOp(c.mlx_tril(&mask, mask, offset, stream));
    try mlxOp(c.mlx_reshape(&mask, mask, &[_]c_int{ 1, 1, seq_len, total_len }, 4, stream));
    try mlxOp(c.mlx_where(result, mask, zero, neg_inf, stream));
}

test "Transformer generating" {
    std.debug.print("\n=== TRANSFORMER.ZIG ===\n\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const initial_tokens = [_]u32{ 9906, 1917 };
    const num_tokens_to_generate = 10;
    var transformer = try DefaultTransformer.init(allocator, null);
    defer transformer.deinit();
    const generated_tokens = try transformer.generate(&initial_tokens, num_tokens_to_generate);
    defer allocator.free(generated_tokens);
    std.debug.print("\nGenerated sequence: ", .{});
    for (generated_tokens) |token| {
        std.debug.print("{d} ", .{token});
    }
    std.debug.print("\n", .{});
}
