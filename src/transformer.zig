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

pub const Linear = struct {
    weight: c.mlx_array,
    scales: ?c.mlx_array,
    biases: ?c.mlx_array,
    quants: ?([2]c_int),
    stream: c.mlx_stream,
    prefix: []const u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent_prefix: []const u8, suffix: []const u8, config: *const LlamaConfig, stream: c.mlx_stream) !Linear {
        const prefix = try joinPath(allocator, parent_prefix, suffix);
        errdefer allocator.free(prefix);
        var quants: ?([2]c_int) = null;
        var scales: ?c.mlx_array = null;
        var biases: ?c.mlx_array = null;
        if (config.quantization) |qconfig| {
            quants = .{ qconfig.group_size, qconfig.bits };
            biases = c.mlx_array_new();
            scales = c.mlx_array_new();
        }
        return Linear{
            .weight = c.mlx_array_new(),
            .scales = scales,
            .biases = biases,
            .quants = quants,
            .stream = stream,
            .prefix = prefix,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Linear, weights_map: *const c.mlx_map_string_to_array) !void {
        if (self.quants) |_| {
            try loadQuantized(&self.weight, &self.scales.?, &self.biases.?, "", self.prefix, weights_map, self.allocator);
        } else {
            try loadWeight(&self.weight, "", self.prefix, weights_map, self.allocator);
        }
    }

    pub fn deinit(self: *Linear) void {
        self.allocator.free(self.prefix);
        _ = c.mlx_array_free(self.weight);
        if (self.quants) |_| {
            _ = c.mlx_array_free(self.scales.?);
            _ = c.mlx_array_free(self.biases.?);
        }
    }

    pub fn forward(self: *Linear, result: *c.mlx_array, x: c.mlx_array) !void {
        if (self.quants) |qconfig| {
            try mlxOp(c.mlx_quantized_matmul(result, x, self.weight, self.scales.?, self.biases.?, true, qconfig[0], qconfig[1], self.stream));
        } else {
            try einsum(result, .{ x, self.weight }, "bsh,vh->bsv", self.stream);
        }
    }
};

pub const Embedding = struct {
    weight: c.mlx_array,
    quants: ?([2]c_int),
    stream: c.mlx_stream,
    prefix: []const u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent_prefix: []const u8, config: *const LlamaConfig, stream: c.mlx_stream) !Embedding {
        const prefix = try joinPath(allocator, parent_prefix, "embed_tokens");
        errdefer allocator.free(prefix);
        var quants: ?([2]c_int) = null;
        if (config.quantization) |qconfig| {
            quants = .{ qconfig.group_size, qconfig.bits };
        }
        return Embedding{
            .weight = c.mlx_array_new(),
            .quants = quants,
            .stream = stream,
            .prefix = prefix,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Embedding, weights_map: *const c.mlx_map_string_to_array) !void {
        if (self.quants) |qconfig| {
            var qeight = c.mlx_array_new();
            var scales = c.mlx_array_new();
            var biases = c.mlx_array_new();
            defer {
                _ = c.mlx_array_free(qeight);
                _ = c.mlx_array_free(scales);
                _ = c.mlx_array_free(biases);
            }
            try loadQuantized(&qeight, &scales, &biases, "", self.prefix, weights_map, self.allocator);
            try mlxOp(c.mlx_dequantize(&self.weight, qeight, scales, biases, qconfig[0], qconfig[1], self.stream));
            try mlxOp(c.mlx_array_eval(self.weight));
        } else {
            try loadWeight(&self.weight, "", self.prefix, weights_map, self.allocator);
        }
    }

    pub fn forward(self: *Embedding, result: *c.mlx_array, toks: c.mlx_array) !void {
        try mlxOp(c.mlx_take(result, self.weight, toks, 0, self.stream));
    }

    pub fn asLinear(self: *Embedding, result: *c.mlx_array, x: c.mlx_array) !void {
        try einsum(result, .{ x, self.weight }, "blh,dh->bld", self.stream);
    }

    pub fn deinit(self: *Embedding) void {
        _ = c.mlx_array_free(self.weight);
        self.allocator.free(self.prefix);
    }
};

pub const RMSNorm = struct {
    weight: c.mlx_array,
    eps: f32,
    stream: c.mlx_stream,
    prefix: []const u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent_prefix: []const u8, suffix: []const u8, eps: f32, stream: c.mlx_stream) !RMSNorm {
        const prefix = try joinPath(allocator, parent_prefix, suffix);
        errdefer allocator.free(prefix);
        return RMSNorm{
            .weight = c.mlx_array_new(),
            .eps = eps,
            .stream = stream,
            .prefix = prefix,
            .allocator = allocator,
        };
    }

    pub fn load(self: *RMSNorm, weights_map: *const c.mlx_map_string_to_array) !void {
        try loadWeight(&self.weight, "", self.prefix, weights_map, self.allocator);
    }

    pub fn forward(self: *RMSNorm, result: *c.mlx_array, x: c.mlx_array) !void {
        try mlxOp(c.mlx_fast_rms_norm(result, x, self.weight, self.eps, self.stream));
    }

    pub fn deinit(self: *RMSNorm) void {
        _ = c.mlx_array_free(self.weight);
        self.allocator.free(self.prefix);
    }
};

pub const MLP = struct {
    gate_proj: c.mlx_array,
    up_proj: c.mlx_array,
    down_proj: c.mlx_array,
    gate_scales: ?c.mlx_array,
    gate_biases: ?c.mlx_array,
    up_scales: ?c.mlx_array,
    up_biases: ?c.mlx_array,
    down_scales: ?c.mlx_array,
    down_biases: ?c.mlx_array,
    quants: ?([2]c_int),
    stream: c.mlx_stream,
    prefix: []const u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent_prefix: []const u8, suffix: []const u8, config: *const LlamaConfig, stream: c.mlx_stream) !MLP {
        const prefix = try joinPath(allocator, parent_prefix, suffix);
        errdefer allocator.free(prefix);
        var quants: ?([2]c_int) = null;
        var gate_scales: ?c.mlx_array = null;
        var gate_biases: ?c.mlx_array = null;
        var up_scales: ?c.mlx_array = null;
        var up_biases: ?c.mlx_array = null;
        var down_scales: ?c.mlx_array = null;
        var down_biases: ?c.mlx_array = null;

        if (config.quantization) |qconfig| {
            quants = .{ qconfig.group_size, qconfig.bits };
            gate_scales = c.mlx_array_new();
            gate_biases = c.mlx_array_new();
            up_scales = c.mlx_array_new();
            up_biases = c.mlx_array_new();
            down_scales = c.mlx_array_new();
            down_biases = c.mlx_array_new();
        }

        return MLP{
            .gate_proj = c.mlx_array_new(),
            .up_proj = c.mlx_array_new(),
            .down_proj = c.mlx_array_new(),
            .gate_scales = gate_scales,
            .gate_biases = gate_biases,
            .up_scales = up_scales,
            .up_biases = up_biases,
            .down_scales = down_scales,
            .down_biases = down_biases,
            .quants = quants,
            .stream = stream,
            .prefix = prefix,
            .allocator = allocator,
        };
    }

    pub fn load(self: *MLP, weights_map: *const c.mlx_map_string_to_array) !void {
        if (self.quants != null) {
            try loadQuantized(&self.gate_proj, &self.gate_scales.?, &self.gate_biases.?, "gate_proj", self.prefix, weights_map, self.allocator);
            try loadQuantized(&self.up_proj, &self.up_scales.?, &self.up_biases.?, "up_proj", self.prefix, weights_map, self.allocator);
            try loadQuantized(&self.down_proj, &self.down_scales.?, &self.down_biases.?, "down_proj", self.prefix, weights_map, self.allocator);
        } else {
            try loadWeight(&self.gate_proj, "gate_proj", self.prefix, weights_map, self.allocator);
            try loadWeight(&self.up_proj, "up_proj", self.prefix, weights_map, self.allocator);
            try loadWeight(&self.down_proj, "down_proj", self.prefix, weights_map, self.allocator);
        }
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

        if (self.quants) |qconfig| {
            try mlxOp(c.mlx_quantized_matmul(&gate, x, self.gate_proj, self.gate_scales.?, self.gate_biases.?, true, qconfig[0], qconfig[1], self.stream));
            try mlxOp(c.mlx_sigmoid(&sigmoid, gate, self.stream));
            try mlxOp(c.mlx_multiply(&gate, gate, sigmoid, self.stream));
            try mlxOp(c.mlx_quantized_matmul(&up, x, self.up_proj, self.up_scales.?, self.up_biases.?, true, qconfig[0], qconfig[1], self.stream));
            try mlxOp(c.mlx_multiply(result, gate, up, self.stream));
            try mlxOp(c.mlx_quantized_matmul(result, result.*, self.down_proj, self.down_scales.?, self.down_biases.?, true, qconfig[0], qconfig[1], self.stream));
        } else {
            try einsum(&gate, .{ x, self.gate_proj }, "bld,hd->blh", self.stream);
            try mlxOp(c.mlx_sigmoid(&sigmoid, gate, self.stream));
            try mlxOp(c.mlx_multiply(&gate, gate, sigmoid, self.stream));
            try einsum(&up, .{ x, self.up_proj }, "bld,hd->blh", self.stream);
            try mlxOp(c.mlx_multiply(result, gate, up, self.stream));
            try einsum(result, .{ result.*, self.down_proj }, "blh,dh->bld", self.stream);
        }
    }

    pub fn deinit(self: *MLP) void {
        _ = c.mlx_array_free(self.gate_proj);
        _ = c.mlx_array_free(self.up_proj);
        _ = c.mlx_array_free(self.down_proj);
        if (self.quants != null) {
            _ = c.mlx_array_free(self.gate_scales.?);
            _ = c.mlx_array_free(self.gate_biases.?);
            _ = c.mlx_array_free(self.up_scales.?);
            _ = c.mlx_array_free(self.up_biases.?);
            _ = c.mlx_array_free(self.down_scales.?);
            _ = c.mlx_array_free(self.down_biases.?);
        }
        self.allocator.free(self.prefix);
    }
};

pub const Attention = struct {
    q_proj: c.mlx_array,
    k_proj: c.mlx_array,
    v_proj: c.mlx_array,
    o_proj: c.mlx_array,
    q_scales: ?c.mlx_array,
    q_biases: ?c.mlx_array,
    k_scales: ?c.mlx_array,
    k_biases: ?c.mlx_array,
    v_scales: ?c.mlx_array,
    v_biases: ?c.mlx_array,
    o_scales: ?c.mlx_array,
    o_biases: ?c.mlx_array,
    quants: ?([2]c_int),
    n_heads: c_int,
    n_kv_heads: c_int,
    head_dim: c_int,
    n_repeat: c_int,
    scale_array: c.mlx_array,
    rope: Llama3RoPE,
    stream: c.mlx_stream,
    prefix: []const u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent_prefix: []const u8, suffix: []const u8, n_heads: c_int, n_kv_heads: c_int, head_dim: c_int, config: *const LlamaConfig, stream: c.mlx_stream) !Attention {
        const prefix = try joinPath(allocator, parent_prefix, suffix);
        errdefer allocator.free(prefix);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const scale_array = c.mlx_array_new_float(scale);
        const n_repeat = @divExact(n_heads, n_kv_heads);
        const rope = try Llama3RoPE.init(head_dim, config.max_position_embeddings, false, config.rope_theta, config.rope_scaling, stream);
        var quants: ?([2]c_int) = null;
        var q_scales: ?c.mlx_array = null;
        var q_biases: ?c.mlx_array = null;
        var k_scales: ?c.mlx_array = null;
        var k_biases: ?c.mlx_array = null;
        var v_scales: ?c.mlx_array = null;
        var v_biases: ?c.mlx_array = null;
        var o_scales: ?c.mlx_array = null;
        var o_biases: ?c.mlx_array = null;

        if (config.quantization) |qconfig| {
            quants = .{ qconfig.group_size, qconfig.bits };
            q_scales = c.mlx_array_new();
            q_biases = c.mlx_array_new();
            k_scales = c.mlx_array_new();
            k_biases = c.mlx_array_new();
            v_scales = c.mlx_array_new();
            v_biases = c.mlx_array_new();
            o_scales = c.mlx_array_new();
            o_biases = c.mlx_array_new();
        }

        return Attention{
            .q_proj = c.mlx_array_new(),
            .k_proj = c.mlx_array_new(),
            .v_proj = c.mlx_array_new(),
            .o_proj = c.mlx_array_new(),
            .q_scales = q_scales,
            .q_biases = q_biases,
            .k_scales = k_scales,
            .k_biases = k_biases,
            .v_scales = v_scales,
            .v_biases = v_biases,
            .o_scales = o_scales,
            .o_biases = o_biases,
            .quants = quants,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .n_repeat = n_repeat,
            .scale_array = scale_array,
            .rope = rope,
            .stream = stream,
            .prefix = prefix,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Attention, weights_map: *const c.mlx_map_string_to_array) !void {
        if (self.quants != null) {
            try loadQuantized(&self.q_proj, &self.q_scales.?, &self.q_biases.?, "q_proj", self.prefix, weights_map, self.allocator);
            try loadQuantized(&self.k_proj, &self.k_scales.?, &self.k_biases.?, "k_proj", self.prefix, weights_map, self.allocator);
            try loadQuantized(&self.v_proj, &self.v_scales.?, &self.v_biases.?, "v_proj", self.prefix, weights_map, self.allocator);
            try loadQuantized(&self.o_proj, &self.o_scales.?, &self.o_biases.?, "o_proj", self.prefix, weights_map, self.allocator);
        } else {
            try loadWeight(&self.q_proj, "q_proj", self.prefix, weights_map, self.allocator);
            try loadWeight(&self.k_proj, "k_proj", self.prefix, weights_map, self.allocator);
            try loadWeight(&self.v_proj, "v_proj", self.prefix, weights_map, self.allocator);
            try loadWeight(&self.o_proj, "o_proj", self.prefix, weights_map, self.allocator);
        }
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
        if (self.quants) |qconfig| {
            try mlxOp(c.mlx_quantized_matmul(&q, x, self.q_proj, self.q_scales.?, self.q_biases.?, true, qconfig[0], qconfig[1], self.stream));
            try mlxOp(c.mlx_quantized_matmul(&k, x, self.k_proj, self.k_scales.?, self.k_biases.?, true, qconfig[0], qconfig[1], self.stream));
            try mlxOp(c.mlx_quantized_matmul(&v, x, self.v_proj, self.v_scales.?, self.v_biases.?, true, qconfig[0], qconfig[1], self.stream));
        } else {
            try einsum(&q, .{ x, self.q_proj }, "b l d, f d -> b l f", self.stream);
            try einsum(&k, .{ x, self.k_proj }, "b l d, f d -> b l f", self.stream);
            try einsum(&v, .{ x, self.v_proj }, "b l d, f d -> b l f", self.stream);
        }
        try reshap(&q, q, "b l (h d) -> b l h d", .{ .h = self.n_heads, .d = self.head_dim }, self.stream);
        try reshap(&k, k, "b l (h d) -> b l h d", .{ .h = self.n_kv_heads, .d = self.head_dim }, self.stream);
        try reshap(&v, v, "b l (h d) -> b l h d", .{ .h = self.n_kv_heads, .d = self.head_dim }, self.stream);
        try einsum(&q, .{q}, "b l h d -> b h l d", self.stream);
        try einsum(&k, .{k}, "b l h d -> b h l d", self.stream);
        try einsum(&v, .{v}, "b l h d -> b h l d", self.stream);
        try self.rope.forward(&q, q, offset);
        try self.rope.forward(&k, k, offset);
        try mlxOp(c.mlx_multiply(&q, q, self.scale_array, self.stream));
        if (cache != null) {
            try cache.?.update(&k, &v, offset, self.stream);
        }
        try repeat(&k, k, "b h l d -> b (repeat h) l d", .{ .repeat = self.n_repeat }, self.stream);
        try repeat(&v, v, "b h l d -> b (repeat h) l d", .{ .repeat = self.n_repeat }, self.stream);
        try einsum(&w, .{ q, k }, "b h l d, b h k d -> b h l k", self.stream);

        if (mask) |mask_val| {
            try mlxOp(c.mlx_add(&w, w, mask_val, self.stream));
        }
        try mlxOp(c.mlx_softmax(&w, w, &[_]c_int{3}, 1, true, self.stream));
        try einsum(&w, .{ w, v }, "b h l k, b h k d -> b h l d", self.stream);
        try einsum(&w, .{w}, "b h l d -> b l h d", self.stream);
        try reshap(&w, w, "b l h d -> b l (h d)", .{}, self.stream);
        if (self.quants) |qconfig| {
            try mlxOp(c.mlx_quantized_matmul(result, w, self.o_proj, self.o_scales.?, self.o_biases.?, true, qconfig[0], qconfig[1], self.stream));
        } else {
            try einsum(result, .{ w, self.o_proj }, "b l f, d f -> b l d", self.stream);
        }
    }

    pub fn deinit(self: *Attention) void {
        _ = c.mlx_array_free(self.q_proj);
        _ = c.mlx_array_free(self.k_proj);
        _ = c.mlx_array_free(self.v_proj);
        _ = c.mlx_array_free(self.o_proj);
        _ = c.mlx_array_free(self.scale_array);

        if (self.quants != null) {
            _ = c.mlx_array_free(self.q_scales.?);
            _ = c.mlx_array_free(self.q_biases.?);
            _ = c.mlx_array_free(self.k_scales.?);
            _ = c.mlx_array_free(self.k_biases.?);
            _ = c.mlx_array_free(self.v_scales.?);
            _ = c.mlx_array_free(self.v_biases.?);
            _ = c.mlx_array_free(self.o_scales.?);
            _ = c.mlx_array_free(self.o_biases.?);
        }

        self.rope.deinit();
        self.allocator.free(self.prefix);
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

pub const Llama3RoPE = struct {
    freqs: c.mlx_array,
    rope_base: c.mlx_optional_float,
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
        var wavelens = c.mlx_array_new();
        var high_freq_mask = c.mlx_array_new();
        var mid_freq_mask = c.mlx_array_new();
        var high_freq = c.mlx_array_new();
        var smooth_factors = c.mlx_array_new();
        var mid_freq = c.mlx_array_new();
        const dims_scalar = c.mlx_array_new_float(@floatFromInt(dims));
        const base_scalar = c.mlx_array_new_float(base);
        const factor_scalar = c.mlx_array_new_float(scaling_config.factor);
        const low_freq_factor_scalar = c.mlx_array_new_float(scaling_config.low_freq_factor);
        const diff_freq_factor_scalar = c.mlx_array_new_float(scaling_config.high_freq_factor - scaling_config.low_freq_factor);
        const og_max_scalar = c.mlx_array_new_float(@floatFromInt(scaling_config.original_max_position_embeddings));
        const two_pi_scalar = c.mlx_array_new_float(2.0 * std.math.pi);
        const one_scalar = c.mlx_array_new_float(1.0);
        const low_freq_wavelen_scalar = c.mlx_array_new_float(@as(f32, @floatFromInt(scaling_config.original_max_position_embeddings)) / scaling_config.low_freq_factor);
        const high_freq_wavelen_scalar = c.mlx_array_new_float(@as(f32, @floatFromInt(scaling_config.original_max_position_embeddings)) / scaling_config.high_freq_factor);
        defer {
            _ = c.mlx_array_free(wavelens);
            _ = c.mlx_array_free(high_freq_mask);
            _ = c.mlx_array_free(mid_freq_mask);
            _ = c.mlx_array_free(high_freq);
            _ = c.mlx_array_free(smooth_factors);
            _ = c.mlx_array_free(mid_freq);
            _ = c.mlx_array_free(dims_scalar);
            _ = c.mlx_array_free(base_scalar);
            _ = c.mlx_array_free(factor_scalar);
            _ = c.mlx_array_free(low_freq_factor_scalar);
            _ = c.mlx_array_free(diff_freq_factor_scalar);
            _ = c.mlx_array_free(og_max_scalar);
            _ = c.mlx_array_free(two_pi_scalar);
            _ = c.mlx_array_free(one_scalar);
            _ = c.mlx_array_free(low_freq_wavelen_scalar);
            _ = c.mlx_array_free(high_freq_wavelen_scalar);
        }
        try mlxOp(c.mlx_arange(&freqs, 0, @as(f64, @floatFromInt(dims)), 2, c.MLX_FLOAT32, stream));
        try mlxOp(c.mlx_divide(&freqs, freqs, dims_scalar, stream));
        try mlxOp(c.mlx_power(&freqs, base_scalar, freqs, stream));
        try mlxOp(c.mlx_multiply(&wavelens, two_pi_scalar, freqs, stream));
        try mlxOp(c.mlx_multiply(&high_freq, freqs, factor_scalar, stream));
        try mlxOp(c.mlx_greater(&high_freq_mask, wavelens, low_freq_wavelen_scalar, stream));
        try mlxOp(c.mlx_where(&high_freq, high_freq_mask, high_freq, freqs, stream));
        try mlxOp(c.mlx_less_equal(&mid_freq_mask, wavelens, high_freq_wavelen_scalar, stream));
        try mlxOp(c.mlx_logical_or(&mid_freq_mask, high_freq_mask, mid_freq_mask, stream));
        try mlxOp(c.mlx_logical_not(&mid_freq_mask, mid_freq_mask, stream));
        try mlxOp(c.mlx_divide(&smooth_factors, og_max_scalar, wavelens, stream));
        try mlxOp(c.mlx_subtract(&smooth_factors, smooth_factors, low_freq_factor_scalar, stream));
        try mlxOp(c.mlx_divide(&smooth_factors, smooth_factors, diff_freq_factor_scalar, stream));
        try mlxOp(c.mlx_subtract(&mid_freq, one_scalar, smooth_factors, stream));
        try mlxOp(c.mlx_divide(&mid_freq, mid_freq, factor_scalar, stream));
        try mlxOp(c.mlx_add(&mid_freq, mid_freq, smooth_factors, stream));
        try mlxOp(c.mlx_divide(&mid_freq, freqs, mid_freq, stream));
        try mlxOp(c.mlx_where(&freqs, mid_freq_mask, mid_freq, high_freq, stream));
        return Llama3RoPE{
            .freqs = freqs,
            .rope_base = c.mlx_optional_float{ .has_value = false, .value = 0.0 },
            .dims = dims,
            .traditional = traditional,
            .max_position_embeddings = max_position_embeddings,
            .stream = stream,
        };
    }

    pub fn forward(self: *Llama3RoPE, result: *c.mlx_array, x: c.mlx_array, offset: c_int) !void {
        try mlxOp(c.mlx_fast_rope(result, x, self.dims, self.traditional, self.rope_base, 1.0, offset, self.freqs, self.stream));
        try mlxOp(c.mlx_astype(result, result.*, c.MLX_BFLOAT16, self.stream));
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
    allocator: std.mem.Allocator,
    prefix: []const u8,
    pub fn init(allocator: std.mem.Allocator, parent_prefix: []const u8, layer_idx: usize, config: *const LlamaConfig, stream: c.mlx_stream) !TransformerBlock {
        const prefix = try joinPath(allocator, parent_prefix, layer_idx);
        errdefer allocator.free(prefix);
        const attention = try Attention.init(allocator, prefix, "self_attn", config.num_attention_heads, config.num_key_value_heads, config.head_dim, config, stream);
        const mlp = try MLP.init(allocator, prefix, "mlp", config, stream);
        const input_layernorm = try RMSNorm.init(allocator, prefix, "input_layernorm", config.rms_norm_eps, stream);
        const post_attention_layernorm = try RMSNorm.init(allocator, prefix, "post_attention_layernorm", config.rms_norm_eps, stream);
        return TransformerBlock{
            .attention = attention,
            .mlp = mlp,
            .input_layernorm = input_layernorm,
            .post_attention_layernorm = post_attention_layernorm,
            .stream = stream,
            .allocator = allocator,
            .prefix = prefix,
        };
    }

    pub fn load(self: *TransformerBlock, weights_map: *const c.mlx_map_string_to_array) !void {
        try self.attention.load(weights_map);
        try self.mlp.load(weights_map);
        try self.input_layernorm.load(weights_map);
        try self.post_attention_layernorm.load(weights_map);
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
        self.allocator.free(self.prefix);
    }
};

pub const LlamaModel = struct {
    embed_tokens: Embedding,
    layers: []TransformerBlock,
    norm: RMSNorm,
    stream: c.mlx_stream,
    allocator: std.mem.Allocator,
    prefix: []const u8,

    pub fn init(allocator: std.mem.Allocator, parent_prefix: []const u8, config: *const LlamaConfig, stream: c.mlx_stream) !LlamaModel {
        const prefix = try allocator.dupe(u8, parent_prefix);
        const embed_tokens = try Embedding.init(allocator, prefix, config, stream);
        const norm = try RMSNorm.init(allocator, prefix, "norm", config.rms_norm_eps, stream);
        const layers = try allocator.alloc(TransformerBlock, @intCast(config.num_hidden_layers));
        const layers_prefix = try joinPath(allocator, prefix, "layers");
        defer allocator.free(layers_prefix);
        for (0..@intCast(config.num_hidden_layers)) |i| {
            layers[i] = try TransformerBlock.init(allocator, layers_prefix, i, config, stream);
        }
        return LlamaModel{
            .embed_tokens = embed_tokens,
            .layers = layers,
            .norm = norm,
            .stream = stream,
            .allocator = allocator,
            .prefix = prefix,
        };
    }

    pub fn load(self: *LlamaModel, weights_map: *const c.mlx_map_string_to_array) !void {
        try self.embed_tokens.load(weights_map);
        for (self.layers) |*layer| {
            try layer.load(weights_map);
        }
        try self.norm.load(weights_map);
    }

    pub fn forward(self: *LlamaModel, result: *c.mlx_array, toks: c.mlx_array, mask: ?c.mlx_array, cache: ?*Cache) !void {
        const seq_len = c.mlx_array_dim(toks, 1);
        const offset = if (cache) |c_| c_.offset else 0;
        var x = c.mlx_array_new();
        defer _ = c.mlx_array_free(x);
        try self.embed_tokens.forward(&x, toks);
        for (self.layers, 0..) |*layer, i| {
            const layer_cache = if (cache) |c_| &c_.layers[i] else null;
            try layer.forward(&x, x, mask, layer_cache, offset);
        }
        try self.norm.forward(result, x);
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
        self.allocator.free(self.prefix);
    }
};

pub const Llama = struct {
    model: LlamaModel,
    tie_word_embeddings: bool,
    lm_head: ?Linear,
    stream: c.mlx_stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: *const LlamaConfig, stream: c.mlx_stream) !Llama {
        const model = try LlamaModel.init(allocator, "model", config, stream);
        var lm_head: ?Linear = null;
        if (!config.tie_word_embeddings) {
            lm_head = try Linear.init(allocator, "", "lm_head", config, stream);
        }
        return Llama{
            .model = model,
            .tie_word_embeddings = config.tie_word_embeddings,
            .lm_head = lm_head,
            .stream = stream,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Llama, weights_map: *const c.mlx_map_string_to_array) !void {
        try self.model.load(weights_map);
        if (!self.tie_word_embeddings) {
            try self.lm_head.?.load(weights_map);
        }
    }

    pub fn forward(self: *Llama, result: *c.mlx_array, toks: c.mlx_array, mask: ?c.mlx_array, cache: ?*Cache) !void {
        var x = c.mlx_array_new();
        defer _ = c.mlx_array_free(x);
        try self.model.forward(&x, toks, mask, cache);
        if (self.tie_word_embeddings) {
            try self.model.embed_tokens.asLinear(result, x);
        } else {
            try self.lm_head.?.forward(result, x);
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

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !DefaultTransformer {
        const stream = c.mlx_default_cpu_stream_new();
        const path_model = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{model_path});
        defer allocator.free(path_model);
        const path_config = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_path});
        defer allocator.free(path_config);
        const file = c.fopen(path_model.ptr, "rb") orelse return error.FileNotFound;
        defer _ = c.fclose(file);
        var weights = c.mlx_map_string_to_array_new();
        defer _ = c.mlx_map_string_to_array_free(weights);
        var meta = c.mlx_map_string_to_string_new();
        defer _ = c.mlx_map_string_to_string_free(meta);
        try mlxOp(c.mlx_load_safetensors_file(&weights, &meta, file, stream));
        try printMap("Metadata", &meta);
        var configJson = try loadJsonFile(LlamaConfig, allocator, path_config, true);
        defer configJson.deinit();
        var model = try Llama.init(allocator, &configJson.value, stream);
        try model.load(&weights);
        return DefaultTransformer{
            .allocator = allocator,
            .stream = stream,
            .model = model,
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
            try mlxOp(c.mlx_take(&logits, logits, c.mlx_array_new_int(-1), 1, self.stream));
            try mlxOp(c.mlx_argmax(&logits, logits, 1, false, self.stream));
            try mlxOp(c.mlx_array_item_uint32(&output_tokens[i], logits));
            try mlxOp(c.mlx_array_set_data(&toks, &output_tokens[i], &[_]c_int{ 1, 1 }, 2, c.MLX_UINT32));
            std.debug.print("Generated token {d}/{d}: {d}\n", .{ i + 1, num_tokens, output_tokens[i] });
        }
        return output_tokens;
    }
};

const QuantConfig = struct {
    group_size: c_int = 64,
    bits: c_int = 4,
};

const RopeScalingConfig = struct {
    factor: f32 = 32.0,
    high_freq_factor: f32 = 4.0,
    low_freq_factor: f32 = 1.0,
    original_max_position_embeddings: c_int = 8192,
};

const LlamaConfig = struct {
    hidden_size: c_int = 2048,
    intermediate_size: c_int = 8192,
    num_attention_heads: c_int = 32,
    num_key_value_heads: c_int = 8,
    head_dim: c_int = 64,
    max_position_embeddings: c_int = 131072,
    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 500000.0,
    rope_scaling: RopeScalingConfig = .{},
    mlp_bias: bool = false,
    attention_bias: bool = false,
    tie_word_embeddings: bool = true,
    vocab_size: c_int = 128256,
    num_hidden_layers: c_int = 16,
    quantization: ?QuantConfig = null,
};

fn mlxOp(result: c_int) !void {
    if (result != 0) return error.MLXOperationFailed;
}

fn printArray(msg: []const u8, arr: c.mlx_array) void {
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

fn printMap(msg: []const u8, map: *c.mlx_map_string_to_string) !void {
    const map_iter = c.mlx_map_string_to_string_iterator_new(map.*);
    defer _ = c.mlx_map_string_to_string_iterator_free(map_iter);
    var key: [*c]const u8 = undefined;
    var value: [*c]const u8 = undefined;
    std.debug.print("{s}:\n", .{msg});
    while (c.mlx_map_string_to_string_iterator_next(&key, &value, map_iter) == 0) std.debug.print("  {s}: {s}\n", .{ key, value });
}

fn createCausalMask(result: *c.mlx_array, seq_len: c_int, offset: c_int, stream: c.mlx_stream) !void {
    const zero = c.mlx_array_new_float(0.0);
    const neg_inf = c.mlx_array_new_float(-std.math.inf(f32));
    var mask = c.mlx_array_new();
    defer {
        _ = c.mlx_array_free(zero);
        _ = c.mlx_array_free(neg_inf);
        _ = c.mlx_array_free(mask);
    }
    try mlxOp(c.mlx_ones(&mask, &[_]c_int{ seq_len, seq_len + offset }, 2, c.MLX_INT32, stream));
    try mlxOp(c.mlx_tril(&mask, mask, offset, stream));
    try mlxOp(c.mlx_where(result, mask, zero, neg_inf, stream));
    try mlxOp(c.mlx_astype(result, result.*, c.MLX_BFLOAT16, stream));
}

pub fn loadJsonFile(comptime T: type, allocator: std.mem.Allocator, filename: []const u8, verbose: bool) !std.json.Parsed(T) {
    const content = try std.fs.cwd().readFileAlloc(allocator, filename, 10 * 1024 * 1024);
    defer allocator.free(content);
    return try loadJsonString(T, allocator, content, verbose);
}

fn loadJsonString(comptime T: type, allocator: std.mem.Allocator, json_string: []const u8, verbose: bool) !std.json.Parsed(T) {
    const parsed = try std.json.parseFromSlice(T, allocator, json_string, .{
        .ignore_unknown_fields = true,
    });
    if (verbose) {
        try printFieldDifferences(T, allocator, json_string);
        try printParsedValue(T, parsed.value, allocator);
    }
    return parsed;
}

fn printParsedValue(comptime T: type, value: T, allocator: std.mem.Allocator) !void {
    var string = std.ArrayList(u8).init(allocator);
    defer string.deinit();
    try std.json.stringify(value, .{ .whitespace = .indent_2 }, string.writer());
    std.debug.print("\nParsed Value:\n", .{});
    std.debug.print("{s}\n", .{string.items});
}

fn printFieldDifferences(comptime T: type, allocator: std.mem.Allocator, json_string: []const u8) !void {
    const struct_info = @typeInfo(T).Struct;
    var generic = try std.json.parseFromSlice(std.json.Value, allocator, json_string, .{});
    defer generic.deinit();
    if (generic.value != .object) return;
    std.debug.print("Ignored fields:\n", .{});
    {
        var found_extra = false;
        var iter = generic.value.object.iterator();
        while (iter.next()) |entry| {
            const field_exists = blk: {
                inline for (struct_info.fields) |field| {
                    if (std.mem.eql(u8, field.name, entry.key_ptr.*)) {
                        break :blk true;
                    }
                }
                break :blk false;
            };

            if (!field_exists) {
                found_extra = true;
                std.debug.print("  - {s}\n", .{entry.key_ptr.*});
            }
        }
        if (!found_extra) {
            std.debug.print("  None\n", .{});
        }
    }
    std.debug.print("Default fields:\n", .{});
    {
        var found_missing = false;
        inline for (struct_info.fields) |field| {
            if (!generic.value.object.contains(field.name)) {
                found_missing = true;
                std.debug.print("  - {s}\n", .{field.name});
            }
        }
        if (!found_missing) {
            std.debug.print("  None\n", .{});
        }
    }
}

fn joinPath(allocator: std.mem.Allocator, prefix: []const u8, suffix: anytype) ![]u8 {
    const T = @TypeOf(suffix);
    const info = @typeInfo(T);
    const fmt = if (info == .Int or info == .ComptimeInt)
        "{d}"
    else
        "{s}";
    return std.fmt.allocPrint(allocator, "{s}." ++ fmt, .{ prefix, suffix });
}

fn joinPaths(allocator: std.mem.Allocator, strings: []const []const u8, separator: []const u8) ![]const u8 {
    var result = std.ArrayList(u8).init(allocator);
    defer result.deinit();
    for (strings) |str| {
        if (str.len > 0) {
            if (result.items.len > 0) try result.appendSlice(separator);
            try result.appendSlice(str);
        }
    }
    return result.toOwnedSlice();
}

fn loadArray(weight: *c.mlx_array, suffix: anytype, prefix: []const u8, weights_map: *const c.mlx_map_string_to_array, allocator: std.mem.Allocator) !void {
    const key = try joinPath(allocator, prefix, suffix);
    defer allocator.free(key);
    try mlxOp(c.mlx_map_string_to_array_get(weight, weights_map.*, key.ptr));
    try mlxOp(c.mlx_array_eval(weight.*));
}

fn loadWeight(weight: *c.mlx_array, suffix: anytype, prefix: []const u8, weights_map: *const c.mlx_map_string_to_array, allocator: std.mem.Allocator) !void {
    const key = try joinPaths(allocator, &[_][]const u8{ prefix, suffix }, ".");
    defer allocator.free(key);
    try loadArray(weight, "weight", key, weights_map, allocator);
}

fn loadQuantized(weight: *c.mlx_array, scales: *c.mlx_array, biases: *c.mlx_array, suffix: anytype, prefix: []const u8, weights_map: *const c.mlx_map_string_to_array, allocator: std.mem.Allocator) !void {
    const key = try joinPaths(allocator, &[_][]const u8{ prefix, suffix }, ".");
    defer allocator.free(key);
    try loadArray(weight, "weight", key, weights_map, allocator);
    try loadArray(scales, "scales", key, weights_map, allocator);
    try loadArray(biases, "biases", key, weights_map, allocator);
}

fn reshap(result: *c.mlx_array, x: c.mlx_array, pattern: []const u8, dim_values: anytype, stream: c.mlx_stream) !void {
    const arrow = std.mem.indexOf(u8, pattern, "->") orelse return error.NoArrow;
    const left_paren = std.mem.indexOf(u8, pattern[0..arrow], "(") != null;
    const side = std.mem.trim(u8, if (left_paren) pattern[0..arrow] else pattern[arrow + 2 ..], " ");
    const p = std.mem.indexOf(u8, side, "(") orelse return error.NoParentheses;
    var shape: [16]c_int = undefined;
    var i: usize = 0;
    var t = std.mem.tokenize(u8, side[0..p], " ");
    while (t.next()) |_| : (i += 1) shape[i] = c.mlx_array_shape(x)[i];
    if (left_paren) {
        const close = std.mem.indexOf(u8, side[p..], ")") orelse return error.UnclosedParenthesis;
        t = std.mem.tokenize(u8, side[p + 1 .. p + close], " ");
        while (t.next()) |dim| {
            inline for (std.meta.fields(@TypeOf(dim_values))) |f| {
                if (std.mem.eql(u8, dim, f.name)) {
                    shape[i] = @field(dim_values, f.name);
                    i += 1;
                }
            }
        }
    } else {
        shape[i] = -1;
        i += 1;
    }
    return mlxOp(c.mlx_reshape(result, x, &shape, i, stream));
}

fn repeat(result: *c.mlx_array, x: c.mlx_array, pattern: []const u8, dim_values: anytype, stream: c.mlx_stream) !void {
    const n_repeat = inline for (std.meta.fields(@TypeOf(dim_values))) |field| {
        break @field(dim_values, field.name);
    } else return error.InvalidNRepeat;
    const output_pattern = std.mem.trim(u8, pattern[(std.mem.indexOf(u8, pattern, "->") orelse return error.InvalidPattern) + 2 .. pattern.len], " ");
    var tokens = std.mem.tokenize(u8, output_pattern, " ");
    const axis = try blk: {
        var i: usize = 0;
        while (tokens.next()) |token| : (i += 1) {
            if (std.mem.indexOfScalar(u8, token, '(') != null) break :blk i;
        }
        break :blk error.NoParenthesisFound;
    };
    try mlxOp(c.mlx_repeat(result, x, @intCast(n_repeat), @intCast(axis), stream));
}

fn einsum(result: *c.mlx_array, arrays: anytype, pattern: [*:0]const u8, stream: c.mlx_stream) !void {
    const info = @typeInfo(@TypeOf(arrays));
    if (info != .Struct or !info.Struct.is_tuple) {
        @compileError("Expected tuple argument for arrays, got " ++ @typeName(@TypeOf(arrays)));
    }
    const fields = info.Struct.fields;
    var array_data: [fields.len]c.mlx_array = undefined;
    inline for (fields, 0..) |field, i| {
        array_data[i] = @field(arrays, field.name);
    }
    const operands = c.mlx_vector_array_new_data(&array_data[0], array_data.len);
    defer _ = c.mlx_vector_array_free(operands);
    try mlxOp(c.mlx_einsum(result, pattern, operands, stream));
}

test "Transformer generating" {
    std.debug.print("\n=== TRANSFORMER.ZIG ===\n\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const initial_tokens = [_]u32{ 9906, 1917 };
    const num_tokens_to_generate = 10;
    var transformer = try DefaultTransformer.init(allocator, "Llama-3.2-1B-Instruct-4bit");
    defer transformer.deinit();
    const generated_tokens = try transformer.generate(&initial_tokens, num_tokens_to_generate);
    defer allocator.free(generated_tokens);
    std.debug.print("\nGenerated sequence: ", .{});
    for (generated_tokens) |token| {
        std.debug.print("{d} ", .{token});
    }
    std.debug.print("\n", .{});
}
