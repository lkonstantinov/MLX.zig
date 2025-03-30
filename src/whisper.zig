const std = @import("std");
const mlx = @import("mlx.zig");
const loadJson = @import("utils.zig").loadJson;
const loadAudio = @import("utils.zig").loadAudio;
const allocJoin = @import("utils.zig").allocJoin;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const formatRange = @import("tokenizer.zig").formatRange;
const formatRangeFloat = @import("tokenizer.zig").formatRangeFloat;

pub const Linear = struct {
    const Self = @This();
    key: []const u8,
    weight: mlx.Weight,
    has_bias: bool,
    bias: ?mlx.Array,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, has_bias: bool, quant_config: ?mlx.QuantConfig, stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        const bias = if (has_bias) mlx.arrayNew() else null;
        return Self{
            .weight = try mlx.Weight.init(quant_config, stream),
            .has_bias = has_bias,
            .bias = bias,
            .stream = stream,
            .key = key,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try self.weight.load(self.key, weights_map);
        if (self.has_bias) {
            try mlx.loadArray(&self.bias.?, self.key, "bias", weights_map);
        }
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        try self.weight.forward(result, x);
        if (self.has_bias) {
            try mlx.add(result, result.*, self.bias.?, self.stream);
        }
    }

    pub fn deinit(self: *Self) void {
        self.weight.deinit();
        if (self.has_bias) {
            _ = mlx.arrayFree(self.bias.?);
        }
        self.allocator.free(self.key);
    }
};

pub const Embedding = struct {
    const Self = @This();
    key: []const u8,
    weight: mlx.Weight,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, quant_config: ?mlx.QuantConfig, stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        return Self{
            .weight = try mlx.Weight.init(quant_config, stream),
            .stream = stream,
            .key = key,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try self.weight.loadDequantized(self.key, weights_map);
    }

    pub fn forward(self: *Self, result: *mlx.Array, toks: mlx.Array) !void {
        try mlx.take(result, self.weight.weight, toks, 0, self.stream);
    }

    pub fn asLinear(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        try self.weight.forward(result, x);
    }

    pub fn deinit(self: *Self) void {
        self.weight.deinit();
        self.allocator.free(self.key);
    }
};

pub const LayerNorm = struct {
    const Self = @This();
    key: []const u8,
    dims: c_int,
    eps: f32,
    weight: mlx.Array,
    bias: mlx.Array,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, dims: c_int, eps: f32, stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        return Self{
            .dims = dims,
            .eps = eps,
            .weight = mlx.arrayNew(),
            .bias = mlx.arrayNew(),
            .stream = stream,
            .key = key,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try mlx.loadArray(&self.weight, self.key, "weight", weights_map);
        try mlx.loadArray(&self.bias, self.key, "bias", weights_map);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        try mlx.mlxOp(mlx.C.mlx_fast_layer_norm(result, x, self.weight, self.bias, self.eps, self.stream));
    }

    pub fn deinit(self: *Self) void {
        mlx.arrayFree(self.weight);
        mlx.arrayFree(self.bias);
        self.allocator.free(self.key);
    }
};

pub const Conv1d = struct {
    const Self = @This();
    key: []const u8,
    in_channels: c_int,
    out_channels: c_int,
    kernel_size: c_int,
    stride: c_int,
    padding: c_int,
    dilation: c_int,
    groups: c_int,
    has_bias: bool,
    weight: mlx.Array,
    bias: ?mlx.Array,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, in_channels: c_int, out_channels: c_int, kernel_size: c_int, stride: c_int, padding: c_int, stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        return Self{
            .in_channels = in_channels,
            .out_channels = out_channels,
            .kernel_size = kernel_size,
            .stride = stride,
            .padding = padding,
            .dilation = 1,
            .groups = 1,
            .has_bias = true,
            .weight = mlx.arrayNew(),
            .bias = mlx.arrayNew(),
            .stream = stream,
            .key = key,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        mlx.arrayFree(self.weight);
        if (self.has_bias) {
            mlx.arrayFree(self.bias.?);
        }
        self.allocator.free(self.key);
    }
    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try mlx.loadArray(&self.weight, self.key, "weight", weights_map);
        try mlx.mlxOp(mlx.C.mlx_swapaxes(&self.weight, self.weight, 1, 2, self.stream));
        if (self.has_bias) {
            try mlx.loadArray(&self.bias.?, self.key, "bias", weights_map);
        }
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        try mlx.mlxOp(mlx.C.mlx_conv1d(result, x, self.weight, self.stride, self.padding, self.dilation, self.groups, self.stream));
        if (self.has_bias) {
            try mlx.add(result, result.*, self.bias.?, self.stream);
        }
    }
};

pub const MultiHeadAttention = struct {
    const Self = @This();
    key: []const u8,
    n_heads: c_int,
    d_model: c_int,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, d_model: c_int, n_heads: c_int, bias_k: bool, stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        const q_proj = try Linear.init(allocator, key, "q_proj", true, null, stream);
        const k_proj = try Linear.init(allocator, key, "k_proj", bias_k, null, stream);
        const v_proj = try Linear.init(allocator, key, "v_proj", true, null, stream);
        const out_proj = try Linear.init(allocator, key, "out_proj", true, null, stream);
        return Self{
            .key = key,
            .n_heads = n_heads,
            .d_model = d_model,
            .q_proj = q_proj,
            .k_proj = k_proj,
            .v_proj = v_proj,
            .out_proj = out_proj,
            .stream = stream,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try self.q_proj.load(weights_map);
        try self.k_proj.load(weights_map);
        try self.v_proj.load(weights_map);
        try self.out_proj.load(weights_map);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, xa: ?mlx.Array, mask: ?mlx.Array, kv_cache: ?*mlx.KVCache) !void {
        var q = mlx.arrayNew();
        var k = mlx.arrayNew();
        var v = mlx.arrayNew();
        var qk = mlx.arrayNew();
        var attn = mlx.arrayNew();
        var wv = mlx.arrayNew();
        defer {
            mlx.arrayFree(q);
            mlx.arrayFree(k);
            mlx.arrayFree(v);
            mlx.arrayFree(qk);
            mlx.arrayFree(attn);
            mlx.arrayFree(wv);
        }
        try self.q_proj.forward(&q, x);
        if (xa) |cross_input| {
            if (kv_cache) |cache| {
                if (!cache.is_empty) {
                    try cache.get(&k, &v);
                } else {
                    try self.k_proj.forward(&k, cross_input);
                    try self.v_proj.forward(&v, cross_input);
                    try cache.set(&k, &v);
                }
            }
        } else {
            try self.k_proj.forward(&k, x);
            try self.v_proj.forward(&v, x);

            if (kv_cache) |cache| {
                try cache.update(&k, &v, null, self.stream);
            }
        }
        const head_dim = @divExact(self.d_model, self.n_heads);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        try mlx.rEshap(&q, q, "b l (h d) -> b l h d", .{ .h = self.n_heads, .d = head_dim }, self.stream);
        try mlx.rEshap(&k, k, "b l (h d) -> b l h d", .{ .h = self.n_heads, .d = head_dim }, self.stream);
        try mlx.rEshap(&v, v, "b l (h d) -> b l h d", .{ .h = self.n_heads, .d = head_dim }, self.stream);
        try mlx.einsum(&q, .{q}, "b l h d -> b h l d", self.stream);
        try mlx.einsum(&k, .{k}, "b l h d -> b h l d", self.stream);
        try mlx.einsum(&v, .{v}, "b l h d -> b h l d", self.stream);
        try mlx.multiply(&q, q, mlx.float(scale), self.stream);
        try mlx.einsum(&qk, .{ q, k }, "b h l d, b h k d -> b h l k", self.stream);
        if (mask) |attention_mask| {
            try mlx.add(&qk, qk, attention_mask, self.stream);
        }
        try mlx.softmax(&attn, qk, &[_]c_int{3}, true, self.stream);
        try mlx.einsum(&wv, .{ attn, v }, "b h l k, b h k d -> b h l d", self.stream);
        try mlx.einsum(&wv, .{wv}, "b h l d -> b l h d", self.stream);
        try mlx.rEshap(&wv, wv, "b l h d -> b l (h d)", .{}, self.stream);
        try self.out_proj.forward(result, wv);
    }

    pub fn deinit(self: *Self) void {
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.out_proj.deinit();
    }
};

pub const ResidualAttentionBlock = struct {
    const Self = @This();
    key: []const u8,
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: LayerNorm,
    cross_attn: ?MultiHeadAttention,
    cross_attn_layer_norm: ?LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, d_model: c_int, n_head: c_int, cross_attention: bool, stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        const self_attn = try MultiHeadAttention.init(allocator, key, "self_attn", d_model, n_head, false, stream);
        const self_attn_layer_norm = try LayerNorm.init(allocator, key, "self_attn_layer_norm", d_model, 1e-5, stream);
        const cross_attn = if (cross_attention) try MultiHeadAttention.init(allocator, key, "encoder_attn", d_model, n_head, false, stream) else null;
        const cross_attn_layer_norm = if (cross_attention) try LayerNorm.init(allocator, key, "encoder_attn_layer_norm", d_model, 1e-5, stream) else null;
        const fc1 = try Linear.init(allocator, key, "fc1", true, null, stream);
        const fc2 = try Linear.init(allocator, key, "fc2", true, null, stream);
        const final_layer_norm = try LayerNorm.init(allocator, key, "final_layer_norm", d_model, 1e-5, stream);
        return Self{
            .self_attn = self_attn,
            .self_attn_layer_norm = self_attn_layer_norm,
            .cross_attn = cross_attn,
            .cross_attn_layer_norm = cross_attn_layer_norm,
            .fc1 = fc1,
            .fc2 = fc2,
            .final_layer_norm = final_layer_norm,
            .stream = stream,
            .allocator = allocator,
            .key = key,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try self.self_attn.load(weights_map);
        try self.self_attn_layer_norm.load(weights_map);
        if (self.cross_attn) |*cross_attn| {
            try cross_attn.load(weights_map);
        }
        if (self.cross_attn_layer_norm) |*cross_attn_layer_norm| {
            try cross_attn_layer_norm.load(weights_map);
        }
        try self.fc1.load(weights_map);
        try self.fc2.load(weights_map);
        try self.final_layer_norm.load(weights_map);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, xa: ?mlx.Array, mask: ?mlx.Array, kv_cache: ?*mlx.KVCache, cross_kv_cache: ?*mlx.KVCache) !void {
        var attn_norm = mlx.arrayNew();
        var attn = mlx.arrayNew();
        var cross_attn_norm = mlx.arrayNew();
        var cross_out = mlx.arrayNew();
        var ff_norm = mlx.arrayNew();
        var fc1_out = mlx.arrayNew();
        var gelu_out = mlx.arrayNew();
        var fc2_out = mlx.arrayNew();
        defer {
            mlx.arrayFree(attn_norm);
            mlx.arrayFree(attn);
            mlx.arrayFree(cross_attn_norm);
            mlx.arrayFree(cross_out);
            mlx.arrayFree(ff_norm);
            mlx.arrayFree(fc1_out);
            mlx.arrayFree(gelu_out);
            mlx.arrayFree(fc2_out);
        }
        try self.self_attn_layer_norm.forward(&attn_norm, x);
        try self.self_attn.forward(&attn, attn_norm, null, mask, kv_cache);
        try mlx.add(&attn, x, attn, self.stream);
        if (self.cross_attn != null and self.cross_attn_layer_norm != null) {
            if (xa) |encoder_out| {
                try self.cross_attn_layer_norm.?.forward(&cross_attn_norm, attn);
                try self.cross_attn.?.forward(&cross_out, cross_attn_norm, encoder_out, null, cross_kv_cache);
                try mlx.add(&attn, attn, cross_out, self.stream);
            }
        }
        try self.final_layer_norm.forward(&ff_norm, attn);
        try self.fc1.forward(&fc1_out, ff_norm);
        try mlx.gelu(&gelu_out, fc1_out, self.stream);
        try self.fc2.forward(&fc2_out, gelu_out);
        try mlx.add(result, attn, fc2_out, self.stream);
    }

    pub fn deinit(self: *Self) void {
        self.self_attn.deinit();
        self.self_attn_layer_norm.deinit();
        if (self.cross_attn) |*cross_attn| {
            cross_attn.deinit();
        }
        if (self.cross_attn_layer_norm) |*cross_attn_layer_norm| {
            cross_attn_layer_norm.deinit();
        }
        self.fc1.deinit();
        self.fc2.deinit();
        self.final_layer_norm.deinit();
        self.allocator.free(self.key);
    }
};

pub const AudioEncoder = struct {
    const Self = @This();
    key: []const u8,
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: mlx.Array,
    layers: []ResidualAttentionBlock,
    layer_norm: LayerNorm,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, config: *const WhisperConfig, stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        const conv1 = try Conv1d.init(allocator, key, "conv1", config.num_mel_bins, config.d_model, 3, 1, 1, stream);
        const conv2 = try Conv1d.init(allocator, key, "conv2", config.d_model, config.d_model, 3, 2, 1, stream);
        var positional_embedding = mlx.arrayNew();
        try createSinusoids(&positional_embedding, config.max_source_positions, config.d_model, stream);
        try mlx.astype(&positional_embedding, positional_embedding, mlx.DTYPE.FLOAT16, stream);
        const layer_norm = try LayerNorm.init(allocator, key, "layer_norm", config.d_model, 1e-5, stream);
        var layers = try allocator.alloc(ResidualAttentionBlock, @intCast(config.encoder_layers));
        errdefer allocator.free(layers);
        const layers_key = try allocJoin(allocator, key, "layers");
        defer allocator.free(layers_key);
        for (0..@intCast(config.encoder_layers)) |i| {
            const layer_name = try std.fmt.allocPrint(allocator, "{d}", .{i});
            defer allocator.free(layer_name);
            layers[i] = try ResidualAttentionBlock.init(allocator, layers_key, layer_name, config.d_model, config.encoder_attention_heads, false, stream);
        }
        return Self{
            .conv1 = conv1,
            .conv2 = conv2,
            .positional_embedding = positional_embedding,
            .layers = layers,
            .layer_norm = layer_norm,
            .stream = stream,
            .allocator = allocator,
            .key = key,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try self.conv1.load(weights_map);
        try self.conv2.load(weights_map);
        for (self.layers) |*layer| {
            try layer.load(weights_map);
        }
        try self.layer_norm.load(weights_map);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        var conv1_out = mlx.arrayNew();
        var conv2_out = mlx.arrayNew();
        var hidden = mlx.arrayNew();
        defer {
            mlx.arrayFree(conv1_out);
            mlx.arrayFree(conv2_out);
            mlx.arrayFree(hidden);
        }
        try self.conv1.forward(&conv1_out, x);
        try mlx.gelu(&hidden, conv1_out, self.stream);
        try self.conv2.forward(&conv2_out, hidden);
        try mlx.gelu(&hidden, conv2_out, self.stream);
        try mlx.add(&hidden, hidden, self.positional_embedding, self.stream);
        for (self.layers) |*layer| {
            try layer.forward(&hidden, hidden, null, null, null, null);
        }
        try self.layer_norm.forward(result, hidden);
    }

    pub fn deinit(self: *Self) void {
        self.conv1.deinit();
        self.conv2.deinit();
        mlx.arrayFree(self.positional_embedding);
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.layer_norm.deinit();
        self.allocator.free(self.key);
    }
};

pub const TextDecoder = struct {
    const Self = @This();
    key: []const u8,
    embed_tokens: Embedding,
    positional_embedding: mlx.Array,
    layers: []ResidualAttentionBlock,
    layer_norm: LayerNorm,
    mask: mlx.Array,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, config: *const WhisperConfig, stream: mlx.Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        const embed_tokens = try Embedding.init(allocator, key, "embed_tokens", null, stream);
        const positional_embedding = mlx.arrayNew();
        const layer_norm = try LayerNorm.init(allocator, key, "layer_norm", config.d_model, 1e-5, stream);
        var mask = mlx.arrayNew();
        try mlx.createCausalMask(&mask, config.max_target_positions, 0, stream);
        var layers = try allocator.alloc(ResidualAttentionBlock, @intCast(config.decoder_layers));
        errdefer allocator.free(layers);
        const layers_key = try allocJoin(allocator, key, "layers");
        defer allocator.free(layers_key);
        for (0..@intCast(config.decoder_layers)) |i| {
            const layer_name = try std.fmt.allocPrint(allocator, "{d}", .{i});
            defer allocator.free(layer_name);
            layers[i] = try ResidualAttentionBlock.init(allocator, layers_key, layer_name, config.d_model, config.decoder_attention_heads, true, stream);
        }
        return Self{
            .embed_tokens = embed_tokens,
            .positional_embedding = positional_embedding,
            .layers = layers,
            .layer_norm = layer_norm,
            .mask = mask,
            .stream = stream,
            .allocator = allocator,
            .key = key,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try self.embed_tokens.load(weights_map);
        try mlx.loadArray(&self.positional_embedding, self.key, "embed_positions.weight", weights_map);
        for (self.layers) |*layer| {
            try layer.load(weights_map);
        }
        try self.layer_norm.load(weights_map);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, xa: mlx.Array, mask: mlx.Array, kv_cache: *mlx.Cache, cross_kv_cache: *mlx.Cache) !void {
        var embedded = mlx.arrayNew();
        var hidden = mlx.arrayNew();
        defer {
            mlx.arrayFree(embedded);
            mlx.arrayFree(hidden);
        }
        const offset = kv_cache.offset;
        try self.embed_tokens.forward(&embedded, x);
        var pos_embed_slice = mlx.arrayNew();
        defer mlx.arrayFree(pos_embed_slice);
        const start = [_]c_int{ offset, 0 };
        const stop = [_]c_int{ offset + mlx.arrayDim(x, 1), mlx.arrayDim(self.positional_embedding, 1) };
        const strides = [_]c_int{ 1, 1 };
        const ndim = 2;
        try mlx.mlxOp(mlx.C.mlx_slice(&pos_embed_slice, self.positional_embedding, &start, ndim, &stop, ndim, &strides, ndim, self.stream));
        try mlx.add(&hidden, embedded, pos_embed_slice, self.stream);
        for (self.layers, 0..) |*layer, i| {
            _ = try layer.forward(&hidden, hidden, xa, mask, &kv_cache.layers[i], &cross_kv_cache.layers[i]);
        }
        try self.layer_norm.forward(&hidden, hidden);
        try self.embed_tokens.asLinear(result, hidden);
    }

    pub fn deinit(self: *Self) void {
        self.embed_tokens.deinit();
        mlx.arrayFree(self.positional_embedding);
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.layer_norm.deinit();
        mlx.arrayFree(self.mask);
        self.allocator.free(self.key);
    }
};

pub const Whisper = struct {
    const Self = @This();
    encoder: AudioEncoder,
    decoder: TextDecoder,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: *const WhisperConfig, stream: mlx.Stream) !Self {
        const encoder = try AudioEncoder.init(allocator, "model", "encoder", config, stream);
        const decoder = try TextDecoder.init(allocator, "model", "decoder", config, stream);
        return Self{
            .encoder = encoder,
            .decoder = decoder,
            .stream = stream,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const mlx.MapStrArr) !void {
        try self.encoder.load(weights_map);
        try self.decoder.load(weights_map);
    }

    pub fn encode(self: *Self, result: *mlx.Array, mel: mlx.Array) !void {
        try self.encoder.forward(result, mel);
    }

    pub fn decode(self: *Self, result: *mlx.Array, txt: mlx.Array, mel: mlx.Array, mask: mlx.Array, kv_cache: *mlx.Cache, cross_kv_cache: *mlx.Cache) !void {
        try self.decoder.forward(result, txt, mel, mask, kv_cache, cross_kv_cache);
    }

    pub fn deinit(self: *Self) void {
        self.encoder.deinit();
        self.decoder.deinit();
    }
};

pub const Transcriber = struct {
    const Self = @This();
    model: Whisper,
    tokenizer: Tokenizer,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        var buf: [1024]u8 = undefined;
        const stream = mlx.defaultCpuStreamNew();
        const path_config = try std.fmt.bufPrintZ(&buf, "{s}/config.json", .{model_path});
        const config = try loadJson(WhisperConfig, allocator, path_config, true);
        defer config.deinit();
        var model = try Whisper.init(allocator, &config.value, stream);
        const pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
        const specials = [_][]const u8{
            "<|endoftext|>",
            "<|startoftranscript|>",
        } ++ formatRange("<|_{d}|>", 0, 100) ++ [_][]const u8{
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nospeech|>",
            "<|notimestamps|>",
        } ++ formatRangeFloat(1501);
        const tokenizer = try Tokenizer.initFromTikToken(allocator, pattern, "multilingual.tiktoken", &specials);
        const path_weight = try std.fmt.bufPrintZ(&buf, "{s}/model.safetensors", .{model_path});
        var safetensors = try mlx.Safetensors.load(path_weight, stream);
        defer safetensors.deinit();
        try model.load(&safetensors.weights);
        return Self{
            .model = model,
            .tokenizer = tokenizer,
            .stream = stream,
            .allocator = allocator,
        };
    }

    pub fn transcribe(self: *Self, audio_path: []const u8) ![]const u8 {
        const audio = try loadAudio(self.allocator, audio_path);
        defer self.allocator.free(audio);
        var mel_all = mlx.arrayNew();
        defer mlx.arrayFree(mel_all);
        var mel = mlx.arrayNew();
        defer mlx.arrayFree(mel);
        try getMel(&mel_all, audio, self.stream);
        var new_tok = std.ArrayList(u32).init(self.allocator);
        defer new_tok.deinit();
        var i: c_int = 0;
        const mel_len = mlx.arrayDim(mel_all, 1);
        while (i + 3000 < mel_len) {
            try mlx.slice(&mel, mel_all, &[_]c_int{ 0, i, 0 }, &[_]c_int{ 1, i + 3000, 128 }, &[_]c_int{ 1, 1, 1 }, self.stream);
            const piece = try self.step(mel);
            defer self.allocator.free(piece);
            const arg_hop = std.mem.indexOfMax(u32, piece);
            const hop = (piece[arg_hop] - 50365) * 2;
            try new_tok.appendSlice(piece[0..arg_hop]);
            i += if (hop > 0) @intCast(hop) else 3000;
        }
        var filtered_tokens = std.ArrayList(u32).init(self.allocator);
        defer filtered_tokens.deinit();
        for (new_tok.items) |token| {
            if (token < 50257) {
                try filtered_tokens.append(token);
            }
        }
        return try self.tokenizer.decode(filtered_tokens.items);
    }

    fn step(self: *Self, raw: mlx.Array) ![]u32 {
        var enc = mlx.arrayNew();
        defer mlx.arrayFree(enc);
        try self.model.encoder.forward(&enc, raw);
        var kv_cache = try mlx.Cache.init(self.allocator, self.model.decoder.layers.len, 1);
        defer kv_cache.deinit();
        var cross_kv_cache = try mlx.Cache.init(self.allocator, self.model.decoder.layers.len, 1);
        defer cross_kv_cache.deinit();
        var output_tokens = try self.allocator.alloc(u32, 500);
        errdefer self.allocator.free(output_tokens);
        var toks = try mlx.arrayNewData(&[_]u32{ 50258, 50360, 50365 }, .{ 1, 3 }, mlx.DTYPE.UINT32);
        defer mlx.arrayFree(toks);
        var logits = mlx.arrayNew();
        defer mlx.arrayFree(logits);
        var mask = mlx.arrayNew();
        defer mlx.arrayFree(mask);
        var i: usize = 0;
        while (i < 446) : (i += 1) {
            try mlx.createCausalMask(&mask, mlx.arrayDim(toks, 1), kv_cache.offset, self.stream);
            try self.model.decoder.forward(&logits, toks, enc, mask, &kv_cache, &cross_kv_cache);
            try mlx.take(&logits, logits, mlx.int(-1), 1, self.stream);
            try mlx.argmax(&logits, logits, 1, false, self.stream);
            try mlx.item(&output_tokens[i], logits);
            kv_cache.offset += mlx.arrayDim(toks, 1);
            try mlx.arraySetData(&toks, &output_tokens[i], .{ 1, 1 }, mlx.DTYPE.UINT32);
            if (output_tokens[i] == 50257) {
                i += 1;
                break;
            }
            std.debug.print("\nGenerated at {d}: {d}\n", .{ i, output_tokens[i] });
        }
        if (i < 500) {
            output_tokens = try self.allocator.realloc(output_tokens, i);
        }
        return output_tokens;
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        self.tokenizer.deinit();
        _ = mlx.streamFree(self.stream);
    }
};

fn getMel(result: *mlx.Array, audio_raw: []f32, stream: mlx.Stream) !void {
    var melp = mlx.arrayNew();
    defer mlx.arrayFree(melp);
    var hanp = mlx.arrayNew();
    defer mlx.arrayFree(hanp);
    var safetensors = try mlx.Safetensors.load("whisper_precomputed.safetensors", stream);
    defer safetensors.deinit();
    try mlx.loadArray(&melp, "melp", null, &safetensors.weights);
    try mlx.loadArray(&hanp, "hanp", null, &safetensors.weights);
    var audio = try mlx.arrayNewData(audio_raw.ptr, .{audio_raw.len}, mlx.DTYPE.FLOAT32);
    defer mlx.arrayFree(audio);
    var threshold = mlx.arrayNew();
    defer mlx.arrayFree(threshold);
    try mlx.pad(&audio, audio, &[_]c_int{0}, &[_]c_int{0}, &[_]c_int{480000}, mlx.float(0.0), "constant", stream);
    var slice1 = mlx.arrayNew();
    defer mlx.arrayFree(slice1);
    var slice2 = mlx.arrayNew();
    defer mlx.arrayFree(slice2);
    try mlx.slice(&slice1, audio, &[_]c_int{200}, &[_]c_int{0}, &[_]c_int{-1}, stream);
    try mlx.slice(&slice2, audio, &[_]c_int{-2}, &[_]c_int{-202}, &[_]c_int{-1}, stream);
    try mlx.concatenate(&audio, .{ slice1, audio, slice2 }, 0, stream);
    try mlx.as_strided(&audio, audio, &[_]c_int{ @divTrunc(mlx.arrayDim(audio, 0) - 240, 160), 400 }, &[_]u64{ 160, 1 }, 0, stream);
    try mlx.multiply(&audio, audio, hanp, stream);
    try mlx.rfft(&audio, audio, 400, 1, stream);
    try mlx.slice(&audio, audio, &[_]c_int{ 0, 0 }, &[_]c_int{ mlx.C.mlx_array_dim(audio, 0) - 1, mlx.C.mlx_array_dim(audio, 1) }, &[_]c_int{ 1, 1 }, stream);
    try mlx.abs(&audio, audio, stream);
    try mlx.square(&audio, audio, stream);
    try mlx.matmul(&audio, audio, melp, stream);
    try mlx.maximum(&audio, audio, mlx.float(1e-10), stream);
    try mlx.log10(&audio, audio, stream);
    try mlx.max_all(&threshold, audio, false, stream);
    try mlx.subtract(&threshold, threshold, mlx.float(8.0), stream);
    try mlx.maximum(&audio, audio, threshold, stream);
    try mlx.add(&audio, audio, mlx.float(4.0), stream);
    try mlx.divide(&audio, audio, mlx.float(4.0), stream);
    try mlx.expand_dims(&audio, audio, &[_]c_int{0}, stream);
    try mlx.astype(result, audio, mlx.DTYPE.FLOAT16, stream);
}

fn createSinusoids(result: *mlx.Array, length: c_int, channels: c_int, stream: mlx.Stream) !void {
    var inv_timescales = mlx.arrayNew();
    defer mlx.arrayFree(inv_timescales);
    var scaled_time = mlx.arrayNew();
    defer mlx.arrayFree(scaled_time);
    var sin_val = mlx.arrayNew();
    defer mlx.arrayFree(sin_val);
    var cos_val = mlx.arrayNew();
    defer mlx.arrayFree(cos_val);
    const log_timescale_increment = std.math.log(f32, std.math.e, 10000) / (@as(f32, @floatFromInt(@divExact(channels, 2))) - 1.0);
    try mlx.arange(&inv_timescales, 0, @floatFromInt(@divExact(channels, 2)), 1, mlx.DTYPE.FLOAT32, stream);
    try mlx.multiply(&inv_timescales, mlx.float(-log_timescale_increment), inv_timescales, stream);
    try mlx.exp(&inv_timescales, inv_timescales, stream);
    try mlx.arange(&scaled_time, 0, @floatFromInt(length), 1, mlx.DTYPE.FLOAT32, stream);
    try mlx.einsum(&scaled_time, .{ scaled_time, inv_timescales }, "i,j->ij", stream);
    try mlx.sin(&sin_val, scaled_time, stream);
    try mlx.cos(&cos_val, scaled_time, stream);
    try mlx.concatenate(result, .{ sin_val, cos_val }, 1, stream);
}

const WhisperConfig = struct {
    d_model: c_int,
    encoder_layers: c_int,
    encoder_attention_heads: c_int,
    decoder_layers: c_int,
    decoder_attention_heads: c_int,
    max_source_positions: c_int,
    max_target_positions: c_int,
    vocab_size: c_int,
    num_mel_bins: c_int,
};
