//! whisper.zig - Whisper-Turbo-Large-v3
//!
//! Copyright 2025 Joe

const std = @import("std");
const mlx = @import("mlx.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const loadJson = @import("utils.zig").loadJson;
const loadAudio = @import("utils.zig").loadAudio;
const allocJoin = @import("utils.zig").allocJoin;
const formatRange = @import("utils.zig").formatRange;
const formatRangeFloat = @import("utils.zig").formatRangeFloat;

pub const MultiHeadAttention = struct {
    const Self = @This();
    n_heads: c_int,
    d_model: c_int,
    head_dim: c_int,
    scale: mlx.Array,
    q_proj: *mlx.Linear,
    k_proj: *mlx.Linear,
    v_proj: *mlx.Linear,
    out_proj: *mlx.Linear,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,
    allocs_to_free: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, d_model: c_int, n_heads: c_int, bias_k: bool, weights_hash: *std.StringHashMap(*mlx.Array), stream: mlx.Stream) !Self {
        var allocs_to_free = std.ArrayList([]const u8).init(allocator);
        const key = try allocJoin(allocator, parent, name);
        try allocs_to_free.append(key);
        const q_proj = try mlx.Linear.init(allocator, key, "q_proj", true, null, weights_hash, stream);
        const k_proj = try mlx.Linear.init(allocator, key, "k_proj", bias_k, null, weights_hash, stream);
        const v_proj = try mlx.Linear.init(allocator, key, "v_proj", true, null, weights_hash, stream);
        const out_proj = try mlx.Linear.init(allocator, key, "out_proj", true, null, weights_hash, stream);
        const head_dim = @divExact(d_model, n_heads);
        const scale = mlx.arrayNewFloat(1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))));
        return .{
            .n_heads = n_heads,
            .d_model = d_model,
            .q_proj = q_proj,
            .k_proj = k_proj,
            .v_proj = v_proj,
            .out_proj = out_proj,
            .stream = stream,
            .allocator = allocator,
            .head_dim = head_dim,
            .scale = scale,
            .allocs_to_free = allocs_to_free,
        };
    }

    pub fn deinit(self: *Self) void {
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.out_proj.deinit();
        mlx.arrayFree(self.scale);
        for (self.allocs_to_free.items) |allocation| self.allocator.free(allocation);
        self.allocs_to_free.deinit();
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, xa: ?mlx.Array, mask: ?mlx.Array, kv_cache: ?*mlx.KVCache) !void {
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
        try mlx.rEshap(&q, q, "b l (h d) -> b h l d", .{ .h = self.n_heads, .d = self.head_dim }, self.stream);
        try mlx.rEshap(&k, k, "b l (h d) -> b h l d", .{ .h = self.n_heads, .d = self.head_dim }, self.stream);
        try mlx.rEshap(&v, v, "b l (h d) -> b h l d", .{ .h = self.n_heads, .d = self.head_dim }, self.stream);
        // try mlx.fastScaledDotProductAttention(result, q, k, v, self.scale, mask, self.stream); // : is slower than naive SDPA impl.. weird.
        try mlx.multiply(&q, q, self.scale, self.stream);
        try mlx.einsum(&w, .{ q, k }, "b h l d, b h k d -> b h l k", self.stream);
        if (mask) |attention_mask| try mlx.add(&w, w, attention_mask, self.stream);
        try mlx.softmax(&w, w, &.{3}, true, self.stream);
        try mlx.einsum(result, .{ w, v }, "b h l k, b h k d -> b h l d", self.stream);
        try mlx.rEshap(result, result.*, "b h l d -> b l (h d)", .{}, self.stream);
        try self.out_proj.forward(result, result.*);
    }
};

pub const ResidualAttentionBlock = struct {
    const Self = @This();
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: *mlx.LayerNorm,
    cross_attn: ?MultiHeadAttention,
    cross_attn_layer_norm: ?*mlx.LayerNorm,
    fc1: *mlx.Linear,
    fc2: *mlx.Linear,
    final_layer_norm: *mlx.LayerNorm,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,
    allocs_to_free: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: usize, d_model: c_int, n_head: c_int, cross_attention: bool, weights_hash: *std.StringHashMap(*mlx.Array), stream: mlx.Stream) !Self {
        var allocs_to_free = std.ArrayList([]const u8).init(allocator);
        const key = try allocJoin(allocator, parent, name);
        try allocs_to_free.append(key);
        const self_attn = try MultiHeadAttention.init(allocator, key, "self_attn", d_model, n_head, false, weights_hash, stream);
        const self_attn_layer_norm = try mlx.LayerNorm.init(allocator, key, "self_attn_layer_norm", 1e-5, weights_hash, stream);
        const cross_attn = if (cross_attention) try MultiHeadAttention.init(allocator, key, "encoder_attn", d_model, n_head, false, weights_hash, stream) else null;
        const cross_attn_layer_norm = if (cross_attention) try mlx.LayerNorm.init(allocator, key, "encoder_attn_layer_norm", 1e-5, weights_hash, stream) else null;
        const fc1 = try mlx.Linear.init(allocator, key, "fc1", true, null, weights_hash, stream);
        const fc2 = try mlx.Linear.init(allocator, key, "fc2", true, null, weights_hash, stream);
        const final_layer_norm = try mlx.LayerNorm.init(allocator, key, "final_layer_norm", 1e-5, weights_hash, stream);
        return .{
            .self_attn = self_attn,
            .self_attn_layer_norm = self_attn_layer_norm,
            .cross_attn = cross_attn,
            .cross_attn_layer_norm = cross_attn_layer_norm,
            .fc1 = fc1,
            .fc2 = fc2,
            .final_layer_norm = final_layer_norm,
            .stream = stream,
            .allocator = allocator,
            .allocs_to_free = allocs_to_free,
        };
    }

    pub fn deinit(self: *Self) void {
        self.self_attn.deinit();
        self.self_attn_layer_norm.deinit();
        if (self.cross_attn) |*cross_attn| {
            cross_attn.deinit();
        }
        if (self.cross_attn_layer_norm) |cross_attn_layer_norm| {
            cross_attn_layer_norm.deinit();
        }
        self.fc1.deinit();
        self.fc2.deinit();
        self.final_layer_norm.deinit();
        for (self.allocs_to_free.items) |allocation| self.allocator.free(allocation);
        self.allocs_to_free.deinit();
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, xa: ?mlx.Array, mask: ?mlx.Array, kv_cache: ?*mlx.KVCache, cross_kv_cache: ?*mlx.KVCache) !void {
        var attn = mlx.arrayNew();
        var xatn = mlx.arrayNew();
        defer {
            mlx.arrayFree(attn);
            mlx.arrayFree(xatn);
        }
        try self.self_attn_layer_norm.forward(&attn, x);
        try self.self_attn.forward(&attn, attn, null, mask, kv_cache);
        try mlx.add(&attn, x, attn, self.stream);
        if (self.cross_attn != null and self.cross_attn_layer_norm != null) {
            if (xa) |encoder_out| {
                try self.cross_attn_layer_norm.?.forward(&xatn, attn);
                try self.cross_attn.?.forward(&xatn, xatn, encoder_out, null, cross_kv_cache);
                try mlx.add(&attn, xatn, attn, self.stream);
            }
        }
        try self.final_layer_norm.forward(&xatn, attn);
        try self.fc1.forward(&xatn, xatn);
        try mlx.gelu(&xatn, xatn, self.stream);
        try self.fc2.forward(&xatn, xatn);
        try mlx.add(result, xatn, attn, self.stream);
    }
};

pub const AudioEncoder = struct {
    const Self = @This();
    conv1: *mlx.Conv1d,
    conv2: *mlx.Conv1d,
    positional_embedding: mlx.Array,
    layers: []ResidualAttentionBlock,
    layer_norm: *mlx.LayerNorm,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,
    allocs_to_free: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, config: *const WhisperConfig, weights_hash: *std.StringHashMap(*mlx.Array), stream: mlx.Stream) !*Self {
        var allocs_to_free = std.ArrayList([]const u8).init(allocator);
        const key = try allocJoin(allocator, parent, name);
        try allocs_to_free.append(key);
        const conv1 = try mlx.Conv1d.init(allocator, key, "conv1", 1, 1, 1, 1, true, weights_hash, stream);
        const conv2 = try mlx.Conv1d.init(allocator, key, "conv2", 2, 1, 1, 1, true, weights_hash, stream);
        const layer_norm = try mlx.LayerNorm.init(allocator, key, "layer_norm", 1e-5, weights_hash, stream);
        var layers = try allocator.alloc(ResidualAttentionBlock, @intCast(config.encoder_layers));
        const layers_key = try allocJoin(allocator, key, "layers");
        try allocs_to_free.append(layers_key);
        for (0..@intCast(config.encoder_layers)) |i| {
            layers[i] = try ResidualAttentionBlock.init(allocator, layers_key, i, config.d_model, config.encoder_attention_heads, false, weights_hash, stream);
        }
        var self = try allocator.create(Self);
        self.* = .{
            .conv1 = conv1,
            .conv2 = conv2,
            .positional_embedding = mlx.arrayNew(),
            .layers = layers,
            .layer_norm = layer_norm,
            .stream = stream,
            .allocator = allocator,
            .allocs_to_free = allocs_to_free,
        };

        // var positional_embedding = mlx.arrayNew();
        // try createSinusoids(&positional_embedding, config.max_source_positions, config.d_model, stream);
        // try mlx.astype(&positional_embedding, positional_embedding, mlx.DTYPE.FLOAT16, stream);

        const pe_key = try allocJoin(allocator, key, "embed_positions.weight");
        try self.allocs_to_free.append(pe_key);
        try weights_hash.put(pe_key, &self.positional_embedding);
        return self;
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
        for (self.allocs_to_free.items) |allocation| self.allocator.free(allocation);
        self.allocs_to_free.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        try self.conv1.forward(result, x);
        try mlx.gelu(result, result.*, self.stream);
        try self.conv2.forward(result, result.*);
        try mlx.gelu(result, result.*, self.stream);
        try mlx.add(result, result.*, self.positional_embedding, self.stream);
        for (self.layers) |*layer| {
            try layer.forward(result, result.*, null, null, null, null);
        }
        try self.layer_norm.forward(result, result.*);
    }
};

pub const TextDecoder = struct {
    const Self = @This();
    embed_tokens: *mlx.Embedding,
    positional_embedding: mlx.Array,
    layers: []ResidualAttentionBlock,
    layer_norm: *mlx.LayerNorm,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,
    allocs_to_free: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, config: *const WhisperConfig, weights_hash: *std.StringHashMap(*mlx.Array), stream: mlx.Stream) !*Self {
        var allocs_to_free = std.ArrayList([]const u8).init(allocator);
        const key = try allocJoin(allocator, parent, name);
        try allocs_to_free.append(key);
        const embed_tokens = try mlx.Embedding.init(allocator, key, "embed_tokens", null, weights_hash, stream);
        const layer_norm = try mlx.LayerNorm.init(allocator, key, "layer_norm", 1e-5, weights_hash, stream);
        var layers = try allocator.alloc(ResidualAttentionBlock, @intCast(config.decoder_layers));
        const layers_key = try allocJoin(allocator, key, "layers");
        try allocs_to_free.append(layers_key);
        for (0..@intCast(config.decoder_layers)) |i| {
            layers[i] = try ResidualAttentionBlock.init(allocator, layers_key, i, config.d_model, config.decoder_attention_heads, true, weights_hash, stream);
        }
        var self = try allocator.create(Self);
        self.* = .{
            .embed_tokens = embed_tokens,
            .positional_embedding = mlx.arrayNew(),
            .layers = layers,
            .layer_norm = layer_norm,
            .stream = stream,
            .allocator = allocator,
            .allocs_to_free = allocs_to_free,
        };
        const pe_key = try allocJoin(allocator, key, "embed_positions.weight");
        try self.allocs_to_free.append(pe_key);
        try weights_hash.put(pe_key, &self.positional_embedding);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.embed_tokens.deinit();
        mlx.arrayFree(self.positional_embedding);
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.layer_norm.deinit();
        for (self.allocs_to_free.items) |allocation| self.allocator.free(allocation);
        self.allocs_to_free.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, xa: mlx.Array, mask: mlx.Array, kv_cache: *mlx.Cache, cross_kv_cache: *mlx.Cache) !void {
        const offset = kv_cache.offset;
        try self.embed_tokens.forward(result, x);
        var pos_embed_slice = mlx.arrayNew();
        defer mlx.arrayFree(pos_embed_slice);
        try mlx.slice(&pos_embed_slice, self.positional_embedding, &[_]c_int{ offset, 0 }, &[_]c_int{ offset + mlx.arrayDim(x, 1), mlx.arrayDim(self.positional_embedding, 1) }, &[_]c_int{ 1, 1 }, self.stream);
        try mlx.add(result, result.*, pos_embed_slice, self.stream);
        for (self.layers, 0..) |*layer, i| {
            _ = try layer.forward(result, result.*, xa, mask, &kv_cache.layers[i], &cross_kv_cache.layers[i]);
        }
        try self.layer_norm.forward(result, result.*);
        try self.embed_tokens.asLinear(result, result.*);
    }
};

pub const Whisper = struct {
    const Self = @This();
    encoder: *AudioEncoder,
    decoder: *TextDecoder,
    stream: mlx.Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: *const WhisperConfig, weights_hash: *std.StringHashMap(*mlx.Array), stream: mlx.Stream) !Self {
        const encoder = try AudioEncoder.init(allocator, "model", "encoder", config, weights_hash, stream);
        const decoder = try TextDecoder.init(allocator, "model", "decoder", config, weights_hash, stream);
        return .{
            .encoder = encoder,
            .decoder = decoder,
            .stream = stream,
            .allocator = allocator,
        };
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
        const stream = mlx.defaultGpuStreamNew();
        const path_config = try std.fmt.bufPrintZ(&buf, "{s}/config.json", .{model_path});
        const config = try loadJson(WhisperConfig, allocator, path_config, true);
        defer config.deinit();
        var weights_hash = std.StringHashMap(*mlx.Array).init(allocator);
        defer weights_hash.deinit();
        var model = try Whisper.init(allocator, &config.value, &weights_hash, stream);
        errdefer model.deinit();
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
        try safetensors.unload(&weights_hash);
        return .{
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
        const start_time = std.time.milliTimestamp();
        while (i + 3000 < mel_len) {
            try mlx.slice(&mel, mel_all, &[_]c_int{ 0, i, 0 }, &[_]c_int{ 1, i + 3000, 128 }, &[_]c_int{ 1, 1, 1 }, self.stream);
            const piece = try self.step(mel);
            defer self.allocator.free(piece);
            const arg_hop = std.mem.indexOfMax(u32, piece);
            const hop = (piece[arg_hop] - 50365) * 2;
            try new_tok.appendSlice(piece[0..arg_hop]);
            i += if (hop > 0) @intCast(hop) else 3000;
        }
        const elapsed: f16 = @floatFromInt(std.time.milliTimestamp() - start_time);
        const ntok = new_tok.items.len;
        const tps = @as(f16, @floatFromInt(ntok)) / (elapsed / 1000.0);
        std.debug.print("\n{d:.2} tokens-per-second ({d} tokens in {d:.2} ms)\n", .{ tps, ntok, elapsed });
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
        var output_tokens = try self.allocator.alloc(u32, 446);
        errdefer self.allocator.free(output_tokens);
        var toks = try mlx.arrayNewData(&[_]u32{ 50258, 50360, 50365 }, .{ 1, 3 }, mlx.DTYPE.UINT32);
        defer mlx.arrayFree(toks);
        var logits = mlx.arrayNew();
        defer mlx.arrayFree(logits);
        var mask = mlx.arrayNew();
        defer mlx.arrayFree(mask);
        var i: usize = 0;
        while (i < 446) : (i += 1) {
            try mlx.createCausalMask(&mask, mlx.arrayDim(toks, 1), kv_cache.offset, mlx.DTYPE.FLOAT16, self.stream);
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
            std.debug.print("Generated at {d}: {d}\n", .{ i, output_tokens[i] });
        }
        if (i < 446) {
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
    try mlx.asStrided(&audio, audio, &[_]c_int{ @divTrunc(mlx.arrayDim(audio, 0) - 240, 160), 400 }, &[_]i64{ 160, 1 }, 0, stream);
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

fn createSinusoids(result: *mlx.Array, length: c_int, channels: c_int, stream: mlx.Stream) !void { // : same as loaded weight
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
    try mlx.astype(result, result.*, mlx.DTYPE.FLOAT16, stream);
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
