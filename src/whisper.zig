//! whisper.zig - Whisper-Turbo-Large-v3
//!
//! Copyright 2025 Joe

const std = @import("std");
const mlx = @import("mlx.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const loadJson = @import("utils.zig").loadJson;
const loadAudio = @import("utils.zig").loadAudio;

const formatRange = @import("utils.zig").formatRange;
const formatRangeFloat = @import("utils.zig").formatRangeFloat;

pub const MultiHeadAttention = struct {
    const Self = @This();
    base: mlx.Module,
    n_heads: c_int,
    head_dim: c_int,
    scale: mlx.Array,
    q_proj: *mlx.Linear,
    k_proj: *mlx.Linear,
    v_proj: *mlx.Linear,
    out_proj: *mlx.Linear,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, d_model: c_int, n_heads: c_int, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .n_heads = n_heads,
            .head_dim = @divExact(d_model, n_heads),
            .scale = mlx.arrayNewFloat(1.0 / @sqrt(@as(f32, @floatFromInt(@divExact(d_model, n_heads))))),
            .q_proj = undefined,
            .k_proj = undefined,
            .v_proj = undefined,
            .out_proj = undefined,
        };
        const q_key = try self.base.allocJoin(key, "q_proj");
        self.q_proj = try mlx.Linear.init(mlx_config, q_key, true, null, weights_hash);
        const k_key = try self.base.allocJoin(key, "k_proj");
        self.k_proj = try mlx.Linear.init(mlx_config, k_key, false, null, weights_hash);
        const v_key = try self.base.allocJoin(key, "v_proj");
        self.v_proj = try mlx.Linear.init(mlx_config, v_key, true, null, weights_hash);
        const o_key = try self.base.allocJoin(key, "out_proj");
        self.out_proj = try mlx.Linear.init(mlx_config, o_key, true, null, weights_hash);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.out_proj.deinit();
        mlx.arrayFree(self.scale);
        self.base.deinit();
        self.base.allocator.destroy(self);
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
                try cache.update(&k, &v, null, self.base.stream);
            }
        }
        try mlx.rEshap(&q, q, "b l (h d) -> b h l d", .{ .h = self.n_heads, .d = self.head_dim }, self.base.stream);
        try mlx.rEshap(&k, k, "b l (h d) -> b h l d", .{ .h = self.n_heads, .d = self.head_dim }, self.base.stream);
        try mlx.rEshap(&v, v, "b l (h d) -> b h l d", .{ .h = self.n_heads, .d = self.head_dim }, self.base.stream);
        try mlx.multiply(&q, q, self.scale, self.base.stream);
        try mlx.einsum(&w, .{ q, k }, "b h l d, b h k d -> b h l k", self.base.stream);
        if (mask) |attention_mask| try mlx.add(&w, w, attention_mask, self.base.stream);
        try mlx.softmax(&w, w, &.{3}, true, self.base.stream);
        try mlx.einsum(result, .{ w, v }, "b h l k, b h k d -> b h l d", self.base.stream);
        try mlx.rEshap(result, result.*, "b h l d -> b l (h d)", .{}, self.base.stream);
        try self.out_proj.forward(result, result.*);
    }
};

pub const ResidualAttentionBlock = struct {
    const Self = @This();
    base: mlx.Module,
    self_attn: *MultiHeadAttention,
    self_attn_layer_norm: *mlx.LayerNorm,
    cross_attn: ?*MultiHeadAttention,
    cross_attn_layer_norm: ?*mlx.LayerNorm,
    fc1: *mlx.Linear,
    fc2: *mlx.Linear,
    final_layer_norm: *mlx.LayerNorm,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, d_model: c_int, n_head: c_int, cross_attention: bool, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .self_attn = undefined,
            .self_attn_layer_norm = undefined,
            .cross_attn = null,
            .cross_attn_layer_norm = null,
            .fc1 = undefined,
            .fc2 = undefined,
            .final_layer_norm = undefined,
        };
        const sa_key = try self.base.allocJoin(key, "self_attn");
        self.self_attn = try MultiHeadAttention.init(mlx_config, sa_key, d_model, n_head, weights_hash);
        const saln_key = try self.base.allocJoin(key, "self_attn_layer_norm");
        self.self_attn_layer_norm = try mlx.LayerNorm.init(mlx_config, saln_key, 1e-5, weights_hash);
        if (cross_attention) {
            const ca_key = try self.base.allocJoin(key, "encoder_attn");
            self.cross_attn = try MultiHeadAttention.init(mlx_config, ca_key, d_model, n_head, weights_hash);
            const caln_key = try self.base.allocJoin(key, "encoder_attn_layer_norm");
            self.cross_attn_layer_norm = try mlx.LayerNorm.init(mlx_config, caln_key, 1e-5, weights_hash);
        }
        const fc1_key = try self.base.allocJoin(key, "fc1");
        self.fc1 = try mlx.Linear.init(mlx_config, fc1_key, true, null, weights_hash);
        const fc2_key = try self.base.allocJoin(key, "fc2");
        self.fc2 = try mlx.Linear.init(mlx_config, fc2_key, true, null, weights_hash);
        const fln_key = try self.base.allocJoin(key, "final_layer_norm");
        self.final_layer_norm = try mlx.LayerNorm.init(mlx_config, fln_key, 1e-5, weights_hash);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.self_attn.deinit();
        self.self_attn_layer_norm.deinit();
        if (self.cross_attn) |cross_attn| {
            cross_attn.deinit();
        }
        if (self.cross_attn_layer_norm) |cross_attn_layer_norm| {
            cross_attn_layer_norm.deinit();
        }
        self.fc1.deinit();
        self.fc2.deinit();
        self.final_layer_norm.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
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
        try mlx.add(&attn, x, attn, self.base.stream);
        if (self.cross_attn != null and self.cross_attn_layer_norm != null) {
            if (xa) |encoder_out| {
                try self.cross_attn_layer_norm.?.forward(&xatn, attn);
                try self.cross_attn.?.forward(&xatn, xatn, encoder_out, null, cross_kv_cache);
                try mlx.add(&attn, xatn, attn, self.base.stream);
            }
        }
        try self.final_layer_norm.forward(&xatn, attn);
        try self.fc1.forward(&xatn, xatn);
        try mlx.gelu(&xatn, xatn, self.base.stream);
        try self.fc2.forward(&xatn, xatn);
        try mlx.add(result, xatn, attn, self.base.stream);
    }
};

pub const AudioEncoder = struct {
    const Self = @This();
    base: mlx.Module,
    conv1: *mlx.Conv1d,
    conv2: *mlx.Conv1d,
    positional_embedding: mlx.Array,
    layers: []*ResidualAttentionBlock,
    layer_norm: *mlx.LayerNorm,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, model_config: *const WhisperConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .conv1 = undefined,
            .conv2 = undefined,
            .positional_embedding = mlx.arrayNew(),
            .layers = undefined,
            .layer_norm = undefined,
        };
        // Option 1:
        try createSinusoids(&self.positional_embedding, model_config.max_source_positions, model_config.d_model, mlx_config.stream);
        try mlx.astype(&self.positional_embedding, self.positional_embedding, mlx.FLOAT16, mlx_config.stream);
        // Option 2:
        // const pe_key = try self.base.allocJoin(key, "embed_positions.weight");
        // try weights_hash.put(pe_key, &self.positional_embedding);
        const conv1_key = try self.base.allocJoin(key, "conv1");
        self.conv1 = try mlx.Conv1d.init(mlx_config, conv1_key, 1, 1, 1, 1, true, weights_hash);
        const conv2_key = try self.base.allocJoin(key, "conv2");
        self.conv2 = try mlx.Conv1d.init(mlx_config, conv2_key, 2, 1, 1, 1, true, weights_hash);
        const ln_key = try self.base.allocJoin(key, "layer_norm");
        self.layer_norm = try mlx.LayerNorm.init(mlx_config, ln_key, 1e-5, weights_hash);
        self.layers = try mlx_config.allocator.alloc(*ResidualAttentionBlock, @intCast(model_config.encoder_layers));
        const layers_key = try self.base.allocJoin(key, "layers");
        for (0..@intCast(model_config.encoder_layers)) |i| {
            const i_key = try self.base.allocJoin(layers_key, i);
            self.layers[i] = try ResidualAttentionBlock.init(mlx_config, i_key, model_config.d_model, model_config.encoder_attention_heads, false, weights_hash);
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.conv1.deinit();
        self.conv2.deinit();
        mlx.arrayFree(self.positional_embedding);
        for (self.layers) |layer| {
            layer.deinit();
        }
        self.base.allocator.free(self.layers);
        self.layer_norm.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        try self.conv1.forward(result, x);
        try mlx.gelu(result, result.*, self.base.stream);
        try self.conv2.forward(result, result.*);
        try mlx.gelu(result, result.*, self.base.stream);
        try mlx.add(result, result.*, self.positional_embedding, self.base.stream);
        for (self.layers) |layer| {
            try layer.forward(result, result.*, null, null, null, null);
        }
        try self.layer_norm.forward(result, result.*);
    }
};

pub const TextDecoder = struct {
    const Self = @This();
    base: mlx.Module,
    embed_tokens: *mlx.Embedding,
    positional_embedding: mlx.Array,
    layers: []*ResidualAttentionBlock,
    layer_norm: *mlx.LayerNorm,

    pub fn init(mlx_config: mlx.MLXConfig, key: []const u8, model_config: *const WhisperConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .embed_tokens = undefined,
            .positional_embedding = mlx.arrayNew(),
            .layers = undefined,
            .layer_norm = undefined,
        };
        const embed_key = try self.base.allocJoin(key, "embed_tokens");
        self.embed_tokens = try mlx.Embedding.init(mlx_config, embed_key, null, weights_hash);
        const pe_key = try self.base.allocJoin(key, "embed_positions.weight");
        try weights_hash.put(pe_key, &self.positional_embedding);
        const ln_key = try self.base.allocJoin(key, "layer_norm");
        self.layer_norm = try mlx.LayerNorm.init(mlx_config, ln_key, 1e-5, weights_hash);
        self.layers = try mlx_config.allocator.alloc(*ResidualAttentionBlock, @intCast(model_config.decoder_layers));
        const layers_key = try self.base.allocJoin(key, "layers");
        for (0..@intCast(model_config.decoder_layers)) |i| {
            const i_key = try self.base.allocJoin(layers_key, i);
            self.layers[i] = try ResidualAttentionBlock.init(mlx_config, i_key, model_config.d_model, model_config.decoder_attention_heads, true, weights_hash);
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.embed_tokens.deinit();
        mlx.arrayFree(self.positional_embedding);
        for (self.layers) |layer| {
            layer.deinit();
        }
        self.base.allocator.free(self.layers);
        self.layer_norm.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, xa: mlx.Array, mask: mlx.Array, kv_cache: *mlx.Cache, cross_kv_cache: *mlx.Cache) !void {
        const offset = kv_cache.offset;
        try self.embed_tokens.forward(result, x);
        var pos_embed_slice = mlx.arrayNew();
        defer mlx.arrayFree(pos_embed_slice);
        try mlx.slice(&pos_embed_slice, self.positional_embedding, &[_]c_int{ offset, 0 }, &[_]c_int{ offset + mlx.arrayDim(x, 1), mlx.arrayDim(self.positional_embedding, 1) }, &[_]c_int{ 1, 1 }, self.base.stream);
        try mlx.add(result, result.*, pos_embed_slice, self.base.stream);
        for (self.layers, 0..) |layer, i| {
            try layer.forward(result, result.*, xa, mask, &kv_cache.layers[i], &cross_kv_cache.layers[i]);
        }
        try self.layer_norm.forward(result, result.*);
        try self.embed_tokens.asLinear(result, result.*);
    }
};

pub const Whisper = struct {
    const Self = @This();
    base: mlx.Module,
    encoder: *AudioEncoder,
    decoder: *TextDecoder,

    pub fn init(mlx_config: mlx.MLXConfig, model_config: *const WhisperConfig, weights_hash: *std.StringHashMap(*mlx.Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .encoder = undefined,
            .decoder = undefined,
        };
        self.encoder = try AudioEncoder.init(mlx_config, "model.encoder", model_config, weights_hash);
        self.decoder = try TextDecoder.init(mlx_config, "model.decoder", model_config, weights_hash);
        return self;
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
        self.base.deinit();
        self.base.allocator.destroy(self);
    }
};

pub const Transcriber = struct {
    const Self = @This();
    mlx_config: mlx.MLXConfig,
    model: *Whisper,
    tokenizer: Tokenizer,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        var buf: [1024]u8 = undefined;
        var mlx_config = try mlx.MLXConfig.init(allocator);
        errdefer mlx_config.deinit();
        const path_config = try std.fmt.bufPrintZ(&buf, "{s}/config.json", .{model_path});
        const model_config = try loadJson(WhisperConfig, allocator, path_config, true);
        defer model_config.deinit();
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
        var weights_hash = std.StringHashMap(*mlx.Array).init(allocator);
        defer weights_hash.deinit();
        var model = try Whisper.init(mlx_config, &model_config.value, &weights_hash);
        errdefer model.deinit();
        const path_weight = try std.fmt.bufPrintZ(&buf, "{s}/model.safetensors", .{model_path});
        var safetensors = try mlx.Safetensors.load(path_weight, mlx_config.stream);
        defer safetensors.deinit();
        try safetensors.unload(&weights_hash);
        return .{
            .mlx_config = mlx_config,
            .model = model,
            .tokenizer = tokenizer,
        };
    }

    pub fn transcribe(self: *Self, audio_path: []const u8) ![]const u8 {
        const audio = try loadAudio(self.mlx_config.allocator, audio_path);
        defer self.mlx_config.allocator.free(audio);
        var mel_all = mlx.arrayNew();
        defer mlx.arrayFree(mel_all);
        var mel = mlx.arrayNew();
        defer mlx.arrayFree(mel);
        try getMel(&mel_all, audio, self.mlx_config.stream);
        var new_tok = std.ArrayList(u32).init(self.mlx_config.allocator);
        defer new_tok.deinit();
        var i: c_int = 0;
        const mel_len = mlx.arrayDim(mel_all, 1);
        const start_time = std.time.milliTimestamp();
        while (i + 3000 < mel_len) {
            try mlx.slice(&mel, mel_all, &[_]c_int{ 0, i, 0 }, &[_]c_int{ 1, i + 3000, 128 }, &[_]c_int{ 1, 1, 1 }, self.mlx_config.stream);
            const piece = try self.step(mel);
            defer self.mlx_config.allocator.free(piece);
            const arg_hop = std.mem.indexOfMax(u32, piece);
            const hop = (piece[arg_hop] - 50365) * 2;
            try new_tok.appendSlice(piece[0..arg_hop]);
            i += if (hop > 0) @intCast(hop) else 3000;
        }
        const elapsed: f16 = @floatFromInt(std.time.milliTimestamp() - start_time);
        const ntok = new_tok.items.len;
        const tps = @as(f16, @floatFromInt(ntok)) / (elapsed / 1000.0);
        std.debug.print("\n{d:.2} tokens-per-second ({d} tokens in {d:.2} ms)\n", .{ tps, ntok, elapsed });
        var filtered_tokens = std.ArrayList(u32).init(self.mlx_config.allocator);
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
        try self.model.encode(&enc, raw);
        var kv_cache = try mlx.Cache.init(self.mlx_config.allocator, self.model.decoder.layers.len, 1);
        defer kv_cache.deinit();
        var cross_kv_cache = try mlx.Cache.init(self.mlx_config.allocator, self.model.decoder.layers.len, 1);
        defer cross_kv_cache.deinit();
        var output_tokens = try self.mlx_config.allocator.alloc(u32, 446);
        errdefer self.mlx_config.allocator.free(output_tokens);
        var toks = try mlx.arrayNewData(&[_]u32{ 50258, 50360, 50365 }, .{ 1, 3 }, mlx.UINT32);
        defer mlx.arrayFree(toks);
        var logits = mlx.arrayNew();
        defer mlx.arrayFree(logits);
        var mask = mlx.arrayNew();
        defer mlx.arrayFree(mask);
        var i: usize = 0;
        while (i < 446) : (i += 1) {
            try mlx.createCausalMask(&mask, mlx.arrayDim(toks, 1), kv_cache.offset, mlx.FLOAT16, self.mlx_config.stream);
            try self.model.decode(&logits, toks, enc, mask, &kv_cache, &cross_kv_cache);
            try mlx.take(&logits, logits, mlx.int(-1), 1, self.mlx_config.stream);
            try mlx.argmax(&logits, logits, 1, false, self.mlx_config.stream);
            try mlx.item(&output_tokens[i], logits);
            kv_cache.offset += mlx.arrayDim(toks, 1);
            try mlx.arraySetData(&toks, &output_tokens[i], .{ 1, 1 }, mlx.UINT32);
            if (output_tokens[i] == 50257) {
                i += 1;
                break;
            }
            std.debug.print("Generated at {d}: {d}\n", .{ i, output_tokens[i] });
        }
        if (i < 446) {
            output_tokens = try self.mlx_config.allocator.realloc(output_tokens, i);
        }
        return output_tokens;
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        self.tokenizer.deinit();
        self.mlx_config.deinit();
        self.mlx_config.allocator.destroy(self);
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
    var audio = try mlx.arrayNewData(audio_raw.ptr, .{audio_raw.len}, mlx.FLOAT32);
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
    try mlx.maxAll(&threshold, audio, false, stream);
    try mlx.subtract(&threshold, threshold, mlx.float(8.0), stream);
    try mlx.maximum(&audio, audio, threshold, stream);
    try mlx.add(&audio, audio, mlx.float(4.0), stream);
    try mlx.divide(&audio, audio, mlx.float(4.0), stream);
    try mlx.expand_dims(&audio, audio, &[_]c_int{0}, stream);
    try mlx.astype(result, audio, mlx.FLOAT16, stream);
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
    try mlx.arange(&inv_timescales, 0, @floatFromInt(@divExact(channels, 2)), 1, mlx.FLOAT32, stream);
    try mlx.multiply(&inv_timescales, mlx.float(-log_timescale_increment), inv_timescales, stream);
    try mlx.exp(&inv_timescales, inv_timescales, stream);
    try mlx.arange(&scaled_time, 0, @floatFromInt(length), 1, mlx.FLOAT32, stream);
    try mlx.einsum(&scaled_time, .{ scaled_time, inv_timescales }, "i,j->ij", stream);
    try mlx.sin(&sin_val, scaled_time, stream);
    try mlx.cos(&cos_val, scaled_time, stream);
    try mlx.concatenate(result, .{ sin_val, cos_val }, 1, stream);
    try mlx.astype(result, result.*, mlx.FLOAT16, stream);
}

pub const WhisperConfig = struct {
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
