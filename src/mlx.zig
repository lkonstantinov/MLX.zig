//! mlx.zig - Improved MLX Bindings
//!
//! Copyright 2025 Joe

const std = @import("std");
const allocJoin = @import("utils.zig").allocJoin;
const comptimeJoin = @import("utils.zig").comptimeJoin;
pub const C = @cImport({
    @cInclude("mlx/c/mlx.h");
    @cInclude("stdio.h");
});

/// ============================================================================
/// Types & Constants
/// ============================================================================
pub const BOOL = C.MLX_BOOL;
pub const INT32 = C.MLX_INT32;
pub const UINT32 = C.MLX_UINT32;
pub const FLOAT16 = C.MLX_FLOAT16;
pub const FLOAT32 = C.MLX_FLOAT32;
pub const FLOAT64 = C.MLX_FLOAT64;
pub const BFLOAT16 = C.MLX_BFLOAT16;
pub const Array = C.mlx_array;
pub const String = C.mlx_string;
pub const Stream = C.mlx_stream;
pub const VectorArray = C.mlx_vector_array;
pub const MapStrArr = C.mlx_map_string_to_array;
pub const MapStrStr = C.mlx_map_string_to_string;
pub const OptionalFloat = C.mlx_optional_float;

/// ============================================================================
/// Error Handling
/// ============================================================================
pub const MLXError = error{
    OperationFailed,
    InvalidArray,
    DeviceError,
    FileNotFound,
    LoadWeightsFailed,
    KeyNotFoundInWeightsHash,
    OutOfMemory,
    NoSpaceLeft,
};

pub fn mlxOp(result: c_int) MLXError!void {
    if (result != 0) return MLXError.OperationFailed;
}

pub fn mlxOpWithLog(result: c_int, comptime func_name: []const u8) MLXError!void {
    if (result != 0) {
        std.log.err("MLX operation '{s}' failed with code {d}", .{ func_name, result });
        return MLXError.OperationFailed;
    }
}

/// ============================================================================
/// Scalar Operations
/// ============================================================================
pub const add = defineBinaryOp("mlx_add");
pub const subtract = defineBinaryOp("mlx_subtract");
pub const multiply = defineBinaryOp("mlx_multiply");
pub const divide = defineBinaryOp("mlx_divide");
pub const power = defineBinaryOp("mlx_power");
pub const greater = defineBinaryOp("mlx_greater");
pub const greaterEqual = defineBinaryOp("mlx_greater_equal");
pub const less = defineBinaryOp("mlx_less");
pub const lessEqual = defineBinaryOp("mlx_less_equal");
pub const logicalOr = defineBinaryOp("mlx_logical_or");
pub const logicalAnd = defineBinaryOp("mlx_logical_and");
pub const matmul = defineBinaryOp("mlx_matmul");
pub const minimum = defineBinaryOp("mlx_minimum");
pub const maximum = defineBinaryOp("mlx_maximum");
pub const logicalNot = defineUnaryOp("mlx_logical_not");
pub const isnan = defineUnaryOp("mlx_isnan");
pub const sigmoid = defineUnaryOp("mlx_sigmoid");
pub const sin = defineUnaryOp("mlx_sin");
pub const cos = defineUnaryOp("mlx_cos");
pub const exp = defineUnaryOp("mlx_exp");
pub const abs = defineUnaryOp("mlx_abs");
pub const square = defineUnaryOp("mlx_square");
pub const log = defineUnaryOp("mlx_log");
pub const log10 = defineUnaryOp("mlx_log10");

pub fn item(dest: anytype, arr: Array) MLXError!void {
    const T = @TypeOf(dest);
    const info = @typeInfo(T);
    if (info != .Pointer) @compileError("Expected pointer, got " ++ @typeName(T));
    const child = info.Pointer.child;
    const c_func_name = switch (child) {
        u32, c_uint => "mlx_array_item_uint32",
        i32, c_int => "mlx_array_item_int32",
        f32 => "mlx_array_item_float32",
        f64 => "mlx_array_item_float64",
        bool => "mlx_array_item_bool",
        u8 => "mlx_array_item_uint8",
        u16 => "mlx_array_item_uint16",
        u64 => "mlx_array_item_uint64",
        i8 => "mlx_array_item_int8",
        i16 => "mlx_array_item_int16",
        i64 => "mlx_array_item_int64",
        else => @compileError("Unsupported item type: " ++ @typeName(child)),
    };
    try mlxOpWithLog(@field(C, c_func_name)(dest, arr), c_func_name);
}

pub fn astype(result: *Array, x: anytype, dtype: C.mlx_dtype, stream: Stream) MLXError!void {
    const x_conv = arrayConverter(x);
    defer x_conv.deinit();
    try mlxOp(C.mlx_astype(result, x_conv.arr, dtype, stream));
}

/// ============================================================================
/// Array Operations
/// ============================================================================
pub const maxAll = defineReduceAllOp("mlx_max_all");
pub const minAll = defineReduceAllOp("mlx_min_all");
pub const max = defineReduceAxesOp("mlx_max");
pub const min = defineReduceAxesOp("mlx_min");
pub const rfft = defineFFTOp("mlx_fft_rfft");
pub const irfft = defineFFTOp("mlx_fft_irfft");
pub const fft = defineFFTOp("mlx_fft_fft");
pub const ifft = defineFFTOp("mlx_fft_ifft");
pub const fft2 = defineFFTNOp("mlx_fft_fft2");
pub const ifft2 = defineFFTNOp("mlx_fft_ifft2");
pub const rfft2 = defineFFTNOp("mlx_fft_rfft2");
pub const irfft2 = defineFFTNOp("mlx_fft_irfft2");
pub const fftn = defineFFTNOp("mlx_fft_fftn");
pub const ifftn = defineFFTNOp("mlx_fft_ifftn");
pub const rfftn = defineFFTNOp("mlx_fft_rfftn");
pub const irfftn = defineFFTNOp("mlx_fft_irfftn");
pub const arrayNew = C.mlx_array_new;
pub const arrayNewFloat = C.mlx_array_new_float;
pub const arrayDim = C.mlx_array_dim;
pub const arrayShape = C.mlx_array_shape;

pub fn arrayNewData(data: *const anyopaque, shape_arg: anytype, dtype: C.mlx_dtype) MLXError!Array {
    var shape: [32]c_int = undefined;
    var len: usize = 0;
    const T = @TypeOf(shape_arg);
    const fields = @typeInfo(T).Struct.fields;
    inline for (fields, 0..) |field, idx| {
        shape[idx] = @intCast(@field(shape_arg, field.name));
        len = idx + 1;
    }
    const arr = C.mlx_array_new_data(data, &shape, @intCast(len), dtype);
    if (arr.ctx == null) return MLXError.InvalidArray;
    return arr;
}

pub fn arraySetData(arr: *Array, data: *const anyopaque, shape_arg: anytype, dtype: C.mlx_dtype) MLXError!void {
    var shape: [32]c_int = undefined;
    var len: usize = 0;
    const T = @TypeOf(shape_arg);
    const fields = @typeInfo(T).Struct.fields;
    inline for (fields, 0..) |field, idx| {
        shape[idx] = @intCast(@field(shape_arg, field.name));
        len = idx + 1;
    }
    try mlxOpWithLog(C.mlx_array_set_data(arr, data, &shape, @intCast(len), dtype), "mlx_array_set_data");
}

pub fn where(result: *Array, cond: Array, x: anytype, y: anytype, stream: Stream) MLXError!void {
    const x_conv = arrayConverter(x);
    const y_conv = arrayConverter(y);
    defer {
        x_conv.deinit();
        y_conv.deinit();
    }
    try mlxOpWithLog(C.mlx_where(result, cond, x_conv.arr, y_conv.arr, stream), "where");
}

pub fn take(result: *Array, x: Array, indices: anytype, axis: c_int, stream: Stream) MLXError!void {
    const indices_conv = arrayConverter(indices);
    defer indices_conv.deinit();
    try mlxOpWithLog(C.mlx_take(result, x, indices_conv.arr, axis, stream), "mlx_take");
}

pub fn pad(result: *Array, x: Array, axes: []const c_int, low_pad: []const c_int, high_pad: []const c_int, pad_value: anytype, pad_mode: [*:0]const u8, stream: Stream) MLXError!void {
    const pad_val_conv = arrayConverter(pad_value);
    defer pad_val_conv.deinit();
    try mlxOp(C.mlx_pad(result, x, axes.ptr, axes.len, low_pad.ptr, low_pad.len, high_pad.ptr, high_pad.len, pad_val_conv.arr, pad_mode, stream));
}

pub fn slice(result: *Array, x: Array, start: []const c_int, stop: []const c_int, strides: []const c_int, stream: Stream) MLXError!void {
    try mlxOp(C.mlx_slice(result, x, start.ptr, start.len, stop.ptr, stop.len, strides.ptr, strides.len, stream));
}

pub fn asStrided(result: *Array, x: Array, shape: []const c_int, strides: []const i64, offset: usize, stream: Stream) MLXError!void {
    try mlxOp(C.mlx_as_strided(result, x, shape.ptr, shape.len, strides.ptr, strides.len, offset, stream));
}

pub fn expand_dims(result: *Array, x: Array, axes: []const c_int, stream: Stream) MLXError!void {
    try mlxOp(C.mlx_expand_dims(result, x, axes.ptr, axes.len, stream));
}

pub fn reshape(result: *Array, x: Array, shape: []const c_int, stream: Stream) MLXError!void {
    try mlxOp(C.mlx_reshape(result, x, shape.ptr, shape.len, stream));
}

pub fn softmax(result: *Array, x: Array, axes: []const c_int, precise: bool, stream: Stream) MLXError!void {
    try mlxOp(C.mlx_softmax(result, x, axes.ptr, axes.len, precise, stream));
}

pub fn ones(result: *Array, shape: []const c_int, dtype: C.mlx_dtype, stream: Stream) MLXError!void {
    try mlxOp(C.mlx_ones(result, shape.ptr, shape.len, dtype, stream));
}

pub fn zeros(result: *Array, shape: []const c_int, dtype: C.mlx_dtype, stream: Stream) MLXError!void {
    try mlxOp(C.mlx_zeros(result, shape.ptr, shape.len, dtype, stream));
}

pub fn argmax(result: *Array, x: Array, axis: c_int, keepdims: bool, stream: Stream) MLXError!void {
    try mlxOp(C.mlx_argmax(result, x, axis, keepdims, stream));
}

pub fn tril(result: *Array, x: Array, offset: c_int, stream: Stream) MLXError!void {
    try mlxOp(C.mlx_tril(result, x, offset, stream));
}

pub fn linspace(result: *Array, start: f64, stop: f64, num: c_int, dtype: C.mlx_dtype, stream: Stream) MLXError!void {
    try mlxOp(C.mlx_linspace(result, start, stop, num, dtype, stream));
}

pub fn arange(result: *Array, start: f64, stop: f64, step: f64, dtype: C.mlx_dtype, stream: Stream) MLXError!void {
    try mlxOp(C.mlx_arange(result, start, stop, step, dtype, stream));
}

pub fn repeat(result: *Array, x: Array, repeats: c_int, axis: c_int, stream: Stream) MLXError!void {
    try mlxOp(C.mlx_repeat(result, x, repeats, axis, stream));
}

pub fn arraySet(arr: *Array, src: Array) MLXError!void {
    try mlxOp(C.mlx_array_set(arr, src));
}

pub fn arrayFree(arr: Array) void {
    _ = C.mlx_array_free(arr);
}

/// ============================================================================
/// Vector Operations
/// ============================================================================
pub fn einsum(result: *Array, arrays: anytype, pattern: [*:0]const u8, stream: Stream) MLXError!void {
    const fields = @typeInfo(@TypeOf(arrays)).Struct.fields;
    var array_data: [fields.len]Array = undefined;
    inline for (fields, 0..) |field, i| array_data[i] = @field(arrays, field.name);
    const operands = C.mlx_vector_array_new_data(&array_data[0], array_data.len);
    defer _ = C.mlx_vector_array_free(operands);
    try mlxOp(C.mlx_einsum(result, pattern, operands, stream));
}

pub fn concatenate(result: *Array, arrays: anytype, axis: c_int, stream: Stream) MLXError!void {
    const fields = @typeInfo(@TypeOf(arrays)).Struct.fields;
    var array_data: [fields.len]Array = undefined;
    inline for (fields, 0..) |field, i| array_data[i] = @field(arrays, field.name);
    const vector_arrays = C.mlx_vector_array_new_data(&array_data[0], array_data.len);
    defer _ = C.mlx_vector_array_free(vector_arrays);
    try mlxOp(C.mlx_concatenate(result, vector_arrays, axis, stream));
}

pub fn split(outputs: []const *Array, a: Array, indices: []const c_int, axis: c_int, stream: Stream) MLXError!void {
    var results = C.mlx_vector_array_new();
    defer _ = C.mlx_vector_array_free(results);
    try mlxOp(C.mlx_split(&results, a, indices.ptr, indices.len, axis, stream));
    for (outputs, 0..) |out_ptr, i| {
        try mlxOp(C.mlx_vector_array_get(out_ptr, results, i));
    }
}

pub fn splitEqualParts(outputs: []const *Array, a: Array, num_splits: c_int, axis: c_int, stream: Stream) MLXError!void {
    var results = C.mlx_vector_array_new();
    defer _ = C.mlx_vector_array_free(results);
    try mlxOp(C.mlx_split_equal_parts(&results, a, num_splits, axis, stream));
    for (outputs, 0..) |out_ptr, i| {
        try mlxOp(C.mlx_vector_array_get(out_ptr, results, i));
    }
}

/// ============================================================================
/// Fast Operations
/// ============================================================================
pub fn fastRope(result: *Array, x: Array, dims: c_int, traditional: bool, base: C.mlx_optional_float, scale: f32, offset: c_int, freqs: Array, s: Stream) MLXError!void {
    try mlxOp(C.mlx_fast_rope(result, x, dims, traditional, base, scale, offset, freqs, s));
}

pub fn fastRmsNorm(result: *Array, x: anytype, weight: anytype, eps: f32, stream: Stream) MLXError!void {
    const x_conv = arrayConverter(x);
    const weight_conv = arrayConverter(weight);
    defer {
        x_conv.deinit();
        weight_conv.deinit();
    }
    try mlxOp(C.mlx_fast_rms_norm(result, x_conv.arr, weight_conv.arr, eps, stream));
}

pub fn fastLayerNorm(result: *Array, x: anytype, weight: anytype, bias: anytype, eps: f32, stream: Stream) MLXError!void {
    const x_conv = arrayConverter(x);
    const weight_conv = arrayConverter(weight);
    const bias_conv = arrayConverter(bias);
    defer {
        x_conv.deinit();
        weight_conv.deinit();
        bias_conv.deinit();
    }
    try mlxOp(C.mlx_fast_layer_norm(result, x_conv.arr, weight_conv.arr, bias_conv.arr, eps, stream));
}

pub fn fastScaledDotProductAttention(result: *Array, queries: Array, keys: Array, values: Array, scale: f32, mask: ?Array, stream: Stream) MLXError!void {
    const memory_threshold = C.mlx_optional_int{ .has_value = false, .value = 0 };
    const mask_ptr = if (mask) |m| m else C.mlx_array_empty;
    try mlxOp(C.mlx_fast_scaled_dot_product_attention(result, queries, keys, values, scale, mask_ptr, memory_threshold, stream));
}

/// ============================================================================
/// Stream Operations
/// ============================================================================
pub fn streamFree(stream: Stream) void {
    _ = C.mlx_stream_free(stream);
}

pub const defaultCpuStreamNew = C.mlx_default_cpu_stream_new;
pub const defaultGpuStreamNew = C.mlx_default_gpu_stream_new;

/// ============================================================================
/// File Operations
/// ============================================================================
pub const Safetensors = struct {
    const Self = @This();
    const MAX_PATH_LEN = 1024;

    file: ?*C.FILE,
    weights: MapStrArr,
    stream: Stream,
    added_tensors: ?std.ArrayList(Self),
    allocator: ?std.mem.Allocator,

    pub fn load(path_safetensors: [:0]const u8, stream: Stream) MLXError!Self {
        const file = C.fopen(path_safetensors.ptr, "rb") orelse return MLXError.FileNotFound;
        var weights = C.mlx_map_string_to_array_new();
        errdefer {
            _ = C.mlx_map_string_to_array_free(weights);
            _ = C.fclose(file);
        }
        var meta = C.mlx_map_string_to_string_new();
        defer _ = C.mlx_map_string_to_string_free(meta);
        if (C.mlx_load_safetensors_file(&weights, &meta, file, stream) != 0) {
            _ = C.fclose(file);
            _ = C.mlx_map_string_to_array_free(weights);
            return MLXError.LoadWeightsFailed;
        }
        return Self{
            .file = file,
            .weights = weights,
            .stream = stream,
            .added_tensors = null,
            .allocator = null,
        };
    }

    pub fn unload(self: *Self, weights_hash: *std.StringHashMap(*Array)) MLXError!void {
        const mapsa_iter = C.mlx_map_string_to_array_iterator_new(self.weights);
        defer _ = C.mlx_map_string_to_array_iterator_free(mapsa_iter);
        var key: [*c]const u8 = undefined;
        var value = C.mlx_array_new();
        defer arrayFree(value);
        while (C.mlx_map_string_to_array_iterator_next(&key, &value, mapsa_iter) == 0) {
            const key_str = std.mem.span(key);
            if (weights_hash.get(key_str)) |weight_ptr| {
                try arraySet(weight_ptr, value);
                try mlxOp(C.mlx_array_eval(weight_ptr.*));
            } else {
                std.debug.print("\nKey not found in weights_hash: {s}\n", .{key_str});
                // return MLXError.KeyNotFoundInWeightsHash; // : sinusoids in whisper.zig can be either loaded/created
            }
        }
    }

    pub fn add(self: *Self, paths: []const [:0]const u8, allocator: std.mem.Allocator) MLXError!void {
        if (self.added_tensors == null) {
            self.added_tensors = std.ArrayList(Self).init(allocator);
            self.allocator = allocator;
        }
        for (paths) |path| {
            const tensor = try Self.load(path, self.stream);
            const add_iter = C.mlx_map_string_to_array_iterator_new(tensor.weights);
            defer _ = C.mlx_map_string_to_array_iterator_free(add_iter);
            var key: [*c]const u8 = undefined;
            var value = C.mlx_array_new();
            defer arrayFree(value);
            while (C.mlx_map_string_to_array_iterator_next(&key, &value, add_iter) == 0) {
                _ = C.mlx_map_string_to_array_insert(self.weights, key, value);
            }
            try self.added_tensors.?.append(tensor);
        }
    }

    pub fn deinit(self: *Self) void {
        if (self.file) |file| {
            _ = C.fclose(file);
            self.file = null;
        }
        if (self.added_tensors) |*tensors| {
            for (tensors.items) |*tensor| {
                var tensor_copy = tensor.*;
                tensor_copy.added_tensors = null;
                tensor_copy.deinit();
            }
            tensors.deinit();
        }
        _ = C.mlx_map_string_to_array_free(self.weights);
    }
};

pub fn loadArray(weight: *Array, name: []const u8, ext: ?[]const u8, weights_map: *const MapStrArr) MLXError!void {
    var buf: [1024]u8 = undefined;
    const key = if (ext) |e| try std.fmt.bufPrintZ(&buf, "{s}.{s}", .{ name, e }) else name;
    try mlxOp(C.mlx_map_string_to_array_get(weight, weights_map.*, key.ptr));
    try mlxOp(C.mlx_array_eval(weight.*));
}

/// ============================================================================
/// NN Operations
/// ============================================================================
pub const MLXConfig = struct {
    allocator: std.mem.Allocator,
    stream: Stream,
    device: enum { CPU, GPU } = .GPU,
    default_dtype: C.mlx_dtype = C.MLX_FLOAT32,

    pub fn init(allocator: std.mem.Allocator) !MLXConfig {
        const stream = if (true) // Placeholder for device detection logic
            C.mlx_default_gpu_stream_new()
        else
            C.mlx_default_cpu_stream_new();

        if (stream.ctx == null) return MLXError.DeviceError;

        return MLXConfig{
            .allocator = allocator,
            .stream = stream,
            .device = .GPU,
            .default_dtype = C.MLX_FLOAT32,
        };
    }

    pub fn deinit(self: *@This()) void {
        streamFree(self.stream);
    }
};

pub const QuantConfig = struct {
    group_size: c_int,
    bits: c_int,
};

pub const Module = struct {
    const Self = @This();
    allocator: std.mem.Allocator,
    stream: Stream,
    allocs_to_free: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, stream: Stream) Self {
        return .{
            .allocator = allocator,
            .stream = stream,
            .allocs_to_free = std.ArrayList([]const u8).init(allocator),
        };
    }

    pub fn allocDupe(self: *Self, key: []const u8) ![]const u8 {
        const owned_key = try self.allocator.dupe(u8, key);
        try self.allocs_to_free.append(owned_key);
        return owned_key;
    }

    pub fn allocJoin(self: *Self, parent: []const u8, name: anytype) ![]const u8 {
        const owned_key = if (@TypeOf(name) == @TypeOf(null) or
            (@typeInfo(@TypeOf(name)) == .Pointer and name.len == 0))
            try self.allocator.dupe(u8, parent)
        else if (@typeInfo(@TypeOf(name)) == .Int or @typeInfo(@TypeOf(name)) == .ComptimeInt)
            try std.fmt.allocPrint(self.allocator, "{s}.{d}", .{ parent, name })
        else
            try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ parent, name });
        try self.allocs_to_free.append(owned_key);
        return owned_key;
    }

    pub fn deinit(self: *Self) void {
        for (self.allocs_to_free.items) |key| self.allocator.free(key);
        self.allocs_to_free.deinit();
    }
};

pub const Weight = struct {
    const Self = @This();
    base: Module,
    weight: Array,
    is_quantized: bool,
    scales: ?Array,
    biases: ?Array,
    group_size: ?c_int,
    bits: ?c_int,

    pub fn init(mlx_config: MLXConfig, quant_config: ?QuantConfig, key: []const u8, weights_hash: *std.StringHashMap(*Array)) !*Self {
        const is_quantized = quant_config != null;
        var self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .weight = arrayNew(),
            .is_quantized = is_quantized,
            .scales = if (is_quantized) arrayNew() else null,
            .biases = if (is_quantized) arrayNew() else null,
            .group_size = if (is_quantized) quant_config.?.group_size else null,
            .bits = if (is_quantized) quant_config.?.bits else null,
        };
        const weight_key = try self.base.allocJoin(key, "weight");
        try weights_hash.put(weight_key, &self.weight);
        if (is_quantized) {
            const scales_key = try self.base.allocJoin(key, "scales");
            try weights_hash.put(scales_key, &self.scales.?);

            const biases_key = try self.base.allocJoin(key, "biases");
            try weights_hash.put(biases_key, &self.biases.?);
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        arrayFree(self.weight);
        if (self.is_quantized) {
            arrayFree(self.scales.?);
            arrayFree(self.biases.?);
        }
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *Array, x: Array) MLXError!void {
        if (self.is_quantized) {
            return mlxOp(C.mlx_quantized_matmul(result, x, self.weight, self.scales.?, self.biases.?, true, self.group_size.?, self.bits.?, self.base.stream));
        } else {
            return einsum(result, .{ x, self.weight }, "blh,dh->bld", self.base.stream);
        }
    }

    pub fn dequantize(self: *Self) MLXError!void {
        if (self.is_quantized) {
            var temp = arrayNew();
            defer arrayFree(temp);
            try mlxOp(C.mlx_dequantize(&temp, self.weight, self.scales.?, self.biases.?, self.group_size.?, self.bits.?, self.base.stream));
            try arraySet(&self.weight, temp);
            try mlxOp(C.mlx_array_eval(self.weight));
            arrayFree(self.scales.?);
            arrayFree(self.biases.?);
            self.scales = null;
            self.biases = null;
            self.is_quantized = false;
        }
    }
};

pub const Linear = struct {
    const Self = @This();
    base: Module,
    weight: *Weight,
    has_bias: bool,
    bias: ?Array,

    pub fn init(mlx_config: MLXConfig, key: []const u8, has_bias: bool, quant_config: ?QuantConfig, weights_hash: *std.StringHashMap(*Array)) !*Self {
        var self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .weight = try Weight.init(mlx_config, quant_config, key, weights_hash),
            .has_bias = has_bias,
            .bias = if (has_bias) arrayNew() else null,
        };
        if (has_bias) {
            const bias_key = try self.base.allocJoin(key, "bias");
            try weights_hash.put(bias_key, &self.bias.?);
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.weight.deinit();
        if (self.has_bias) {
            arrayFree(self.bias.?);
        }
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *Array, x: Array) MLXError!void {
        try self.weight.forward(result, x);
        if (self.has_bias) {
            try add(result, result.*, self.bias.?, self.base.stream);
        }
    }
};

pub const Embedding = struct {
    const Self = @This();
    base: Module,
    weight: *Weight,
    is_sanitized: bool = false,

    pub fn init(mlx_config: MLXConfig, key: []const u8, quant_config: ?QuantConfig, weights_hash: *std.StringHashMap(*Array)) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .weight = try Weight.init(mlx_config, quant_config, key, weights_hash),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.weight.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn sanitize(self: *Self) MLXError!void {
        if (!self.is_sanitized) {
            try self.weight.dequantize();
            try self.weight.dequantize();
            self.is_sanitized = true;
        }
    }

    pub fn forward(self: *Self, result: *Array, toks: Array) MLXError!void {
        try self.sanitize();
        try take(result, self.weight.weight, toks, 0, self.base.stream);
    }

    pub fn asLinear(self: *Self, result: *Array, x: Array) MLXError!void {
        try self.weight.forward(result, x);
    }
};

pub const RMSNorm = struct {
    const Self = @This();
    base: Module,
    eps: f32,
    weight: Array,

    pub fn init(mlx_config: MLXConfig, key: []const u8, eps: f32, weights_hash: *std.StringHashMap(*Array)) !*Self {
        var self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .weight = arrayNew(),
            .eps = eps,
        };
        const weight_key = try self.base.allocJoin(key, "weight");
        try weights_hash.put(weight_key, &self.weight);
        return self;
    }

    pub fn deinit(self: *Self) void {
        arrayFree(self.weight);
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *Array, x: Array) MLXError!void {
        try fastRmsNorm(result, x, self.weight, self.eps, self.base.stream);
    }
};

pub const LayerNorm = struct {
    const Self = @This();
    base: Module,
    eps: f32,
    weight: Array,
    bias: Array,

    pub fn init(mlx_config: MLXConfig, key: []const u8, eps: f32, weights_hash: *std.StringHashMap(*Array)) !*Self {
        var self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .eps = eps,
            .weight = arrayNew(),
            .bias = arrayNew(),
        };
        const weight_key = try self.base.allocJoin(key, "weight");
        try weights_hash.put(weight_key, &self.weight);
        const bias_key = try self.base.allocJoin(key, "bias");
        try weights_hash.put(bias_key, &self.bias);
        return self;
    }

    pub fn deinit(self: *Self) void {
        arrayFree(self.weight);
        arrayFree(self.bias);
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *Array, x: Array) MLXError!void {
        try fastLayerNorm(result, x, self.weight, self.bias, self.eps, self.base.stream);
    }
};

pub const Conv1d = struct {
    const Self = @This();
    base: Module,
    stride: c_int,
    padding: c_int,
    dilation: c_int,
    groups: c_int,
    has_bias: bool,
    weight: Array,
    is_sanitized: bool = false,
    bias: ?Array,

    pub fn init(mlx_config: MLXConfig, key: []const u8, stride: c_int, padding: c_int, dilation: c_int, groups: c_int, has_bias: bool, weights_hash: *std.StringHashMap(*Array)) !*Self {
        var self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .stride = stride,
            .padding = padding,
            .dilation = dilation,
            .groups = groups,
            .has_bias = has_bias,
            .weight = arrayNew(),
            .bias = if (has_bias) arrayNew() else null,
        };
        const weight_key = try self.base.allocJoin(key, "weight");
        try weights_hash.put(weight_key, &self.weight);
        if (has_bias) {
            const bias_key = try self.base.allocJoin(key, "bias");
            try weights_hash.put(bias_key, &self.bias.?);
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        arrayFree(self.weight);
        if (self.has_bias) {
            arrayFree(self.bias.?);
        }
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn sanitize(self: *Self) !void {
        if (!self.is_sanitized) {
            try mlxOp(C.mlx_swapaxes(&self.weight, self.weight, 1, 2, self.base.stream));
            self.is_sanitized = true;
        }
    }

    pub fn forward(self: *Self, result: *Array, x: Array) !void {
        try self.sanitize();
        try mlxOp(C.mlx_conv1d(result, x, self.weight, self.stride, self.padding, self.dilation, self.groups, self.base.stream));
        if (self.has_bias) {
            try add(result, result.*, self.bias.?, self.base.stream);
        }
    }
};

pub fn gelu(result: *Array, x: Array, stream: Stream) MLXError!void {
    var tmp = arrayNew();
    defer arrayFree(tmp);
    try divide(&tmp, x, float(@sqrt(2.0)), stream);
    try mlxOp(C.mlx_erf(&tmp, tmp, stream));
    try add(&tmp, tmp, float(1.0), stream);
    try multiply(&tmp, tmp, float(0.5), stream);
    try multiply(result, x, tmp, stream);
}

pub fn silu(result: *Array, x: Array, stream: Stream) MLXError!void {
    var tmp = arrayNew();
    defer arrayFree(tmp);
    try sigmoid(&tmp, x, stream);
    try multiply(result, tmp, x, stream);
}

/// ============================================================================
/// KV Cache
/// ============================================================================
pub const KVCache = struct {
    const Self = @This();

    k: Array,
    v: Array,
    axis: c_int,
    is_empty: bool = true,

    pub fn init(axis: c_int) Self {
        return Self{ .k = C.mlx_array_new(), .v = C.mlx_array_new(), .axis = axis };
    }

    fn sliceCache(self: *Self, offset: c_int, stream: Stream) MLXError!void {
        if (offset >= arrayDim(self.k, self.axis)) return;
        const ndim = C.mlx_array_ndim(self.k);
        const start = [_]c_int{0} ** 4;
        const strides = [_]c_int{1} ** 4;
        var stop = [_]c_int{0} ** 4;
        for (0..ndim) |idx| stop[idx] = arrayDim(self.k, @intCast(idx));
        stop[@intCast(self.axis)] = offset;
        try mlxOp(C.mlx_slice(&self.k, self.k, &start, ndim, &stop, ndim, &strides, ndim, stream));
        try mlxOp(C.mlx_slice(&self.k, self.v, &start, ndim, &stop, ndim, &strides, ndim, stream));
        std.debug.print("Cache offset set to {d}\n", .{offset});
    }

    pub fn update(self: *Self, k: *Array, v: *Array, offset: ?c_int, stream: Stream) MLXError!void {
        const offset_ = if (offset) |o| o else if (self.is_empty) 0 else arrayDim(self.k, self.axis);
        if (offset_ > 0) {
            try self.sliceCache(offset_, stream);
            var k_concat = [_]Array{ self.k, k.* };
            const k_vec = C.mlx_vector_array_new_data(&k_concat[0], 2);
            defer _ = C.mlx_vector_array_free(k_vec);
            try mlxOp(C.mlx_concatenate(k, k_vec, self.axis, stream));
            var v_concat = [_]Array{ self.v, v.* };
            const v_vec = C.mlx_vector_array_new_data(&v_concat[0], 2);
            defer _ = C.mlx_vector_array_free(v_vec);
            try mlxOp(C.mlx_concatenate(v, v_vec, self.axis, stream));
        }
        try arraySet(&self.k, k.*);
        try arraySet(&self.v, v.*);
        self.is_empty = false;
    }

    pub fn get(self: *Self, k: *Array, v: *Array) MLXError!void {
        try arraySet(k, self.k);
        try arraySet(v, self.v);
    }

    pub fn set(self: *Self, k: *Array, v: *Array) MLXError!void {
        try arraySet(&self.k, k.*);
        try arraySet(&self.v, v.*);
        self.is_empty = false;
    }

    pub fn deinit(self: *Self) void {
        arrayFree(self.k);
        arrayFree(self.v);
    }
};

pub const Cache = struct {
    const Self = @This();

    layers: []KVCache,
    offset: c_int,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_layers: usize, axis: c_int) MLXError!Self {
        var layers = try allocator.alloc(KVCache, num_layers);
        for (0..num_layers) |idx| {
            layers[idx] = KVCache.init(axis);
        }
        return Self{
            .layers = layers,
            .offset = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.layers) |*cache| {
            cache.deinit();
        }
        self.allocator.free(self.layers);
    }
};

/// ============================================================================
/// Utility Functions
/// ============================================================================
pub fn createCausalMask(result: *Array, seq_len: c_int, offset: c_int, dtype: C.mlx_dtype, stream: Stream) MLXError!void {
    try ones(result, &[_]c_int{ seq_len, seq_len + offset }, C.MLX_INT32, stream);
    try tril(result, result.*, offset, stream);
    try where(result, result.*, float(0.0), float(-std.math.inf(f32)), stream);
    try astype(result, result.*, dtype, stream);
}

pub fn printArray(msg: []const u8, arr: Array) void {
    var str = C.mlx_string_new();
    defer _ = C.mlx_string_free(str);
    const ndim = C.mlx_array_ndim(arr);
    const shape = C.mlx_array_shape(arr);
    _ = C.mlx_array_tostring(&str, arr);
    std.debug.print("{s}\n{s}\n", .{ msg, C.mlx_string_data(str) });
    std.debug.print("Shape: [", .{});
    for (0..ndim) |idx| {
        if (idx > 0) std.debug.print(",", .{});
        std.debug.print("{d}", .{shape[idx]});
    }
    std.debug.print("]\n", .{});
}

pub fn printMapStr(msg: []const u8, map: *C.mlx_map_string_to_string) MLXError!void {
    const map_iter = C.mlx_map_string_to_string_iterator_new(map.*);
    defer _ = C.mlx_map_string_to_string_iterator_free(map_iter);
    var key: [*c]const u8 = undefined;
    var value: [*c]const u8 = undefined;
    std.debug.print("\n{s}:\n", .{msg});
    while (C.mlx_map_string_to_string_iterator_next(&key, &value, map_iter) == 0) std.debug.print("  {s}: {s}\n", .{ key, value });
}

pub fn printMapArr(msg: []const u8, map: *const MapStrArr) MLXError!void {
    const map_iter = C.mlx_map_string_to_array_iterator_new(map.*);
    defer _ = C.mlx_map_string_to_array_iterator_free(map_iter);
    var key: [*c]const u8 = undefined;
    var value = C.mlx_array_new();
    defer arrayFree(value);
    std.debug.print("\n{s}:\n", .{msg});
    while (C.mlx_map_string_to_array_iterator_next(&key, &value, map_iter) == 0) {
        const ndim = C.mlx_array_ndim(value);
        const shape = C.mlx_array_shape(value);
        std.debug.print("  {s}: shape=[", .{key});
        for (0..ndim) |idx| {
            if (idx > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{shape[idx]});
        }
        std.debug.print("]\n", .{});
    }
}

/// ============================================================================
/// Helper Functions
/// ============================================================================
fn defineUnaryOp(comptime c_func_name: []const u8) fn (*Array, anytype, Stream) MLXError!void {
    return struct {
        fn impl(result: *Array, a: anytype, stream: Stream) MLXError!void {
            const a_conv = arrayConverter(a);
            defer a_conv.deinit();
            try mlxOpWithLog(@field(C, c_func_name)(result, a_conv.arr, stream), c_func_name);
        }
    }.impl;
}

fn defineBinaryOp(comptime c_func_name: []const u8) fn (*Array, anytype, anytype, Stream) MLXError!void {
    const BinaryOpImpl = struct {
        fn impl(result: *Array, a: anytype, b: anytype, stream: Stream) MLXError!void {
            const a_conv = arrayConverter(a);
            const b_conv = arrayConverter(b);
            defer {
                a_conv.deinit();
                b_conv.deinit();
            }
            try mlxOpWithLog(@field(C, c_func_name)(result, a_conv.arr, b_conv.arr, stream), c_func_name);
        }
    };
    return BinaryOpImpl.impl;
}

fn defineReduceAllOp(comptime c_func_name: []const u8) fn (*Array, Array, bool, Stream) MLXError!void {
    return struct {
        fn impl(result: *Array, x: Array, keepdims: bool, stream: Stream) MLXError!void {
            try mlxOpWithLog(@field(C, c_func_name)(result, x, keepdims, stream), c_func_name);
        }
    }.impl;
}

fn defineReduceAxesOp(comptime c_func_name: []const u8) fn (*Array, Array, []const c_int, bool, Stream) MLXError!void {
    return struct {
        fn impl(result: *Array, x: Array, axes: []const c_int, keepdims: bool, stream: Stream) MLXError!void {
            try mlxOpWithLog(@field(C, c_func_name)(result, x, axes.ptr, axes.len, keepdims, stream), c_func_name);
        }
    }.impl;
}

fn defineFFTOp(comptime c_func_name: []const u8) fn (*Array, Array, c_int, c_int, Stream) MLXError!void {
    return struct {
        fn impl(result: *Array, x: Array, n: c_int, axis: c_int, stream: Stream) MLXError!void {
            try mlxOpWithLog(@field(C, c_func_name)(result, x, n, axis, stream), c_func_name);
        }
    }.impl;
}

fn defineFFTNOp(comptime c_func_name: []const u8) fn (*Array, Array, []const c_int, []const c_int, Stream) MLXError!void {
    return struct {
        fn impl(result: *Array, x: Array, n: []const c_int, axes: []const c_int, stream: Stream) MLXError!void {
            try mlxOpWithLog(@field(C, c_func_name)(result, x, n.ptr, n.len, axes.ptr, axes.len, stream), c_func_name);
        }
    }.impl;
}

fn arrayConverter(value: anytype) ArrayHandle {
    const T = @TypeOf(value);
    if (T == Array) {
        return ArrayHandle.init(value, false);
    } else if (T == FloatArg) {
        return ArrayHandle.init(C.mlx_array_new_float(value.value), true);
    } else if (T == IntArg) {
        return ArrayHandle.init(C.mlx_array_new_int(value.value), true);
    } else if (T == BoolArg) {
        return ArrayHandle.init(C.mlx_array_new_bool(value.value), true);
    } else {
        @compileError("Unsupported type: " ++ @typeName(T));
    }
}

const ArrayHandle = struct {
    arr: Array,
    owned: bool,

    pub fn init(array: Array, owned: bool) ArrayHandle {
        return .{ .arr = array, .owned = owned };
    }

    pub fn deinit(self: ArrayHandle) void {
        if (self.owned) {
            arrayFree(self.arr);
        }
    }
};

const FloatArg = struct {
    value: f32,

    pub fn init(value: f32) FloatArg {
        return .{ .value = value };
    }
};

const IntArg = struct {
    value: c_int,

    pub fn init(value: anytype) IntArg {
        return .{ .value = @intCast(value) };
    }
};

const BoolArg = struct {
    value: bool,

    pub fn init(value: bool) BoolArg {
        return .{ .value = value };
    }
};

pub const float = FloatArg.init;
pub const int = IntArg.init;
pub const bool_ = BoolArg.init;

/// ============================================================================
/// Experimental Array Shape Operations
/// ============================================================================
pub fn rEpeat(result: *Array, x: Array, comptime pattern: []const u8, dim_values: anytype, stream: Stream) !void {
    const n_repeat: c_int = inline for (std.meta.fields(@TypeOf(dim_values))) |field| break @field(dim_values, field.name);
    const axis = comptime blk: {
        var tokens = std.mem.tokenizeAny(u8, pattern[std.mem.indexOf(u8, pattern, "->").? + 2 ..], " ");
        var i: c_int = 0;
        while (tokens.next()) |token| : (i += 1) if (token[0] == '(') break :blk i;
    };
    try mlxOpWithLog(C.mlx_repeat(result, x, n_repeat, axis, stream), "rEpeat");
}

pub fn rEshap(result: *Array, x: Array, comptime pattern: []const u8, dim_values: anytype, stream: Stream) !void {
    const args = comptime blk: {
        const arrow = std.mem.indexOf(u8, pattern, "->").?;
        const popen = std.mem.indexOf(u8, pattern, "(").?;
        const pclos = std.mem.indexOf(u8, pattern, ")").?;
        const is_lt = popen < arrow;
        const axis_start = a1: {
            const side = if (is_lt) pattern[0..popen] else pattern[arrow + 2 .. popen];
            var toks_1 = std.mem.tokenize(u8, side, " ");
            var i = 0;
            while (toks_1.next()) |_| i += 1;
            break :a1 i;
        };
        var shape: [16][]const u8 = undefined;
        var toks_2 = std.mem.tokenize(u8, pattern[popen + 1 .. pclos], " ");
        var i: usize = 0;
        while (toks_2.next()) |tok| : (i += 1) {
            shape[i] = tok;
        }
        const axis_end = axis_start + i - 1; // flatten end index is inclusive
        var toks_3 = std.mem.tokenizeAny(u8, pattern[0..arrow], " ()");
        var toks_4 = std.mem.tokenizeAny(u8, pattern[arrow + 2 ..], " ()");
        var swap_start: ?c_int = null;
        var swap_end: ?c_int = null;
        var c: c_int = 0;
        while (toks_3.next()) |t3| : (c += 1) {
            if (!std.mem.eql(u8, t3, toks_4.next().?)) {
                if (swap_start != null) {
                    swap_end = c;
                    break;
                } else {
                    swap_start = c;
                }
            }
        }
        break :blk .{ .is_lt = is_lt, .axis_start = axis_start, .axis_end = axis_end, .shape = shape[0..i].*, .swap_start = swap_start, .swap_end = swap_end };
    };
    try arraySet(result, x);
    if (args.is_lt) {
        var shape: [args.shape.len]c_int = undefined;
        inline for (args.shape, 0..) |tok, i| {
            shape[i] = @field(dim_values, tok);
        }
        try mlxOpWithLog(C.mlx_unflatten(result, result.*, args.axis_start, &shape, args.shape.len, stream), "rEshap.unflatten");
        if (args.swap_start != null) try mlxOpWithLog(C.mlx_swapaxes(result, result.*, args.swap_start.?, args.swap_end.?, stream), "rEshap.swap");
    } else {
        if (args.swap_start != null) try mlxOpWithLog(C.mlx_swapaxes(result, result.*, args.swap_start.?, args.swap_end.?, stream), "rEshap.swap");
        try mlxOpWithLog(C.mlx_flatten(result, result.*, args.axis_start, args.axis_end, stream), "rEshap.flatten");
    }
}
