//! mlx.zig - Zig bindings for the MLX C API
//!
//! This module provides Zig-friendly wrappers around the MLX C API, offering
//! both low-level access to MLX functionality and higher-level abstractions
//! for common machine learning operations.
//!
//! Copyright 2025 Joe

const std = @import("std");
const C = @cImport({
    @cInclude("mlx/c/mlx.h");
    @cInclude("stdio.h");
});

/// ============================================================================
/// Types & Constants
/// ============================================================================
pub const Array = C.mlx_array;
pub const Stream = C.mlx_stream;
pub const VectorArray = C.mlx_vector_array;
pub const String = C.mlx_string;
pub const MapStrArr = C.mlx_map_string_to_array;
pub const MapStrStr = C.mlx_map_string_to_string;
pub const OptionalFloat = C.mlx_optional_float;
pub const defaultCpuStreamNew = C.mlx_default_cpu_stream_new;
pub const arrayNew = C.mlx_array_new;
pub const arrayDim = C.mlx_array_dim;
pub const arrayShape = C.mlx_array_shape;
pub const neg_inf_f32 = float(-std.math.inf(f32));
pub const pos_inf_f32 = float(std.math.inf(f32));
pub const PI = float(std.math.pi);
pub const TWO_PI = float(2.0 * std.math.pi);
pub const float = FloatArg.init;
pub const int = IntArg.init;
pub const bool_ = BoolArg.init;

pub const DTYPE = struct {
    pub const BOOL = C.MLX_BOOL;
    pub const INT32 = C.MLX_INT32;
    pub const UINT32 = C.MLX_UINT32;
    pub const FLOAT32 = C.MLX_FLOAT32;
    pub const FLOAT64 = C.MLX_FLOAT64;
    pub const BFLOAT16 = C.MLX_BFLOAT16;
};

pub const QuantConfig = struct {
    group_size: c_int,
    bits: c_int,
};

pub const FloatArg = struct {
    value: f32,

    pub fn init(value: f32) FloatArg {
        return .{ .value = value };
    }
};

pub const IntArg = struct {
    value: c_int,

    pub fn init(value: anytype) IntArg {
        return .{ .value = @intCast(value) };
    }
};

pub const BoolArg = struct {
    value: bool,

    pub fn init(value: bool) BoolArg {
        return .{ .value = value };
    }
};

/// ============================================================================
/// Array Operations
/// ============================================================================
pub fn add(result: *C.mlx_array, a: anytype, right: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_add", result, a, right, stream);
}

pub fn subtract(result: *C.mlx_array, a: anytype, right: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_subtract", result, a, right, stream);
}

pub fn multiply(result: *C.mlx_array, a: anytype, right: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_multiply", result, a, right, stream);
}

pub fn divide(result: *C.mlx_array, a: anytype, right: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_divide", result, a, right, stream);
}

pub fn power(result: *C.mlx_array, a: anytype, right: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_power", result, a, right, stream);
}

pub fn greater(result: *C.mlx_array, a: anytype, right: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_greater", result, a, right, stream);
}

pub fn greaterEqual(result: *C.mlx_array, a: anytype, right: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_greater_equal", result, a, right, stream);
}

pub fn less(result: *C.mlx_array, a: anytype, right: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_less", result, a, right, stream);
}

pub fn lessEqual(result: *C.mlx_array, a: anytype, right: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_less_equal", result, a, right, stream);
}

pub fn logicalOr(result: *C.mlx_array, a: anytype, right: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_logical_or", result, a, right, stream);
}

pub fn logicalAnd(result: *C.mlx_array, a: anytype, right: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_logical_and", result, a, right, stream);
}

pub fn sigmoid(result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_sigmoid", result, a, stream);
}

pub fn logicalNot(result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_logical_not", result, a, stream);
}

pub fn isnan(result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_isnan", result, a, stream);
}

pub fn where(result: *C.mlx_array, cond: C.mlx_array, x: anytype, y: anytype, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    const y_conv = toArray(y);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
        if (y_conv.temp) _ = C.mlx_array_free(y_conv.arr);
    }
    try mlxOp(C.mlx_where(result, cond, x_conv.arr, y_conv.arr, stream));
}

pub fn take(result: *C.mlx_array, a: anytype, indices: anytype, axis: c_int, stream: C.mlx_stream) !void {
    const a_conv = toArray(a);
    const indices_conv = toArray(indices);
    defer {
        if (a_conv.temp) _ = C.mlx_array_free(a_conv.arr);
        if (indices_conv.temp) _ = C.mlx_array_free(indices_conv.arr);
    }
    try mlxOp(C.mlx_take(result, a_conv.arr, indices_conv.arr, axis, stream));
}

pub fn softmax(result: *C.mlx_array, x: anytype, axes: []const c_int, precise: bool, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_softmax(result, x_conv.arr, axes.ptr, axes.len, precise, stream));
}

pub fn einsum(result: *C.mlx_array, arrays: anytype, pattern: [*:0]const u8, stream: C.mlx_stream) !void {
    const fields = @typeInfo(@TypeOf(arrays)).Struct.fields;
    var array_data: [fields.len]C.mlx_array = undefined;
    inline for (fields, 0..) |field, i| array_data[i] = @field(arrays, field.name);
    const operands = C.mlx_vector_array_new_data(&array_data[0], array_data.len);
    defer _ = C.mlx_vector_array_free(operands);
    try mlxOp(C.mlx_einsum(result, pattern, operands, stream));
}

pub fn repeat(result: *C.mlx_array, arr: anytype, repeats: c_int, axis: c_int, stream: C.mlx_stream) !void {
    const arr_conv = toArray(arr);
    defer {
        if (arr_conv.temp) _ = C.mlx_array_free(arr_conv.arr);
    }

    try mlxOp(C.mlx_repeat(result, arr_conv.arr, repeats, axis, stream));
}

pub fn reshape(result: *C.mlx_array, x: anytype, shape: []const c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_reshape(result, x_conv.arr, shape.ptr, shape.len, stream));
}

pub fn astype(result: *C.mlx_array, x: anytype, dtype: C.mlx_dtype, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_astype(result, x_conv.arr, dtype, stream));
}

pub fn tril(result: *C.mlx_array, x: anytype, offset: c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_tril(result, x_conv.arr, offset, stream));
}

pub fn argmax(result: *C.mlx_array, x: anytype, axis: c_int, keepdims: bool, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_argmax(result, x_conv.arr, axis, keepdims, stream));
}

pub fn ones(result: *C.mlx_array, shape: []const c_int, dtype: C.mlx_dtype, stream: C.mlx_stream) !void {
    try mlxOp(C.mlx_ones(result, shape.ptr, shape.len, dtype, stream));
}

pub fn arrayFree(arr: C.mlx_array) void {
    _ = C.mlx_array_free(arr);
}

pub fn streamFree(stream: C.mlx_stream) void {
    _ = C.mlx_stream_free(stream);
}

pub fn fastRope(result: *C.mlx_array, x: C.mlx_array, dims: c_int, traditional: bool, base: C.mlx_optional_float, scale: f32, offset: c_int, freqs: C.mlx_array, s: C.mlx_stream) !void {
    try mlxOp(C.mlx_fast_rope(result, x, dims, traditional, base, scale, offset, freqs, s));
}

pub fn arange(result: *C.mlx_array, start: f64, stop: f64, step: f64, dtype: C.mlx_dtype, stream: C.mlx_stream) !void {
    try mlxOp(C.mlx_arange(result, start, stop, step, dtype, stream));
}

pub fn rmsNorm(result: *C.mlx_array, x: anytype, weight: anytype, eps: f32, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    const weight_conv = toArray(weight);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
        if (weight_conv.temp) _ = C.mlx_array_free(weight_conv.arr);
    }
    try mlxOp(C.mlx_fast_rms_norm(result, x_conv.arr, weight_conv.arr, eps, stream));
}

pub fn item(dest: anytype, arr: C.mlx_array) !void {
    const T = @TypeOf(dest);
    const info = @typeInfo(T);

    if (info != .Pointer) @compileError("Expected pointer, got " ++ @typeName(T));

    const child = info.Pointer.child;
    const func_name = switch (child) {
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

    try mlxOp(@field(C, func_name)(dest, arr));
}

pub fn arraySetData(arr: *C.mlx_array, data: *const anyopaque, shape_arg: anytype, dtype: C.mlx_dtype) !void {
    var shape: [32]c_int = undefined;
    var len: usize = 0;

    const T = @TypeOf(shape_arg);
    const fields = @typeInfo(T).Struct.fields;

    inline for (fields, 0..) |field, idx| {
        shape[idx] = @intCast(@field(shape_arg, field.name));
        len = idx + 1;
    }

    try mlxOp(C.mlx_array_set_data(arr, data, &shape, @intCast(len), dtype));
}

pub fn arrayNewData(data: *const anyopaque, shape_arg: anytype, dtype: C.mlx_dtype) !C.mlx_array {
    var shape: [32]c_int = undefined;
    var len: usize = 0;
    const T = @TypeOf(shape_arg);
    const fields = @typeInfo(T).Struct.fields;

    inline for (fields, 0..) |field, idx| {
        shape[idx] = @intCast(@field(shape_arg, field.name));
        len = idx + 1;
    }

    const arr = C.mlx_array_new_data(data, &shape, @intCast(len), dtype);
    if (arr.ctx == null) return error.FailedToCreateArray;
    return arr;
}

fn mlxOp1(comptime func_name: []const u8, result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    const a_conv = toArray(a);
    defer {
        if (a_conv.temp) _ = C.mlx_array_free(a_conv.arr);
    }
    try mlxOp(@field(C, func_name)(result, a_conv.arr, stream));
}

fn mlxOp2(comptime func_name: []const u8, result: *C.mlx_array, a: anytype, right: anytype, stream: C.mlx_stream) !void {
    const a_conv = toArray(a);
    const b_conv = toArray(right);
    defer {
        if (a_conv.temp) _ = C.mlx_array_free(a_conv.arr);
        if (b_conv.temp) _ = C.mlx_array_free(b_conv.arr);
    }
    try mlxOp(@field(C, func_name)(result, a_conv.arr, b_conv.arr, stream));
}

fn toArray(value: anytype) struct { arr: C.mlx_array, temp: bool } {
    const T = @TypeOf(value);
    if (T == C.mlx_array) {
        return .{ .arr = value, .temp = false };
    } else if (T == FloatArg) {
        return .{ .arr = C.mlx_array_new_float(value.value), .temp = true };
    } else if (T == IntArg) {
        return .{ .arr = C.mlx_array_new_int(value.value), .temp = true };
    } else if (T == BoolArg) {
        return .{ .arr = C.mlx_array_new_bool(value.value), .temp = true };
    } else {
        @compileError("Unsupported type: " ++ @typeName(T));
    }
}

/// ============================================================================
/// Data Structures
/// ============================================================================
pub const KVCache = struct {
    const Self = @This();

    k: C.mlx_array,
    v: C.mlx_array,

    pub fn init() Self {
        return Self{
            .k = C.mlx_array_new(),
            .v = C.mlx_array_new(),
        };
    }

    fn sliceCache(self: *Self, offset: c_int, stream: C.mlx_stream) !void {
        if (offset >= C.mlx_array_dim(self.k, 2)) return;
        const ndim = C.mlx_array_ndim(self.k);
        const start = [_]c_int{0} ** 4;
        const strides = [_]c_int{1} ** 4;
        var stop = [_]c_int{0} ** 4;
        for (0..ndim) |idx| stop[idx] = C.mlx_array_dim(self.k, @intCast(idx));
        stop[2] = offset;
        try mlxOp(C.mlx_slice(&self.k, self.k, &start, ndim, &stop, ndim, &strides, ndim, stream));
        try mlxOp(C.mlx_slice(&self.k, self.v, &start, ndim, &stop, ndim, &strides, ndim, stream));
        std.debug.print("Cache offset set to {d}\n", .{offset});
    }

    pub fn update(self: *Self, k: *C.mlx_array, v: *C.mlx_array, offset: c_int, stream: C.mlx_stream) !void {
        if (offset > 0) {
            try self.sliceCache(offset, stream);
            var k_concat = [_]C.mlx_array{ self.k, k.* };
            const k_vec = C.mlx_vector_array_new_data(&k_concat[0], 2);
            defer _ = C.mlx_vector_array_free(k_vec);
            try mlxOp(C.mlx_concatenate(k, k_vec, 2, stream));
            var v_concat = [_]C.mlx_array{ self.v, v.* };
            const v_vec = C.mlx_vector_array_new_data(&v_concat[0], 2);
            defer _ = C.mlx_vector_array_free(v_vec);
            try mlxOp(C.mlx_concatenate(v, v_vec, 2, stream));
        }
        try mlxOp(C.mlx_array_set(&self.k, k.*));
        try mlxOp(C.mlx_array_set(&self.v, v.*));
    }

    pub fn deinit(self: *Self) void {
        _ = C.mlx_array_free(self.k);
        _ = C.mlx_array_free(self.v);
    }
};

pub const Cache = struct {
    const Self = @This();

    layers: []KVCache,
    offset: c_int,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_layers: usize) !Self {
        var layers = try allocator.alloc(KVCache, num_layers);
        for (0..num_layers) |idx| {
            layers[idx] = KVCache.init();
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

pub const Weight = struct {
    const Self = @This();
    weight: C.mlx_array,
    is_quantized: bool,
    scales: ?C.mlx_array = null,
    biases: ?C.mlx_array = null,
    group_size: ?c_int = null,
    bits: ?c_int = null,
    stream: C.mlx_stream,

    pub fn init(quant_config: ?QuantConfig, stream: C.mlx_stream) !Self {
        const is_quantized = quant_config != null;
        return Self{
            .weight = C.mlx_array_new(),
            .is_quantized = is_quantized,
            .scales = if (is_quantized) C.mlx_array_new() else null,
            .biases = if (is_quantized) C.mlx_array_new() else null,
            .group_size = if (is_quantized) quant_config.?.group_size else null,
            .bits = if (is_quantized) quant_config.?.bits else null,
            .stream = stream,
        };
    }

    pub fn load(self: *Self, key: []const u8, weights_map: *const C.mlx_map_string_to_array) !void {
        if (self.is_quantized) {
            try loadArray(&self.weight, key, "weight", weights_map);
            try loadArray(&self.scales.?, key, "scales", weights_map);
            try loadArray(&self.biases.?, key, "biases", weights_map);
        } else {
            try loadArray(&self.weight, key, "weight", weights_map);
        }
    }

    pub fn loadDequantized(self: *Self, key: []const u8, weights_map: *const C.mlx_map_string_to_array) !void {
        try self.load(key, weights_map);
        if (self.is_quantized) {
            try mlxOp(C.mlx_dequantize(&self.weight, self.weight, self.scales.?, self.biases.?, self.group_size.?, self.bits.?, self.stream));
            try mlxOp(C.mlx_array_free(self.scales.?));
            try mlxOp(C.mlx_array_free(self.biases.?));
            try mlxOp(C.mlx_array_eval(self.weight));
            self.is_quantized = false;
            self.group_size = null;
            self.bits = null;
        }
    }

    pub fn forward(self: *Self, result: *C.mlx_array, x: C.mlx_array) !void {
        if (self.is_quantized) {
            try mlxOp(C.mlx_quantized_matmul(result, x, self.weight, self.scales.?, self.biases.?, true, self.group_size.?, self.bits.?, self.stream));
        } else {
            try einsum(result, .{ x, self.weight }, "blh,dh->bld", self.stream);
        }
    }

    pub fn deinit(self: *Self) void {
        _ = C.mlx_array_free(self.weight);
        if (self.is_quantized) {
            _ = C.mlx_array_free(self.scales.?);
            _ = C.mlx_array_free(self.biases.?);
        }
    }
};

/// ============================================================================
/// File I/O and Model Loading
/// ============================================================================
pub const Safetensors = struct {
    const Self = @This();
    const MAX_PATH_LEN = 1024;

    file: ?*C.FILE,
    weights: C.mlx_map_string_to_array,
    stream: C.mlx_stream,

    pub fn load(path_safetensors: [:0]const u8, stream: C.mlx_stream) !Self {
        const file = C.fopen(path_safetensors.ptr, "rb") orelse return error.FileNotFound;
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
            return error.LoadWeightsFailed;
        }
        try printMap("Metadata", &meta);
        return Self{
            .file = file,
            .weights = weights,
            .stream = stream,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.file) |file| {
            _ = C.fclose(file);
            self.file = null;
        }
        _ = C.mlx_map_string_to_array_free(self.weights);
    }
};

pub fn loadArray(weight: *C.mlx_array, name: []const u8, ext: ?[]const u8, weights_map: *const C.mlx_map_string_to_array) !void {
    var buf: [1024]u8 = undefined;
    const key = if (ext) |e| try std.fmt.bufPrintZ(&buf, "{s}.{s}", .{ name, e }) else name;
    try mlxOp(C.mlx_map_string_to_array_get(weight, weights_map.*, key.ptr));
    try mlxOp(C.mlx_array_eval(weight.*));
}

/// ============================================================================
/// Utility Functions
/// ============================================================================
pub fn mlxOp(result: c_int) !void {
    if (result != 0) return error.MLXOperationFailed;
}

pub fn createCausalMask(result: *C.mlx_array, seq_len: c_int, offset: c_int, stream: C.mlx_stream) !void {
    try ones(result, &[_]c_int{ seq_len, seq_len + offset }, DTYPE.INT32, stream);
    try tril(result, result.*, offset, stream);
    try where(result, result.*, float(0.0), neg_inf_f32, stream);
    try astype(result, result.*, DTYPE.BFLOAT16, stream);
}

pub fn printArray(msg: []const u8, arr: C.mlx_array) void {
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

pub fn printMap(msg: []const u8, map: *C.mlx_map_string_to_string) !void {
    const map_iter = C.mlx_map_string_to_string_iterator_new(map.*);
    defer _ = C.mlx_map_string_to_string_iterator_free(map_iter);
    var key: [*c]const u8 = undefined;
    var value: [*c]const u8 = undefined;
    std.debug.print("\n{s}:\n", .{msg});
    while (C.mlx_map_string_to_string_iterator_next(&key, &value, map_iter) == 0) std.debug.print("  {s}: {s}\n", .{ key, value });
}

/// ============================================================================
/// Experimental Array Shape Operations
/// ============================================================================
/// CAUTION: These functions are experimental implementations of einops-like
/// operations with significant limitations. They are poorly optimized and are
/// GUARANTEED to fail for most common use cases and should be considered prototypes
/// rather than production-ready.
///
/// Limitations include:
/// - Expecting a specific format with parentheses at exact positions in the pattern
/// - Only supporting a small subset of simple patterns
/// - Lacking proper error handling for many edge cases (may silently produce incorrect
///   results for unsupported patterns)
///
/// Use standard MLX reshape/transpose or repeat operations for production code and these
/// functions only when you've thoroughly tested them with your specific patterns.
pub fn rEpeat(result: *C.mlx_array, x: C.mlx_array, comptime pattern: []const u8, dim_values: anytype, stream: C.mlx_stream) !void {
    const n_repeat: c_int = inline for (std.meta.fields(@TypeOf(dim_values))) |field| break @field(dim_values, field.name);
    const axis = comptime blk: {
        var tokens = std.mem.tokenizeAny(u8, pattern[std.mem.indexOf(u8, pattern, "->").? + 2 ..], " ");
        var i: c_int = 0;
        while (tokens.next()) |token| : (i += 1) if (token[0] == '(') break :blk i;
    };
    try mlxOp(C.mlx_repeat(result, x, @intCast(n_repeat), @intCast(axis), stream));
}

pub fn rEshap(result: *C.mlx_array, x: C.mlx_array, comptime pattern: []const u8, dim_values: anytype, stream: C.mlx_stream) !void {
    const arrow = comptime std.mem.indexOf(u8, pattern, "->").?;
    const popen = comptime std.mem.indexOf(u8, pattern, "(").?;
    const pclos = comptime std.mem.indexOf(u8, pattern, ")").?;
    const is_lt = comptime popen < arrow;
    const side = comptime if (is_lt) pattern[0..popen] else pattern[arrow + 2 .. popen];
    var toks_1 = comptime std.mem.tokenize(u8, side, " ");
    var toks_2 = comptime std.mem.tokenize(u8, pattern[popen + 1 .. pclos], " ");
    const i_shape = C.mlx_array_shape(x);
    var shape: [16]c_int = undefined;
    var i: usize = 0;
    while (toks_1.next()) |_| : (i += 1) shape[i] = i_shape[i];
    if (is_lt) {
        while (toks_2.next()) |tok| {
            inline for (std.meta.fields(@TypeOf(dim_values))) |f| {
                if (std.mem.eql(u8, tok, f.name)) {
                    shape[i] = @field(dim_values, f.name);
                    i += 1;
                }
            }
        }
    } else {
        shape[i] = -1;
        i += 1;
    }
    return mlxOp(C.mlx_reshape(result, x, &shape, i, stream));
}
