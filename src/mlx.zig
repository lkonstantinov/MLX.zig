//! mlx.zig - MLX Bindings
//!
//! Copyright 2025 Joe

const std = @import("std");
const allocJoin = @import("utils.zig").allocJoin;
pub const C = @cImport({
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

pub const DTYPE = struct {
    pub const BOOL = C.MLX_BOOL;
    pub const INT32 = C.MLX_INT32;
    pub const UINT32 = C.MLX_UINT32;
    pub const FLOAT16 = C.MLX_FLOAT16;
    pub const FLOAT32 = C.MLX_FLOAT32;
    pub const FLOAT64 = C.MLX_FLOAT64;
    pub const BFLOAT16 = C.MLX_BFLOAT16;
};

/// ============================================================================
/// Unary Operations
/// ============================================================================
pub fn sigmoid(result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_sigmoid", result, a, stream);
}

pub fn logicalNot(result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_logical_not", result, a, stream);
}

pub fn isnan(result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_isnan", result, a, stream);
}

pub fn sin(result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_sin", result, a, stream);
}

pub fn cos(result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_cos", result, a, stream);
}

pub fn exp(result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_exp", result, a, stream);
}

pub fn abs(result: *C.mlx_array, x: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_abs", result, x, stream);
}

pub fn square(result: *C.mlx_array, x: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_square", result, x, stream);
}

pub fn log(result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_log", result, a, stream);
}

pub fn log10(result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_log10", result, a, stream);
}

pub fn transpose_all(result: *C.mlx_array, a: anytype, stream: C.mlx_stream) !void {
    return mlxOp1("mlx_transpose_all", result, a, stream);
}

pub fn softmax(result: *C.mlx_array, x: anytype, axes: []const c_int, precise: bool, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_softmax(result, x_conv.arr, axes.ptr, axes.len, precise, stream));
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

pub fn slice(result: *C.mlx_array, a: anytype, start: []const c_int, stop: []const c_int, strides: []const c_int, stream: C.mlx_stream) !void {
    const a_conv = toArray(a);
    defer {
        if (a_conv.temp) _ = C.mlx_array_free(a_conv.arr);
    }
    try mlxOp(C.mlx_slice(result, a_conv.arr, start.ptr, start.len, stop.ptr, stop.len, strides.ptr, strides.len, stream));
}

pub fn asStrided(result: *C.mlx_array, a: anytype, shape: []const c_int, strides: []const i64, offset: usize, stream: C.mlx_stream) !void {
    const a_conv = toArray(a);
    defer {
        if (a_conv.temp) _ = C.mlx_array_free(a_conv.arr);
    }
    try mlxOp(C.mlx_as_strided(result, a_conv.arr, shape.ptr, shape.len, strides.ptr, strides.len, offset, stream));
}

pub fn expand_dims(result: *C.mlx_array, a: anytype, axes: []const c_int, stream: C.mlx_stream) !void {
    const a_conv = toArray(a);
    defer {
        if (a_conv.temp) _ = C.mlx_array_free(a_conv.arr);
    }
    try mlxOp(C.mlx_expand_dims(result, a_conv.arr, axes.ptr, axes.len, stream));
}

/// ============================================================================
/// Binary Operations
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

pub fn matmul(result: *C.mlx_array, a: anytype, b: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_matmul", result, a, b, stream);
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

pub fn pad(result: *C.mlx_array, a: anytype, axes: []const c_int, low_pad: []const c_int, high_pad: []const c_int, pad_value: anytype, pad_mode: [*:0]const u8, stream: C.mlx_stream) !void {
    const a_conv = toArray(a);
    const pad_val_conv = toArray(pad_value);
    defer {
        if (a_conv.temp) _ = C.mlx_array_free(a_conv.arr);
        if (pad_val_conv.temp) _ = C.mlx_array_free(pad_val_conv.arr);
    }
    try mlxOp(C.mlx_pad(result, a_conv.arr, axes.ptr, axes.len, low_pad.ptr, low_pad.len, high_pad.ptr, high_pad.len, pad_val_conv.arr, pad_mode, stream));
}

/// ============================================================================
/// Vector Operations
/// ============================================================================
pub fn einsum(result: *C.mlx_array, arrays: anytype, pattern: [*:0]const u8, stream: C.mlx_stream) !void {
    const fields = @typeInfo(@TypeOf(arrays)).Struct.fields;
    var array_data: [fields.len]C.mlx_array = undefined;
    inline for (fields, 0..) |field, i| array_data[i] = @field(arrays, field.name);
    const operands = C.mlx_vector_array_new_data(&array_data[0], array_data.len);
    defer _ = C.mlx_vector_array_free(operands);
    try mlxOp(C.mlx_einsum(result, pattern, operands, stream));
}

pub fn concatenate(result: *C.mlx_array, arrays: anytype, axis: c_int, stream: C.mlx_stream) !void {
    const fields = @typeInfo(@TypeOf(arrays)).Struct.fields;
    var array_data: [fields.len]C.mlx_array = undefined;
    inline for (fields, 0..) |field, i| array_data[i] = @field(arrays, field.name);
    const vector_arrays = C.mlx_vector_array_new_data(&array_data[0], array_data.len);
    defer _ = C.mlx_vector_array_free(vector_arrays);
    try mlxOp(C.mlx_concatenate(result, vector_arrays, axis, stream));
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

/// ============================================================================
/// Min/Max Operations
/// ============================================================================
pub fn minimum(result: *C.mlx_array, a: anytype, b: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_minimum", result, a, b, stream);
}

pub fn maximum(result: *C.mlx_array, a: anytype, b: anytype, stream: C.mlx_stream) !void {
    return mlxOp2("mlx_maximum", result, a, b, stream);
}

pub fn min_all(result: *C.mlx_array, a: anytype, keepdims: bool, stream: C.mlx_stream) !void {
    const a_conv = toArray(a);
    defer {
        if (a_conv.temp) _ = C.mlx_array_free(a_conv.arr);
    }
    try mlxOp(C.mlx_min_all(result, a_conv.arr, keepdims, stream));
}

pub fn min(result: *C.mlx_array, a: anytype, axes: []const c_int, keepdims: bool, stream: C.mlx_stream) !void {
    const a_conv = toArray(a);
    defer {
        if (a_conv.temp) _ = C.mlx_array_free(a_conv.arr);
    }
    try mlxOp(C.mlx_min(result, a_conv.arr, axes.ptr, axes.len, keepdims, stream));
}

pub fn max_all(result: *C.mlx_array, a: anytype, keepdims: bool, stream: C.mlx_stream) !void {
    const a_conv = toArray(a);
    defer {
        if (a_conv.temp) _ = C.mlx_array_free(a_conv.arr);
    }
    try mlxOp(C.mlx_max_all(result, a_conv.arr, keepdims, stream));
}

pub fn max(result: *C.mlx_array, a: anytype, axes: []const c_int, keepdims: bool, stream: C.mlx_stream) !void {
    const a_conv = toArray(a);
    defer {
        if (a_conv.temp) _ = C.mlx_array_free(a_conv.arr);
    }
    try mlxOp(C.mlx_max(result, a_conv.arr, axes.ptr, axes.len, keepdims, stream));
}

/// ============================================================================
/// FFT Operations
/// ============================================================================
pub fn rfft(result: *C.mlx_array, x: anytype, n: c_int, axis: c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_fft_rfft(result, x_conv.arr, n, axis, stream));
}

pub fn irfft(result: *C.mlx_array, x: anytype, n: c_int, axis: c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_fft_irfft(result, x_conv.arr, n, axis, stream));
}

pub fn fft(result: *C.mlx_array, x: anytype, n: c_int, axis: c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_fft_fft(result, x_conv.arr, n, axis, stream));
}

pub fn ifft(result: *C.mlx_array, x: anytype, n: c_int, axis: c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_fft_ifft(result, x_conv.arr, n, axis, stream));
}

pub fn fft2(result: *C.mlx_array, x: anytype, n: []const c_int, axes: []const c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_fft_fft2(result, x_conv.arr, n.ptr, n.len, axes.ptr, axes.len, stream));
}

pub fn ifft2(result: *C.mlx_array, x: anytype, n: []const c_int, axes: []const c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_fft_ifft2(result, x_conv.arr, n.ptr, n.len, axes.ptr, axes.len, stream));
}

pub fn rfft2(result: *C.mlx_array, x: anytype, n: []const c_int, axes: []const c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_fft_rfft2(result, x_conv.arr, n.ptr, n.len, axes.ptr, axes.len, stream));
}

pub fn irfft2(result: *C.mlx_array, x: anytype, n: []const c_int, axes: []const c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_fft_irfft2(result, x_conv.arr, n.ptr, n.len, axes.ptr, axes.len, stream));
}

pub fn fftn(result: *C.mlx_array, x: anytype, n: []const c_int, axes: []const c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_fft_fftn(result, x_conv.arr, n.ptr, n.len, axes.ptr, axes.len, stream));
}

pub fn ifftn(result: *C.mlx_array, x: anytype, n: []const c_int, axes: []const c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_fft_ifftn(result, x_conv.arr, n.ptr, n.len, axes.ptr, axes.len, stream));
}

pub fn rfftn(result: *C.mlx_array, x: anytype, n: []const c_int, axes: []const c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_fft_rfftn(result, x_conv.arr, n.ptr, n.len, axes.ptr, axes.len, stream));
}

pub fn irfftn(result: *C.mlx_array, x: anytype, n: []const c_int, axes: []const c_int, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
    }
    try mlxOp(C.mlx_fft_irfftn(result, x_conv.arr, n.ptr, n.len, axes.ptr, axes.len, stream));
}

/// ============================================================================
/// Other Operations
/// ============================================================================
pub fn linspace(result: *C.mlx_array, start: f64, stop: f64, num: c_int, dtype: C.mlx_dtype, stream: C.mlx_stream) !void {
    try mlxOp(C.mlx_linspace(result, start, stop, num, dtype, stream));
}

pub fn arange(result: *C.mlx_array, start: f64, stop: f64, step: f64, dtype: C.mlx_dtype, stream: C.mlx_stream) !void {
    try mlxOp(C.mlx_arange(result, start, stop, step, dtype, stream));
}

pub fn ones(result: *C.mlx_array, shape: []const c_int, dtype: C.mlx_dtype, stream: C.mlx_stream) !void {
    try mlxOp(C.mlx_ones(result, shape.ptr, shape.len, dtype, stream));
}

pub fn split(outputs: anytype, a: C.mlx_array, indices: []const c_int, axis: c_int, stream: C.mlx_stream) !void {
    var results = C.mlx_vector_array_new();
    defer _ = C.mlx_vector_array_free(results);
    try mlxOp(C.mlx_split(&results, a, indices.ptr, indices.len, axis, stream));
    inline for (std.meta.fields(@TypeOf(outputs)), 0..) |field, i| {
        try mlxOp(C.mlx_vector_array_get(@field(outputs, field.name), results, i));
    }
}

pub fn arrayFree(arr: C.mlx_array) void {
    _ = C.mlx_array_free(arr);
}

pub fn streamFree(stream: C.mlx_stream) void {
    _ = C.mlx_stream_free(stream);
}

pub const defaultCpuStreamNew = C.mlx_default_cpu_stream_new;
pub const defaultGpuStreamNew = C.mlx_default_gpu_stream_new;
pub const arrayNew = C.mlx_array_new;
pub const arrayNewFloat = C.mlx_array_new_float;
pub const arrayDim = C.mlx_array_dim;
pub const arrayShape = C.mlx_array_shape;

/// ============================================================================
/// Fast Operations
/// ============================================================================
pub fn fastRope(result: *C.mlx_array, x: C.mlx_array, dims: c_int, traditional: bool, base: C.mlx_optional_float, scale: f32, offset: c_int, freqs: C.mlx_array, s: C.mlx_stream) !void {
    try mlxOp(C.mlx_fast_rope(result, x, dims, traditional, base, scale, offset, freqs, s));
}

pub fn fastRmsNorm(result: *C.mlx_array, x: anytype, weight: anytype, eps: f32, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    const weight_conv = toArray(weight);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
        if (weight_conv.temp) _ = C.mlx_array_free(weight_conv.arr);
    }
    try mlxOp(C.mlx_fast_rms_norm(result, x_conv.arr, weight_conv.arr, eps, stream));
}

pub fn fastLayerNorm(result: *C.mlx_array, x: anytype, weight: anytype, bias: anytype, eps: f32, stream: C.mlx_stream) !void {
    const x_conv = toArray(x);
    const weight_conv = toArray(weight);
    const bias_conv = toArray(bias);
    defer {
        if (x_conv.temp) _ = C.mlx_array_free(x_conv.arr);
        if (weight_conv.temp) _ = C.mlx_array_free(weight_conv.arr);
        if (bias_conv.temp) _ = C.mlx_array_free(bias_conv.arr);
    }
    try mlxOp(C.mlx_fast_layer_norm(result, x_conv.arr, weight_conv.arr, bias_conv.arr, eps, stream));
}

pub fn fastScaledDotProductAttention(result: *C.mlx_array, queries: C.mlx_array, keys: C.mlx_array, values: C.mlx_array, scale: f32, mask: ?C.mlx_array, stream: C.mlx_stream) !void {
    const memory_threshold = C.mlx_optional_int{ .has_value = false, .value = 0 };
    const mask_ptr = if (mask) |m| m else C.mlx_array_empty;
    try mlxOp(C.mlx_fast_scaled_dot_product_attention(result, queries, keys, values, scale, mask_ptr, memory_threshold, stream));
}

/// ============================================================================
/// Custom Operations
/// ============================================================================
pub const Linear = struct {
    const Self = @This();
    key: []const u8,
    weight: Weight,
    has_bias: bool,
    bias: ?Array,
    stream: Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, has_bias: bool, quant_config: ?QuantConfig, stream: Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        const bias = if (has_bias) arrayNew() else null;
        return Self{
            .weight = try Weight.init(quant_config, stream),
            .has_bias = has_bias,
            .bias = bias,
            .stream = stream,
            .key = key,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const MapStrArr) !void {
        try self.weight.load(self.key, weights_map);
        if (self.has_bias) {
            try loadArray(&self.bias.?, self.key, "bias", weights_map);
        }
    }

    pub fn forward(self: *Self, result: *Array, x: Array) !void {
        try self.weight.forward(result, x);
        if (self.has_bias) {
            try add(result, result.*, self.bias.?, self.stream);
        }
    }

    pub fn deinit(self: *Self) void {
        self.weight.deinit();
        if (self.has_bias) {
            _ = arrayFree(self.bias.?);
        }
        self.allocator.free(self.key);
    }
};

pub const Embedding = struct {
    const Self = @This();
    key: []const u8,
    weight: Weight,
    stream: Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, quant_config: ?QuantConfig, stream: Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        return Self{
            .weight = try Weight.init(quant_config, stream),
            .stream = stream,
            .key = key,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const MapStrArr) !void {
        try self.weight.loadDequantized(self.key, weights_map);
    }

    pub fn forward(self: *Self, result: *Array, toks: Array) !void {
        try take(result, self.weight.weight, toks, 0, self.stream);
    }

    pub fn asLinear(self: *Self, result: *Array, x: Array) !void {
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
    weight: Array,
    bias: Array,
    stream: Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, dims: c_int, eps: f32, stream: Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        return Self{
            .dims = dims,
            .eps = eps,
            .weight = arrayNew(),
            .bias = arrayNew(),
            .stream = stream,
            .key = key,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const MapStrArr) !void {
        try loadArray(&self.weight, self.key, "weight", weights_map);
        try loadArray(&self.bias, self.key, "bias", weights_map);
    }

    pub fn forward(self: *Self, result: *Array, x: Array) !void {
        try fastLayerNorm(result, x, self.weight, self.bias, self.eps, self.stream);
    }

    pub fn deinit(self: *Self) void {
        arrayFree(self.weight);
        arrayFree(self.bias);
        self.allocator.free(self.key);
    }
};

pub const RMSNorm = struct {
    const Self = @This();
    key: []const u8,
    eps: f32,
    weight: Array,
    stream: Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, eps: f32, stream: Stream) !Self {
        const key = try allocJoin(allocator, parent, name);
        errdefer allocator.free(key);
        return Self{
            .weight = arrayNew(),
            .eps = eps,
            .stream = stream,
            .key = key,
            .allocator = allocator,
        };
    }

    pub fn load(self: *Self, weights_map: *const MapStrArr) !void {
        try loadArray(&self.weight, self.key, "weight", weights_map);
    }

    pub fn forward(self: *Self, result: *Array, x: Array) !void {
        try fastRmsNorm(result, x, self.weight, self.eps, self.stream);
    }

    pub fn deinit(self: *Self) void {
        arrayFree(self.weight);
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
    weight: Array,
    bias: ?Array,
    stream: Stream,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: []const u8, name: []const u8, in_channels: c_int, out_channels: c_int, kernel_size: c_int, stride: c_int, padding: c_int, stream: Stream) !Self {
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
            .weight = arrayNew(),
            .bias = arrayNew(),
            .stream = stream,
            .key = key,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        arrayFree(self.weight);
        if (self.has_bias) {
            arrayFree(self.bias.?);
        }
        self.allocator.free(self.key);
    }
    pub fn load(self: *Self, weights_map: *const MapStrArr) !void {
        try loadArray(&self.weight, self.key, "weight", weights_map);
        try mlxOp(C.mlx_swapaxes(&self.weight, self.weight, 1, 2, self.stream));
        if (self.has_bias) {
            try loadArray(&self.bias.?, self.key, "bias", weights_map);
        }
    }

    pub fn forward(self: *Self, result: *Array, x: Array) !void {
        try mlxOp(C.mlx_conv1d(result, x, self.weight, self.stride, self.padding, self.dilation, self.groups, self.stream));
        if (self.has_bias) {
            try add(result, result.*, self.bias.?, self.stream);
        }
    }
};

pub fn gelu(result: *Array, x: Array, stream: Stream) !void {
    var erf_x = arrayNew();
    defer arrayFree(erf_x);
    try divide(&erf_x, x, float(@sqrt(2.0)), stream);
    try mlxOp(C.mlx_erf(&erf_x, erf_x, stream));
    try add(&erf_x, erf_x, float(1.0), stream);
    try multiply(&erf_x, erf_x, float(0.5), stream);
    try multiply(result, x, erf_x, stream);
}

/// ============================================================================
/// Helpers
/// ============================================================================
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

pub const float = FloatArg.init;
pub const int = IntArg.init;
pub const bool_ = BoolArg.init;

pub fn mlxOp(result: c_int) !void {
    if (result != 0) return error.MLXOperationFailed;
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
pub const QuantConfig = struct {
    group_size: c_int,
    bits: c_int,
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

pub const KVCache = struct {
    const Self = @This();

    k: C.mlx_array,
    v: C.mlx_array,
    axis: c_int,
    is_empty: bool = true,

    pub fn init(axis: c_int) Self {
        return Self{ .k = C.mlx_array_new(), .v = C.mlx_array_new(), .axis = axis };
    }

    fn sliceCache(self: *Self, offset: c_int, stream: C.mlx_stream) !void {
        if (offset >= C.mlx_array_dim(self.k, self.axis)) return;
        const ndim = C.mlx_array_ndim(self.k);
        const start = [_]c_int{0} ** 4;
        const strides = [_]c_int{1} ** 4;
        var stop = [_]c_int{0} ** 4;
        for (0..ndim) |idx| stop[idx] = C.mlx_array_dim(self.k, @intCast(idx));
        stop[@intCast(self.axis)] = offset;
        try mlxOp(C.mlx_slice(&self.k, self.k, &start, ndim, &stop, ndim, &strides, ndim, stream));
        try mlxOp(C.mlx_slice(&self.k, self.v, &start, ndim, &stop, ndim, &strides, ndim, stream));
        std.debug.print("Cache offset set to {d}\n", .{offset});
    }

    pub fn update(self: *Self, k: *C.mlx_array, v: *C.mlx_array, offset: ?c_int, stream: C.mlx_stream) !void {
        const offset_ = if (offset) |o| o else if (self.is_empty) 0 else C.mlx_array_dim(self.k, self.axis);
        if (offset_ > 0) {
            try self.sliceCache(offset_, stream);
            var k_concat = [_]C.mlx_array{ self.k, k.* };
            const k_vec = C.mlx_vector_array_new_data(&k_concat[0], 2);
            defer _ = C.mlx_vector_array_free(k_vec);
            try mlxOp(C.mlx_concatenate(k, k_vec, self.axis, stream));
            var v_concat = [_]C.mlx_array{ self.v, v.* };
            const v_vec = C.mlx_vector_array_new_data(&v_concat[0], 2);
            defer _ = C.mlx_vector_array_free(v_vec);
            try mlxOp(C.mlx_concatenate(v, v_vec, self.axis, stream));
        }
        try mlxOp(C.mlx_array_set(&self.k, k.*));
        try mlxOp(C.mlx_array_set(&self.v, v.*));
        self.is_empty = false;
    }

    pub fn get(self: *Self, k: *C.mlx_array, v: *C.mlx_array) !void {
        try mlxOp(C.mlx_array_set(k, self.k));
        try mlxOp(C.mlx_array_set(v, self.v));
    }

    pub fn set(self: *Self, k: *C.mlx_array, v: *C.mlx_array) !void {
        try mlxOp(C.mlx_array_set(&self.k, k.*));
        try mlxOp(C.mlx_array_set(&self.v, v.*));
        self.is_empty = false;
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

    pub fn init(allocator: std.mem.Allocator, num_layers: usize, axis: c_int) !Self {
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
        // try printMapStr("Metadata", &meta);
        // try printMapArr("Weights", &weights);
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
    // std.debug.print("Loading {s}\n", .{key});
    try mlxOp(C.mlx_map_string_to_array_get(weight, weights_map.*, key.ptr));
    try mlxOp(C.mlx_array_eval(weight.*));
}

/// ============================================================================
/// Utility Functions
/// ============================================================================
pub fn createCausalMask(result: *C.mlx_array, seq_len: c_int, offset: c_int, dtype: C.mlx_dtype, stream: C.mlx_stream) !void {
    try ones(result, &[_]c_int{ seq_len, seq_len + offset }, DTYPE.INT32, stream);
    try tril(result, result.*, offset, stream);
    try where(result, result.*, float(0.0), float(-std.math.inf(f32)), stream);
    try astype(result, result.*, dtype, stream);
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

pub fn printMapStr(msg: []const u8, map: *C.mlx_map_string_to_string) !void {
    const map_iter = C.mlx_map_string_to_string_iterator_new(map.*);
    defer _ = C.mlx_map_string_to_string_iterator_free(map_iter);
    var key: [*c]const u8 = undefined;
    var value: [*c]const u8 = undefined;
    std.debug.print("\n{s}:\n", .{msg});
    while (C.mlx_map_string_to_string_iterator_next(&key, &value, map_iter) == 0) std.debug.print("  {s}: {s}\n", .{ key, value });
}

pub fn printMapArr(msg: []const u8, map: *const C.mlx_map_string_to_array) !void {
    const map_iter = C.mlx_map_string_to_array_iterator_new(map.*);
    defer _ = C.mlx_map_string_to_array_iterator_free(map_iter);

    var key: [*c]const u8 = undefined;
    var value = C.mlx_array_new();
    defer _ = C.mlx_array_free(value);

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
    var toks_3 = comptime std.mem.tokenizeAny(u8, pattern[0..arrow], " ()");
    var toks_4 = comptime std.mem.tokenizeAny(u8, pattern[arrow + 2 ..], " ()");
    var a: ?c_int = null;
    var b: ?c_int = null;
    var c: c_int = 0;
    while (toks_3.next()) |t3| : (c += 1) {
        if (!std.mem.eql(u8, t3, toks_4.next().?)) {
            if (a != null) {
                b = c;
                break;
            } else {
                a = c;
            }
        }
    }
    var tmp = arrayNew();
    defer arrayFree(tmp);
    if (!is_lt and a != null) {
        try mlxOp(C.mlx_swapaxes(&tmp, x, a.?, b.?, stream));
    } else {
        try mlxOp(C.mlx_array_set(&tmp, x));
    }
    const i_shape = C.mlx_array_shape(tmp);
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
    try mlxOp(C.mlx_reshape(&tmp, tmp, &shape, i, stream));
    if (is_lt and a != null) {
        try mlxOp(C.mlx_swapaxes(result, tmp, a.?, b.?, stream));
    } else {
        try mlxOp(C.mlx_array_set(result, tmp));
    }
}
