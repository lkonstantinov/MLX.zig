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

const MLXBuildOptions = struct {
    build_metal: bool = false,
    build_cpu: bool = true,
    build_safetensors: bool = true,
    build_gguf: bool = false,
    build_examples: bool = false,
    metal_output_path: []const u8,
    fn fromOptions(b: *std.Build) !MLXBuildOptions {
        const default_rel_dir = "lib/metal";
        const default_path = try std.fmt.allocPrint(b.allocator, "{s}/{s}", .{ b.install_prefix, default_rel_dir });
        return .{
            .metal_output_path = b.option([]const u8, "metal-output-path", "Path to the metallib") orelse default_path,
        };
    }
};

const CPP_FLAGS = [_][]const u8{
    "-std=c++17",
    "-fPIC",
    "-DACCELERATE_NEW_LAPACK",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-fexceptions",
};

const MLXC_CPP_FLAGS = [_][]const u8{
    "-std=c++17",
    "-fPIC",
    "-frtti",
    "-fexceptions",
};

const C_FLAGS = [_][]const u8{
    "-fPIC",
    "-DACCELERATE_NEW_LAPACK",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-fexceptions",
};

const core_sources = [_][]const u8{
    "allocator.cpp",
    "array.cpp",
    "compile.cpp",
    "device.cpp",
    "dtype.cpp",
    "einsum.cpp",
    "fast.cpp",
    "fft.cpp",
    "ops.cpp",
    "graph_utils.cpp",
    "primitives.cpp",
    "random.cpp",
    "scheduler.cpp",
    "transforms.cpp",
    "utils.cpp",
    "linalg.cpp",
    "io/load.cpp",
};

const common_sources = [_][]const u8{
    "backend/common/arg_reduce.cpp",
    "backend/common/binary.cpp",
    "backend/common/compiled.cpp",
    "backend/common/common.cpp",
    "backend/common/conv.cpp",
    "backend/common/copy.cpp",
    "backend/common/eigh.cpp",
    "backend/common/erf.cpp",
    "backend/common/fft.cpp",
    "backend/common/hadamard.cpp",
    "backend/common/masked_mm.cpp",
    "backend/common/primitives.cpp",
    "backend/common/quantized.cpp",
    "backend/common/reduce.cpp",
    "backend/common/reduce_utils.cpp",
    "backend/common/scan.cpp",
    "backend/common/select.cpp",
    "backend/common/slicing.cpp",
    "backend/common/softmax.cpp",
    "backend/common/sort.cpp",
    "backend/common/threefry.cpp",
    "backend/common/indexing.cpp",
    "backend/common/load.cpp",
    "backend/common/qrf.cpp",
    "backend/common/svd.cpp",
    "backend/common/inverse.cpp",
    "backend/common/cholesky.cpp",
    "backend/common/utils.cpp",
};

const accelerate_sources = [_][]const u8{
    "backend/accelerate/conv.cpp",
    "backend/accelerate/matmul.cpp",
    "backend/accelerate/primitives.cpp",
    "backend/accelerate/quantized.cpp",
    "backend/accelerate/reduce.cpp",
    "backend/accelerate/softmax.cpp",
};

const no_metal_sources = [_][]const u8{
    "backend/no_metal/allocator.cpp",
    "backend/no_metal/event.cpp",
    "backend/no_metal/metal.cpp",
    "backend/no_metal/primitives.cpp",
};

const mlx_c_sources = [_][]const u8{
    "array.cpp",
    "closure.cpp",
    "compile.cpp",
    "device.cpp",
    "distributed.cpp",
    "distributed_group.cpp",
    "error.cpp",
    "fast.cpp",
    "fft.cpp",
    "io.cpp",
    "linalg.cpp",
    "map.cpp",
    "metal.cpp",
    "ops.cpp",
    "random.cpp",
    "stream.cpp",
    "string.cpp",
    "transforms.cpp",
    "transforms_impl.cpp",
    "vector.cpp",
};

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = std.builtin.OptimizeMode.ReleaseFast;
    const options = try MLXBuildOptions.fromOptions(b);
    const pcre2_dep = b.dependency("pcre2", .{ .target = target, .optimize = optimize });
    const og_mlx = b.dependency("mlx", .{
        .target = target,
        .optimize = optimize,
    });
    const fmt_dep = b.dependency("fmt", .{
        .target = target,
        .optimize = optimize,
    });
    const mlx_lib = b.addStaticLibrary(.{
        .name = "mlx",
        .target = target,
        .optimize = optimize,
    });
    mlx_lib.addIncludePath(fmt_dep.path("include"));
    mlx_lib.defineCMacro("FMT_HEADER_ONLY", "1");
    mlx_lib.addIncludePath(og_mlx.path("."));
    mlx_lib.linkLibCpp();
    mlx_lib.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &core_sources, .flags = &CPP_FLAGS });
    if (options.build_cpu) {
        mlx_lib.addCSourceFiles(.{
            .root = og_mlx.path("mlx"),
            .files = &common_sources,
            .flags = &CPP_FLAGS,
        });

        mlx_lib.addCSourceFile(.{
            .file = og_mlx.path("mlx/backend/common/compiled_cpu.cpp"),
            .flags = &CPP_FLAGS,
        });

        mlx_lib.linkFramework("Accelerate");
        mlx_lib.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &accelerate_sources, .flags = &CPP_FLAGS });
    }
    if (options.build_metal) {
        @panic("-Dbuild-metal=true is not supported");
    } else {
        mlx_lib.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &no_metal_sources, .flags = &CPP_FLAGS });
    }
    if (options.build_safetensors) {
        const json_dep = b.dependency("json", .{
            .target = target,
            .optimize = optimize,
        });
        mlx_lib.addIncludePath(json_dep.path("single_include/nlohmann"));
        mlx_lib.addCSourceFile(.{
            .file = og_mlx.path("mlx/io/safetensors.cpp"),
            .flags = &CPP_FLAGS,
        });
    } else {
        mlx_lib.addCSourceFile(.{
            .file = og_mlx.path("mlx/io/no_safetensors.cpp"),
            .flags = &CPP_FLAGS,
        });
    }
    if (options.build_gguf) {
        const gguflib_dep = b.dependency("gguflib", .{
            .target = target,
            .optimize = optimize,
        });
        mlx_lib.addIncludePath(gguflib_dep.path("."));
        const gguflib_lib = b.addStaticLibrary(.{
            .name = "gguflib",
            .target = target,
            .optimize = optimize,
        });
        const gguflib_sources = [_][]const u8{
            "fp16.c",
            "gguflib.c",
        };
        gguflib_lib.addCSourceFiles(.{
            .root = gguflib_dep.path("."),
            .files = &gguflib_sources,
            .flags = &C_FLAGS,
        });
        mlx_lib.linkLibrary(gguflib_lib);
        const gguf_sources = [_][]const u8{
            "io/gguf.cpp",
            "io/gguf_quants.cpp",
        };
        mlx_lib.addCSourceFiles(.{
            .root = og_mlx.path("mlx"),
            .files = &gguf_sources,
            .flags = &C_FLAGS,
        });
    } else {
        mlx_lib.addCSourceFile(.{
            .file = og_mlx.path("mlx/io/no_gguf.cpp"),
            .flags = &CPP_FLAGS,
        });
    }
    const og_mlx_c = b.dependency("mlx-c", .{
        .target = target,
        .optimize = optimize,
    });
    const mlx_c_lib = b.addStaticLibrary(.{
        .name = "mlx-c",
        .target = target,
        .optimize = optimize,
    });
    mlx_c_lib.addIncludePath(og_mlx_c.path("."));
    mlx_c_lib.addIncludePath(og_mlx.path("."));
    mlx_c_lib.addCSourceFiles(.{
        .root = og_mlx_c.path("mlx/c"),
        .files = &mlx_c_sources,
        .flags = &MLXC_CPP_FLAGS,
    });
    mlx_c_lib.linkLibCpp();
    mlx_c_lib.linkLibrary(mlx_lib);
    const exe = b.addExecutable(.{
        .name = "mlx_zig_exe",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.step.dependOn(&mlx_c_lib.step);
    exe.linkLibrary(mlx_c_lib);
    exe.linkLibrary(pcre2_dep.artifact("pcre2-8"));
    exe.addIncludePath(og_mlx_c.path("."));
    b.installArtifact(exe);
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(&exe.step);
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
    const test_exe = b.addTest(.{
        .name = "mlx_zig_test",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_exe.step.dependOn(&mlx_c_lib.step);
    test_exe.linkLibrary(mlx_c_lib);
    test_exe.linkLibrary(pcre2_dep.artifact("pcre2-8"));
    test_exe.addIncludePath(og_mlx_c.path("."));
    b.installArtifact(test_exe);
    const run_tests = b.addRunArtifact(test_exe);
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_tests.step);
}
