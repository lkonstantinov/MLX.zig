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

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = std.builtin.OptimizeMode.ReleaseFast;
    const options = try MLXBuildOptions.fromOptions(b);
    const version = "0.0.0";
    const mlx_lib = try buildMlx(b, target, optimize, options);
    const mlx_c_lib = try buildMlxC(b, mlx_lib, target, optimize);
    b.installArtifact(mlx_lib);
    b.installArtifact(mlx_c_lib);
    createPkgConfig(b, version);
    const whisper_exe = buildExecutable(b, mlx_c_lib, target, optimize, "mlx_whisper", "src/whisper_main.zig");
    b.installArtifact(whisper_exe);
    const whisper_run_cmd = b.addRunArtifact(whisper_exe);
    const whisper_run_step = b.step("run-whisper", "Run the whisper transcription app");
    whisper_run_step.dependOn(&whisper_run_cmd.step);
    const llama_exe = buildExecutable(b, mlx_c_lib, target, optimize, "mlx_llama", "src/llama_main.zig");
    b.installArtifact(llama_exe);
    const llama_run_cmd = b.addRunArtifact(llama_exe);
    const llama_run_step = b.step("run-llama", "Run the llama chat app");
    llama_run_step.dependOn(&llama_run_cmd.step);
    const default_exe = buildExecutable(b, mlx_c_lib, target, optimize, "mlx_zig_exe", "src/main.zig");
    b.installArtifact(default_exe);
    const default_run_cmd = b.addRunArtifact(default_exe);
    const default_run_step = b.step("run", "Run the default app");
    default_run_step.dependOn(&default_run_cmd.step);
    const test_exe = buildTestExecutable(b, mlx_c_lib, target, optimize);
    const test_run = b.addRunArtifact(test_exe);
    b.installArtifact(test_exe);
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&test_run.step);
    if (options.create_package) {
        createDistributablePackage(b);
    }
    const libs_step = b.step("libs", "Build MLX and MLX-C libraries");
    libs_step.dependOn(&mlx_lib.step);
    libs_step.dependOn(&mlx_c_lib.step);
}

fn buildExecutable(b: *std.Build, mlx_c_lib: *std.Build.Step.Compile, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, name: []const u8, source_path: []const u8) *std.Build.Step.Compile {
    const pcre2_dep = b.dependency("pcre2", .{
        .target = target,
        .optimize = optimize,
    });
    const og_mlx_c = b.dependency("mlx-c", .{
        .target = target,
        .optimize = optimize,
    });
    const exe = b.addExecutable(.{
        .name = name,
        .root_source_file = b.path(source_path),
        .target = target,
        .optimize = optimize,
    });
    exe.linkLibrary(mlx_c_lib);
    exe.linkLibrary(pcre2_dep.artifact("pcre2-8"));
    exe.addIncludePath(og_mlx_c.path("."));
    return exe;
}

fn buildTestExecutable(b: *std.Build, mlx_c_lib: *std.Build.Step.Compile, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const pcre2_dep = b.dependency("pcre2", .{
        .target = target,
        .optimize = optimize,
    });
    const og_mlx_c = b.dependency("mlx-c", .{
        .target = target,
        .optimize = optimize,
    });
    const test_exe = b.addTest(.{
        .name = "mlx_zig_test",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_exe.linkLibrary(mlx_c_lib);
    test_exe.linkLibrary(pcre2_dep.artifact("pcre2-8"));
    test_exe.addIncludePath(og_mlx_c.path("."));
    return test_exe;
}

const MLXBuildOptions = struct {
    build_metal: bool,
    metal_jit: bool,
    build_cpu: bool,
    build_safetensors: bool,
    build_gguf: bool,
    create_package: bool,
    metal_output_path: []const u8,
    fn fromOptions(b: *std.Build) !MLXBuildOptions {
        const default_rel_dir = "lib/metal";
        const default_path = try std.fmt.allocPrint(b.allocator, "{s}/{s}", .{ b.install_prefix, default_rel_dir });
        return .{
            .build_metal = b.option(bool, "build-metal", "Build metal backend") orelse false,
            .metal_jit = b.option(bool, "metal-jit", "Use JIT compilation for Metal kernels") orelse false,
            .build_cpu = b.option(bool, "build-cpu", "Build CPU backend") orelse true,
            .build_safetensors = b.option(bool, "build-safetensors", "Include support for safetensors format") orelse true,
            .build_gguf = b.option(bool, "build-gguf", "Include support for GGUF format") orelse false,
            .create_package = b.option(bool, "create-package", "Create distributable package") orelse false,
            .metal_output_path = b.option([]const u8, "metal-output-path", "Path to the metallib") orelse default_path,
        };
    }
};

fn buildMlx(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, options: MLXBuildOptions) !*std.Build.Step.Compile {
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
        const metal_cpp_dep = b.dependency("metal-cpp", .{
            .target = target,
            .optimize = optimize,
        });
        mlx_lib.addIncludePath(metal_cpp_dep.path("."));
        mlx_lib.installHeadersDirectory(metal_cpp_dep.path("."), ".", .{ .include_extensions = &.{".hpp"} });
        mlx_lib.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &metal_sources, .flags = &CPP_FLAGS });
        const formatted_metal_output_path = try std.fmt.allocPrint(b.allocator, "\"{s}/mlx.metallib\"", .{options.metal_output_path});
        mlx_lib.defineCMacro("METAL_PATH", formatted_metal_output_path);
        try buildAllKernels(b, mlx_lib, og_mlx, options);
        mlx_lib.linkFramework("Metal");
        mlx_lib.linkFramework("Foundation");
        mlx_lib.linkFramework("QuartzCore");
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
        b.installArtifact(gguflib_lib);
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
    mlx_lib.installHeadersDirectory(og_mlx.path("."), ".", .{});
    return mlx_lib;
}

fn buildMlxC(b: *std.Build, mlx_lib: *std.Build.Step.Compile, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) !*std.Build.Step.Compile {
    const og_mlx = b.dependency("mlx", .{
        .target = target,
        .optimize = optimize,
    });
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
    mlx_c_lib.installHeadersDirectory(og_mlx_c.path("."), ".", .{});
    return mlx_c_lib;
}

fn createDistributablePackage(b: *std.Build) void {
    const pkg_dir = b.pathJoin(&.{ b.install_path, "mlx-c-dist" });
    const mkdir_cmd = b.addSystemCommand(&[_][]const u8{
        "mkdir", "-p", pkg_dir,
    });
    const cp_libs_cmd = b.addSystemCommand(&[_][]const u8{
        "cp",                                    "-r",
        b.pathJoin(&.{ b.install_path, "lib" }), pkg_dir,
    });
    cp_libs_cmd.step.dependOn(&mkdir_cmd.step);
    const cp_headers_cmd = b.addSystemCommand(&[_][]const u8{
        "cp",                                        "-r",
        b.pathJoin(&.{ b.install_path, "include" }), pkg_dir,
    });
    cp_headers_cmd.step.dependOn(&mkdir_cmd.step);
    const tar_cmd = b.addSystemCommand(&[_][]const u8{
        "tar",                                                "czf",
        b.pathJoin(&.{ b.install_path, "mlx-c-lib.tar.gz" }), "-C",
        b.pathJoin(&.{b.install_path}),                       "mlx-c-dist",
    });
    tar_cmd.step.dependOn(&cp_libs_cmd.step);
    tar_cmd.step.dependOn(&cp_headers_cmd.step);
    const pkg_step = b.step("package", "Create distributable package");
    pkg_step.dependOn(&tar_cmd.step);
}

fn createPkgConfig(b: *std.Build, version: []const u8) void {
    const pc_content = b.fmt(
        \\prefix={s}
        \\exec_prefix=${{prefix}}
        \\libdir=${{prefix}}/lib
        \\includedir=${{prefix}}/include
        \\
        \\Name: mlx-c
        \\Description: C API for MLX machine learning library
        \\Version: {s}
        \\Libs: -L${{libdir}} -lmlx-c -lmlx
        \\Cflags: -I${{includedir}}
        \\
    , .{ b.install_prefix, version });
    const pc_path = b.pathJoin(&.{ b.install_path, "lib", "pkgconfig" });
    const mkdir_cmd = b.addSystemCommand(&[_][]const u8{
        "mkdir", "-p", pc_path,
    });
    const pc_file = b.pathJoin(&.{ pc_path, "mlx-c.pc" });
    const write_cmd = b.addWriteFile(pc_file, pc_content);
    write_cmd.step.dependOn(&mkdir_cmd.step);
    b.getInstallStep().dependOn(&write_cmd.step);
}

fn getVersionIncludes(metal_version: u32) []const u8 {
    return if (metal_version >= 310)
        "mlx/backend/metal/kernels/metal_3_1"
    else
        "mlx/backend/metal/kernels/metal_3_0";
}

fn buildAllKernels(b: *std.Build, lib: *std.Build.Step.Compile, og_mlx: *std.Build.Dependency, options: MLXBuildOptions) !void {
    var airFiles = std.ArrayList(std.Build.LazyPath).init(b.allocator);
    defer airFiles.deinit();
    inline for (default_kernels) |kernel| {
        try airFiles.append(try buildKernel(b, kernel, og_mlx));
    }
    try buildMetallib(b, lib, airFiles.items, options);
}

fn buildKernel(b: *std.Build, comptime rel_path: []const u8, og_mlx: *std.Build.Dependency) !std.Build.LazyPath {
    const name = comptime (if (std.mem.lastIndexOf(u8, rel_path, "/")) |last_slash|
        rel_path[(last_slash + 1)..]
    else
        rel_path);
    var metal_flags = std.ArrayList([]const u8).init(b.allocator);
    defer metal_flags.deinit();
    try metal_flags.appendSlice(&[_][]const u8{
        "-Wall",
        "-Wextra",
        "-fno-fast-math",
    });
    const version_include = getVersionIncludes(310);
    try metal_flags.appendSlice(&[_][]const u8{
        "-I",
        og_mlx.path(version_include).getPath(b),
    });
    try metal_flags.appendSlice(&[_][]const u8{
        "-I",
        og_mlx.path(".").getPath(b),
    });
    const source_path = "mlx/backend/metal/kernels/" ++ rel_path ++ ".metal";
    const source_path_lazy = og_mlx.path(source_path);
    const metal_cmd = b.addSystemCommand(&[_][]const u8{
        "xcrun",
        "-sdk",
        "macosx",
        "metal",
    });
    metal_cmd.addArgs(metal_flags.items);
    metal_cmd.addArg("-c");
    metal_cmd.addArg(source_path_lazy.getPath(b));
    metal_cmd.addArg("-o");
    const out_file_name = name ++ ".air";
    const output_path = metal_cmd.addOutputFileArg(out_file_name);
    return output_path;
}

fn buildMetallib(b: *std.Build, lib: *std.Build.Step.Compile, air_files: []std.Build.LazyPath, options: MLXBuildOptions) !void {
    const metallib_cmd = b.addSystemCommand(&[_][]const u8{
        "xcrun",
        "-sdk",
        "macosx",
        "metallib",
    });
    for (air_files) |air| {
        metallib_cmd.addFileArg(air);
    }
    metallib_cmd.addArg("-o");
    const metallib_file = metallib_cmd.addOutputFileArg("mlx.metallib");
    const copy_step = CopyMetalLibStep.create(b, metallib_file, options.metal_output_path);
    copy_step.step.dependOn(&metallib_cmd.step);
    lib.step.dependOn(&copy_step.step);
}

const CopyMetalLibStep = struct {
    const Self = @This();
    step: std.Build.Step,
    b: *std.Build,
    metallib_file: std.Build.LazyPath,
    metal_output_path: []const u8,
    pub fn create(b: *std.Build, metallib_file: std.Build.LazyPath, metal_output_path: []const u8) *CopyMetalLibStep {
        const new = b.allocator.create(Self) catch @panic("OOM");
        new.* = .{
            .b = b,
            .metallib_file = metallib_file,
            .metal_output_path = metal_output_path,
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = "copy_mlx_metallib",
                .owner = b,
                .makeFn = make,
            }),
        };
        return new;
    }
    fn make(step: *std.Build.Step, prog_node: std.Progress.Node) anyerror!void {
        _ = prog_node;
        const self: *Self = @fieldParentPtr("step", step);
        const src_path = self.metallib_file.getPath(self.b);
        const dest_file = try std.fmt.allocPrint(self.b.allocator, "{s}/mlx.metallib", .{self.metal_output_path});
        var fs = std.fs.cwd();
        var dest_dir = fs.makeOpenPath(self.metal_output_path, .{}) catch |err| {
            return err;
        };
        defer dest_dir.close();
        try std.fs.copyFileAbsolute(src_path, dest_file, .{});
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

const metal_sources = [_][]const u8{
    "backend/metal/allocator.cpp",
    "backend/metal/binary.cpp",
    "backend/metal/conv.cpp",
    "backend/metal/compiled.cpp",
    "backend/metal/copy.cpp",
    "backend/metal/custom_kernel.cpp",
    "backend/metal/distributed.cpp",
    "backend/metal/device.cpp",
    "backend/metal/event.cpp",
    "backend/metal/fft.cpp",
    "backend/metal/hadamard.cpp",
    "backend/metal/indexing.cpp",
    "backend/metal/matmul.cpp",
    "backend/metal/scaled_dot_product_attention.cpp",
    "backend/metal/metal.cpp",
    "backend/metal/primitives.cpp",
    "backend/metal/quantized.cpp",
    "backend/metal/normalization.cpp",
    "backend/metal/rope.cpp",
    "backend/metal/scan.cpp",
    "backend/metal/slicing.cpp",
    "backend/metal/softmax.cpp",
    "backend/metal/sort.cpp",
    "backend/metal/reduce.cpp",
    "backend/metal/ternary.cpp",
    "backend/metal/unary.cpp",
    "backend/metal/resident.cpp",
    "backend/metal/utils.cpp",
};

const default_kernels = [_][]const u8{
    "arg_reduce",
    "conv",
    "gemv",
    "layer_norm",
    "random",
    "rms_norm",
    "rope",
    "scaled_dot_product_attention",
    "steel/attn/kernels/steel_attention",
};
