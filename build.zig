const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = std.builtin.OptimizeMode.ReleaseFast;
    const mlx_c_path = b.pathJoin(&.{ b.cache_root.path.?, "mlx-c" });
    const mlx_c_build_path = b.pathJoin(&.{ mlx_c_path, "build" });
    const mlx_c_lib_path = b.pathJoin(&.{ mlx_c_build_path, "libmlxc.a" });
    const metallib_path = b.pathJoin(&.{ mlx_c_build_path, "_deps/mlx-build/mlx.metallib" });
    const install_step = b.step("install-mlx-c", "Install MLX-C if needed");
    const needs_install = !doesFileExist(mlx_c_lib_path);
    if (needs_install) {
        const clone_cmd = b.addSystemCommand(&[_][]const u8{ "sh", "-c", b.fmt("if [ ! -d {s} ]; then git clone https://github.com/ml-explore/mlx-c.git {s}; fi", .{ mlx_c_path, mlx_c_path }) });
        const mkdir_cmd = b.addSystemCommand(&[_][]const u8{ "mkdir", "-p", mlx_c_build_path });
        mkdir_cmd.step.dependOn(&clone_cmd.step);
        const cmake_cmd = b.addSystemCommand(&[_][]const u8{ "cmake", "..", "-DCMAKE_BUILD_TYPE=Release" });
        cmake_cmd.setCwd(.{ .cwd_relative = mlx_c_build_path });
        cmake_cmd.step.dependOn(&mkdir_cmd.step);
        const make_cmd = b.addSystemCommand(&[_][]const u8{ "make", "-j" });
        make_cmd.setCwd(.{ .cwd_relative = mlx_c_build_path });
        make_cmd.step.dependOn(&cmake_cmd.step);
        install_step.dependOn(&make_cmd.step);
    }
    b.getInstallStep().dependOn(install_step);
    const pcre2_dep = b.dependency("pcre2", .{
        .target = target,
        .optimize = optimize,
    });
    const whisper_exe = buildExe(b, "mlx_whisper", "src/whisper_main.zig", mlx_c_path, mlx_c_build_path, pcre2_dep, target, optimize, install_step);
    const llama_exe = buildExe(b, "mlx_llama", "src/llama_main.zig", mlx_c_path, mlx_c_build_path, pcre2_dep, target, optimize, install_step);
    const default_exe = buildExe(b, "mlx_zig_exe", "src/main.zig", mlx_c_path, mlx_c_build_path, pcre2_dep, target, optimize, install_step);
    const whisper_run = b.addRunArtifact(whisper_exe);
    const whisper_step = b.step("run-whisper", "Run whisper transcription");
    whisper_step.dependOn(&whisper_run.step);
    const llama_run = b.addRunArtifact(llama_exe);
    const llama_step = b.step("run-llama", "Run llama chat");
    llama_step.dependOn(&llama_run.step);
    const default_run = b.addRunArtifact(default_exe);
    const default_step = b.step("run", "Run default app");
    default_step.dependOn(&default_run.step);
    const test_exe = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_exe.step.dependOn(install_step);
    configureExecutable(test_exe, b, mlx_c_path, mlx_c_build_path, pcre2_dep);
    const test_run = b.addRunArtifact(test_exe);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&test_run.step);
    const dest_dir = b.pathJoin(&.{ b.install_path, "lib", "metal" });
    const mkdir_cmd = b.addSystemCommand(&.{ "mkdir", "-p", dest_dir });
    const dest_path = b.pathJoin(&.{ dest_dir, "mlx.metallib" });
    const copy_cmd = b.addSystemCommand(&.{ "cp", metallib_path, dest_path });
    copy_cmd.step.dependOn(&mkdir_cmd.step);
    copy_cmd.step.dependOn(install_step);
    b.getInstallStep().dependOn(&copy_cmd.step);
}

fn buildExe(
    b: *std.Build,
    name: []const u8,
    source_path: []const u8,
    mlx_c_path: []const u8,
    mlx_c_build_path: []const u8,
    pcre2_dep: *std.Build.Dependency,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    install_step: *std.Build.Step,
) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = name,
        .root_source_file = b.path(source_path),
        .target = target,
        .optimize = optimize,
    });
    exe.step.dependOn(install_step);
    configureExecutable(exe, b, mlx_c_path, mlx_c_build_path, pcre2_dep);
    b.installArtifact(exe);
    return exe;
}

fn configureExecutable(
    exe: *std.Build.Step.Compile,
    b: *std.Build,
    mlx_c_path: []const u8,
    mlx_c_build_path: []const u8,
    pcre2_dep: *std.Build.Dependency,
) void {
    exe.addIncludePath(.{ .cwd_relative = mlx_c_path });
    exe.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ mlx_c_build_path, "libmlxc.a" }) });
    exe.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ mlx_c_build_path, "_deps/mlx-build/libmlx.a" }) });
    exe.linkFramework("Metal");
    exe.linkFramework("Foundation");
    exe.linkFramework("QuartzCore");
    exe.linkFramework("Accelerate");
    exe.linkLibCpp();
    exe.linkLibrary(pcre2_dep.artifact("pcre2-8"));
}

fn doesFileExist(path: []const u8) bool {
    var file = std.fs.cwd().openFile(path, .{}) catch |err| {
        if (err == error.FileNotFound) return false;
        return false;
    };
    file.close();
    return true;
}
