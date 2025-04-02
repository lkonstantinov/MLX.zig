const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = std.builtin.OptimizeMode.ReleaseFast;
    const mlx_c_path = b.pathJoin(&.{ b.cache_root.path.?, "mlx-c" });
    const mlx_c_build_path = b.pathJoin(&.{ mlx_c_path, "build" });
    const mlx_c_lib_path = b.pathJoin(&.{ mlx_c_build_path, "libmlxc.a" });
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
    const pcre2_dep = b.dependency("pcre2", .{
        .target = target,
        .optimize = optimize,
    });
    const whisper_exe = b.addExecutable(.{
        .name = "whisper",
        .root_source_file = b.path("src/whisper_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const llama_exe = b.addExecutable(.{
        .name = "llama",
        .root_source_file = b.path("src/llama_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const phi_exe = b.addExecutable(.{
        .name = "phi",
        .root_source_file = b.path("src/phi_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const main_exe = if (doesFileExist("src/main.zig")) b.addExecutable(.{
        .name = "main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    }) else null;
    configureExecutable(whisper_exe, b, mlx_c_path, mlx_c_build_path, pcre2_dep, install_step);
    configureExecutable(llama_exe, b, mlx_c_path, mlx_c_build_path, pcre2_dep, install_step);
    configureExecutable(phi_exe, b, mlx_c_path, mlx_c_build_path, pcre2_dep, install_step);
    if (main_exe) |exe| {
        configureExecutable(exe, b, mlx_c_path, mlx_c_build_path, pcre2_dep, install_step);
    }
    b.installArtifact(whisper_exe);
    b.installArtifact(llama_exe);
    b.installArtifact(phi_exe);
    if (main_exe) |exe| {
        b.installArtifact(exe);
    }
    const whisper_run = b.addRunArtifact(whisper_exe);
    if (b.args) |args| {
        whisper_run.addArgs(args);
    }
    const run_whisper = b.step("run-whisper", "Run the whisper transcription app");
    run_whisper.dependOn(&whisper_run.step);
    const llama_run = b.addRunArtifact(llama_exe);
    const run_llama = b.step("run-llama", "Run the llama chat app");
    run_llama.dependOn(&llama_run.step);
    const phi_run = b.addRunArtifact(phi_exe);
    const run_phi = b.step("run-phi", "Run the phi demo");
    run_phi.dependOn(&phi_run.step);
    if (main_exe) |exe| {
        const main_run = b.addRunArtifact(exe);
        const run_main = b.step("run", "Run the main app");
        run_main.dependOn(&main_run.step);
    }
    if (doesFileExist(b.pathJoin(&.{ mlx_c_build_path, "_deps/mlx-build/mlx.metallib" }))) {
        const dest_dir = b.pathJoin(&.{ b.install_path, "lib", "metal" });
        const mkdir_cmd = b.addSystemCommand(&.{ "mkdir", "-p", dest_dir });
        const copy_cmd = b.addSystemCommand(&.{
            "cp",
            b.pathJoin(&.{ mlx_c_build_path, "_deps/mlx-build/mlx.metallib" }),
            b.pathJoin(&.{ dest_dir, "mlx.metallib" }),
        });
        copy_cmd.step.dependOn(&mkdir_cmd.step);
        b.getInstallStep().dependOn(&copy_cmd.step);
    }
    const help_step = b.step("help", "Explains how to use the build system");
    const help_msg = b.addSystemCommand(&[_][]const u8{
        "echo",
        \\
        \\Zig Build System Usage:
        \\
        \\  zig build                           - Builds all executables and installs to zig-out/bin
        \\  zig build run-whisper -- [file]     - Builds and runs whisper with optional audio file
        \\  zig build run-phi                   - Builds and runs phi demo
        \\  zig build run-llama                 - Builds and runs llama chat
        \\  zig build run                       - Builds and runs main.zig (for development/testing)
        \\
        \\Direct executable usage:
        \\
        \\  zig-out/bin/whisper [audio_file]    - Run whisper with optional audio file
        \\  zig-out/bin/phi [prompt]            - Run phi with optional user prompt
        \\  zig-out/bin/llama                   - Run llama chat
        \\  zig-out/bin/main                    - Run main app with build instructions
        \\
    });
    help_step.dependOn(&help_msg.step);
    const test_step = b.step("test", "Run all tests");
    const main_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    configureExecutable(main_tests, b, mlx_c_path, mlx_c_build_path, pcre2_dep, install_step);
    const run_main_tests = b.addRunArtifact(main_tests);
    test_step.dependOn(&run_main_tests.step);
}

fn configureExecutable(
    exe: *std.Build.Step.Compile,
    b: *std.Build,
    mlx_c_path: []const u8,
    mlx_c_build_path: []const u8,
    pcre2_dep: *std.Build.Dependency,
    install_step: *std.Build.Step,
) void {
    exe.step.dependOn(install_step);
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
