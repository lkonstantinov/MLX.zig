//! utils.zig - Utility Functions
//!
//! General utility functions for file operations, model downloading, and JSON
//! parsing.
//!
//! Copyright 2025 Joe

const std = @import("std");

pub fn download(allocator: std.mem.Allocator, repo_name: []const u8, model_name: []const u8) !void {
    const mkdir = struct {
        fn mkdir(dir: []const u8) !void {
            std.fs.cwd().makeDir(dir) catch |err| {
                if (err == error.PathAlreadyExists) {
                    std.debug.print("Directory '{s}' already exists.\n", .{dir});
                    return;
                } else {
                    return err;
                }
            };
            std.debug.print("Directory '{s}' created successfully.\n", .{dir});
        }
    }.mkdir;
    const fileExists = struct {
        fn fileExists(path: []const u8) bool {
            std.fs.cwd().access(path, .{}) catch return false;
            return true;
        }
    }.fileExists;
    try mkdir(model_name);
    const filenames = [_][]const u8{
        "model.safetensors",
        "config.json",
        "tokenizer.json",
    };
    var args = std.ArrayList([]const u8).init(allocator);
    defer args.deinit();
    try args.append("curl");
    try args.append("--location");
    try args.append("--parallel");
    var paths_to_free = std.ArrayList([]const u8).init(allocator);
    defer {
        for (paths_to_free.items) |path| {
            allocator.free(path);
        }
        paths_to_free.deinit();
    }
    var all_exist = true;
    for (filenames) |filename| {
        const local_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_name, filename });
        try paths_to_free.append(local_path);
        if (fileExists(local_path)) {
            std.debug.print("File '{s}' already exists. Skipping download.\n", .{local_path});
        } else {
            all_exist = false;
            const url_path = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/{s}/resolve/main/{s}", .{ repo_name, model_name, filename });
            try paths_to_free.append(url_path);

            try args.append(url_path);
            try args.append("-o");
            try args.append(local_path);
        }
    }
    if (all_exist) {
        std.debug.print("All files already exist. No download needed.\n", .{});
        return;
    }
    var proc = std.process.Child.init(args.items, allocator);
    try proc.spawn();
    const result = try proc.wait();
    switch (result) {
        .Exited => |code| {
            if (code == 0) {
                std.debug.print("Download successful.\n", .{});
            } else {
                std.debug.print("Download failed with exit code: {d}\n", .{code});
                return error.DownloadFailed;
            }
        },
        else => {
            std.debug.print("Download process terminated abnormally.\n", .{});
            return error.DownloadFailed;
        },
    }
}

pub fn allocJoin(allocator: std.mem.Allocator, parent: []const u8, name: anytype) ![]u8 {
    const T = @TypeOf(name);
    const info = @typeInfo(T);
    const fmt = if (info == .Int or info == .ComptimeInt) ".{d}" else if (T == @TypeOf(null)) "{s}" else ".{s}";
    return std.fmt.allocPrint(allocator, "{s}" ++ fmt, .{ parent, name });
}

pub fn loadAudio(alloc: std.mem.Allocator, file_path: []const u8) ![]f32 {
    const buffer_size = 16384;
    const args = [_][]const u8{
        "ffmpeg",
        "-nostdin",
        "-threads",
        "4",
        "-i",
        file_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-loglevel",
        "error",
        "-",
    };
    var process = std.process.Child.init(&args, alloc);
    process.stdout_behavior = .Pipe;
    process.stderr_behavior = .Pipe;
    try process.spawn();
    var float_samples = std.ArrayList(f32).init(alloc);
    defer float_samples.deinit();
    try float_samples.ensureTotalCapacity(16000 * 10); // Reserve space for ~10s of audio
    const stdout = process.stdout.?.reader();
    var buffer: [buffer_size]u8 = undefined;
    var i16_buffer: [buffer_size / 2]i16 = undefined;
    while (true) {
        const bytes_read = try stdout.read(&buffer);
        if (bytes_read == 0) break; // EOF
        const valid_bytes = bytes_read & ~@as(usize, 1);
        const samples = valid_bytes / 2;
        var i: usize = 0;
        while (i < valid_bytes) : (i += 2) {
            const sample_idx = i / 2;
            const lo = @as(i16, @intCast(buffer[i]));
            const hi = @as(i16, @intCast(buffer[i + 1]));
            i16_buffer[sample_idx] = lo | (hi << 8);
        }
        try float_samples.ensureUnusedCapacity(samples);
        for (i16_buffer[0..samples]) |sample| {
            try float_samples.append(@as(f32, @floatFromInt(sample)) / 32768.0);
        }
    }
    const term = try process.wait();
    if (term != .Exited or term.Exited != 0) {
        const stderr = process.stderr.?.reader();
        var error_msg = std.ArrayList(u8).init(alloc);
        defer error_msg.deinit();
        try stderr.readAllArrayList(&error_msg, 4096);
        std.log.err("Failed to load audio: {s}", .{error_msg.items});
        return error.FfmpegFailed;
    }
    return float_samples.toOwnedSlice();
}

pub fn loadJson(comptime T: type, allocator: std.mem.Allocator, filename: []const u8, verbose: bool) !std.json.Parsed(T) {
    const printVals = struct {
        fn printParsedValue(comptime T_: type, value: T_, allocator_: std.mem.Allocator) !void {
            var string = std.ArrayList(u8).init(allocator_);
            defer string.deinit();
            try std.json.stringify(value, .{ .whitespace = .indent_2 }, string.writer());
            std.debug.print("\nParsed Value:\n", .{});
            std.debug.print("{s}\n", .{string.items});
        }
    }.printParsedValue;
    const printDiff = struct {
        fn printFieldDifferences(comptime T_: type, allocator_: std.mem.Allocator, json_string: []const u8) !void {
            const struct_info = @typeInfo(T_).Struct;
            var generic = try std.json.parseFromSlice(std.json.Value, allocator_, json_string, .{});
            defer generic.deinit();
            if (generic.value != .object) return;
            std.debug.print("\nIgnored fields:\n", .{});
            {
                var found_extra = false;
                var iter = generic.value.object.iterator();
                while (iter.next()) |entry| {
                    const field_exists = blk: {
                        inline for (struct_info.fields) |field| {
                            if (std.mem.eql(u8, field.name, entry.key_ptr.*)) break :blk true;
                        }
                        break :blk false;
                    };
                    if (!field_exists) {
                        found_extra = true;
                        std.debug.print("  - {s}\n", .{entry.key_ptr.*});
                    }
                }
                if (!found_extra) std.debug.print("  None\n", .{});
            }
            std.debug.print("Default fields:\n", .{});
            {
                var found_missing = false;
                inline for (struct_info.fields) |field| {
                    if (!generic.value.object.contains(field.name)) {
                        found_missing = true;
                        std.debug.print("  - {s}\n", .{field.name});
                    }
                }
                if (!found_missing) std.debug.print("  None\n", .{});
            }
        }
    }.printFieldDifferences;
    const json_string = try std.fs.cwd().readFileAlloc(allocator, filename, 10 * 1024 * 1024);
    defer allocator.free(json_string);
    const parsed = try std.json.parseFromSlice(T, allocator, json_string, .{
        .ignore_unknown_fields = true,
    });
    if (verbose) {
        try printDiff(T, allocator, json_string);
        try printVals(T, parsed.value, allocator);
    }
    return parsed;
}
