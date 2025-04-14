//! utils.zig - Utility Functions
//!
//! Copyright 2025 Joe

const std = @import("std");

pub fn downloadModel(allocator_: std.mem.Allocator, repo_name_: []const u8, model_name_: []const u8) !void {
    const getModelFiles = struct {
        fn getModelFiles(allocator: std.mem.Allocator, repo_name: []const u8, model_name: []const u8) !?[]const []const u8 {
            const index_path = try std.fmt.allocPrint(allocator, "{s}/model.safetensors.index.json", .{model_name});
            defer allocator.free(index_path);
            if (!fileExists(index_path)) {
                download(allocator, repo_name, model_name, &[_][]const u8{"model.safetensors.index.json"}) catch return null;
                if (!fileExists(index_path)) return null;
            }
            const index_content = std.fs.cwd().readFileAlloc(allocator, index_path, 10 * 1024 * 1024) catch return null;
            defer allocator.free(index_content);
            var parsed = std.json.parseFromSlice(std.json.Value, allocator, index_content, .{}) catch return null;
            defer parsed.deinit();
            const weight_map = parsed.value.object.get("weight_map") orelse return null;
            if (weight_map != .object or weight_map.object.count() == 0) return null;
            var iter = weight_map.object.iterator();
            const first_entry = iter.next() orelse return null;
            if (first_entry.value_ptr.* != .string) return null;
            const pattern = first_entry.value_ptr.*.string;
            var files = std.ArrayList([]const u8).init(allocator);
            errdefer {
                for (files.items) |file| allocator.free(file);
                files.deinit();
            }
            try files.append(try allocator.dupe(u8, "config.json"));
            try files.append(try allocator.dupe(u8, "tokenizer.json"));
            if (std.mem.indexOf(u8, pattern, "-of-")) |idx| {
                const total_part = pattern[idx + 4 ..];
                const dash_idx = std.mem.indexOf(u8, total_part, "-") orelse
                    std.mem.indexOf(u8, total_part, ".") orelse return null;
                const count_str = total_part[0..dash_idx];
                const count = std.fmt.parseInt(usize, count_str, 10) catch return null;
                const base = pattern[0..std.mem.indexOf(u8, pattern, "-00").?];
                const ext = pattern[std.mem.lastIndexOf(u8, pattern, ".").?..];
                for (1..count + 1) |i| {
                    const filename = try std.fmt.allocPrint(allocator, "{s}-{:0>5}-of-{:0>5}{s}", .{ base, i, count, ext });
                    try files.append(filename);
                }
            } else {
                try files.append(try allocator.dupe(u8, pattern));
            }
            const result = try files.toOwnedSlice();
            return result;
        }
    }.getModelFiles;
    const custom_files = try getModelFiles(allocator_, repo_name_, model_name_);
    defer if (custom_files) |files| {
        for (files) |file| allocator_.free(file);
        allocator_.free(files);
    };
    return download(allocator_, repo_name_, model_name_, custom_files);
}

pub fn download(allocator: std.mem.Allocator, repo_name: []const u8, model_name: []const u8, file_names: ?[]const []const u8) !void {
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
    try mkdir(model_name);
    const default_filenames = [_][]const u8{
        "model.safetensors",
        "config.json",
        "tokenizer.json",
    };
    const filenames = if (file_names) |f| f else &default_filenames;
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

pub fn formatDynamic(
    allocator: std.mem.Allocator,
    chat_fmt: []const u8,
    replacements: []const []const u8,
) ![]const u8 {
    var segments = std.ArrayList([]const u8).init(allocator);
    defer segments.deinit();
    var splitter = std.mem.splitSequence(u8, chat_fmt, "{s}");
    while (splitter.next()) |segment| {
        try segments.append(segment);
    }
    const expected_replacements = segments.items.len - 1;
    if (replacements.len != expected_replacements) {
        return error.ReplacementCountMismatch;
    }
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();
    for (segments.items[0 .. segments.items.len - 1], replacements) |segment, replacement| {
        try result.appendSlice(segment);
        try result.appendSlice(replacement);
    }
    try result.appendSlice(segments.items[segments.items.len - 1]);
    return try result.toOwnedSlice();
}

pub fn formatRange(comptime format: []const u8, comptime start: usize, comptime end: usize) [end - start][]const u8 {
    var result: [end - start][]const u8 = undefined;
    inline for (&result, 0..) |*ptr, i| {
        ptr.* = std.fmt.comptimePrint(format, .{start + i});
    }
    return result;
}

pub fn formatRangeFloat(comptime count: usize) [count][]const u8 {
    var result: [count][]const u8 = undefined;
    inline for (&result, 0..) |*ptr, i| {
        const seconds = @as(f32, @floatFromInt(i)) * 0.02;
        const whole = @as(u32, @intFromFloat(seconds));
        const frac = @as(u32, @intFromFloat((seconds - @as(f32, @floatFromInt(whole))) * 100.0 + 0.5));
        if (frac < 10) {
            ptr.* = std.fmt.comptimePrint("<|{d}.0{d}|>", .{ whole, frac });
        } else {
            ptr.* = std.fmt.comptimePrint("<|{d}.{d}|>", .{ whole, frac });
        }
    }
    return result;
}

pub fn loadAudio(allocator: std.mem.Allocator, audio_file: []const u8) ![]f32 {
    const file_path = try std.fs.cwd().realpathAlloc(allocator, audio_file);
    defer allocator.free(file_path);
    std.debug.print("\nAUDIO: {s}\n", .{file_path});
    std.fs.cwd().access(file_path, .{}) catch return error.AudioFileNotFound;
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
    var process = std.process.Child.init(&args, allocator);
    process.stdout_behavior = .Pipe;
    process.stderr_behavior = .Pipe;
    try process.spawn();
    var float_samples = std.ArrayList(f32).init(allocator);
    defer float_samples.deinit();
    try float_samples.ensureTotalCapacity(16000 * 10);
    const stdout = process.stdout.?.reader();
    var buffer: [buffer_size]u8 = undefined;
    var i16_buffer: [buffer_size / 2]i16 = undefined;
    while (true) {
        const bytes_read = try stdout.read(&buffer);
        if (bytes_read == 0) break;
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
        var error_msg = std.ArrayList(u8).init(allocator);
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

pub fn loadConfigJson(comptime T: type, allocator: std.mem.Allocator, model_path: []const u8, verbose: bool) !std.json.Parsed(T) {
    var buf: [1024]u8 = undefined;
    const path_config = try std.fmt.bufPrintZ(&buf, "{s}/config.json", .{model_path});
    var parsed = try loadJson(T, allocator, path_config, verbose);
    if (@hasField(T, "eos_token_id")) {
        if (@TypeOf(parsed.value.eos_token_id) == u32) {
            var eos_ids = try allocator.alloc(u32, 1);
            eos_ids[0] = parsed.value.eos_token_id;
            parsed.value.eos_token_ids = eos_ids;
        } else {
            parsed.value.eos_token_ids = try allocator.dupe(u32, parsed.value.eos_token_id);
        }
    }
    return parsed;
}

fn fileExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}
