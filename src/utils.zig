//! utils.zig - Utility Functions
//!
//! General utility functions for file operations, model downloading, and JSON
//! parsing.
//!
//! Copyright 2025 Joe

const std = @import("std");

pub fn download(allocator: std.mem.Allocator, model_name: []const u8) !void {
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
            const url_path = try std.fmt.allocPrint(allocator, "https://huggingface.co/mlx-community/{s}/resolve/main/{s}", .{ model_name, filename });
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

pub fn loadJson(comptime T: type, allocator: std.mem.Allocator, filename: []const u8, verbose: bool) !std.json.Parsed(T) {
    const json_string = try std.fs.cwd().readFileAlloc(allocator, filename, 10 * 1024 * 1024);
    defer allocator.free(json_string);
    const parsed = try std.json.parseFromSlice(T, allocator, json_string, .{
        .ignore_unknown_fields = true,
    });
    if (verbose) {
        try printFieldDifferences(T, allocator, json_string);
        try printParsedValue(T, parsed.value, allocator);
    }
    return parsed;
}

fn printParsedValue(comptime T: type, value: T, allocator: std.mem.Allocator) !void {
    var string = std.ArrayList(u8).init(allocator);
    defer string.deinit();
    try std.json.stringify(value, .{ .whitespace = .indent_2 }, string.writer());
    std.debug.print("\nParsed Value:\n", .{});
    std.debug.print("{s}\n", .{string.items});
}

fn printFieldDifferences(comptime T: type, allocator: std.mem.Allocator, json_string: []const u8) !void {
    const struct_info = @typeInfo(T).Struct;
    var generic = try std.json.parseFromSlice(std.json.Value, allocator, json_string, .{});
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

fn fileExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}
