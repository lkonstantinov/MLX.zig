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
