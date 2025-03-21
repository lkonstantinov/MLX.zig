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
const Regex = @import("regex.zig").Regex;

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    pattern_regex: Regex,
    special_regex: ?Regex,
    vocab: std.StringHashMap(u32),
    id_to_token: std.AutoHashMap(u32, []const u8),
    specials: []const []const u8,

    pub fn init(allocator: std.mem.Allocator, path_json: []const u8) !Tokenizer {
        const json_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{path_json});
        defer allocator.free(json_path);
        const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
        defer allocator.free(json_content);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
        defer parsed.deinit();
        const pattern = parsed.value.object.get("pre_tokenizer").?.object.get("pretokenizers").?.array.items[0].object.get("pattern").?.object.get("Regex").?.string;
        var pattern_regex = try Regex.init(pattern);
        errdefer pattern_regex.deinit();
        var vocab = std.StringHashMap(u32).init(allocator);
        errdefer vocab.deinit();
        var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
        errdefer id_to_token.deinit();
        const replacements = [_]struct { bytes: [2]u8, replacement: u8 }{
            .{ .bytes = .{ 0xC4, 0xA0 }, .replacement = ' ' },
            .{ .bytes = .{ 0xC4, 0x8A }, .replacement = '\n' },
            .{ .bytes = .{ 0xC4, 0x89 }, .replacement = '\t' },
        };
        var vocab_iter = parsed.value.object.get("model").?.object.get("vocab").?.object.iterator();
        while (vocab_iter.next()) |entry| {
            const token_str = entry.key_ptr.*;
            const id = @as(u32, @intCast(entry.value_ptr.*.integer));
            var buf = std.ArrayList(u8).init(allocator);
            defer buf.deinit();
            var i: usize = 0;
            while (i < token_str.len) {
                var replaced = false;
                for (replacements) |repl| {
                    if (i + 1 < token_str.len and
                        token_str[i] == repl.bytes[0] and
                        token_str[i + 1] == repl.bytes[1])
                    {
                        try buf.append(repl.replacement);
                        i += 2;
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) {
                    try buf.append(token_str[i]);
                    i += 1;
                }
            }
            const processed_token = try buf.toOwnedSlice();
            try vocab.put(processed_token, id);
            try id_to_token.put(id, processed_token);
        }
        var special_tokens = std.ArrayList([]const u8).init(allocator);
        defer special_tokens.deinit();
        const added_tokens = parsed.value.object.get("added_tokens").?.array;
        for (added_tokens.items) |token| {
            if (token != .object) continue;
            const content = token.object.get("content") orelse continue;
            const id = token.object.get("id") orelse continue;
            if (content != .string or id != .integer) continue;
            const token_str = content.string;
            const token_id = @as(u32, @intCast(id.integer));
            const token_copy = try allocator.dupe(u8, token_str);
            try vocab.put(token_copy, token_id);
            try id_to_token.put(token_id, token_copy);
            try special_tokens.append(token_copy);
        }
        const specials = try allocator.dupe([]const u8, special_tokens.items);
        const special_regex = try createSpecialRegex(allocator, specials);
        return Tokenizer{
            .allocator = allocator,
            .pattern_regex = pattern_regex,
            .special_regex = special_regex,
            .vocab = vocab,
            .id_to_token = id_to_token,
            .specials = specials,
        };
    }

    pub fn initFromTikToken(allocator: std.mem.Allocator, pattern: []const u8, vocabulary_path: []const u8, specials: []const []const u8) !Tokenizer {
        var pattern_regex = try Regex.init(pattern);
        errdefer pattern_regex.deinit();
        var vocab = std.StringHashMap(u32).init(allocator);
        errdefer vocab.deinit();
        var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
        errdefer id_to_token.deinit();
        const file = try std.fs.cwd().openFile(vocabulary_path, .{});
        defer file.close();
        var buf_reader = std.io.bufferedReader(file.reader());
        var in_stream = buf_reader.reader();
        var line_buf: [1024]u8 = undefined;
        while (try in_stream.readUntilDelimiterOrEof(&line_buf, '\n')) |line| {
            if (line.len == 0) continue;
            var iter = std.mem.tokenize(u8, line, " ");
            const token_b64 = iter.next() orelse continue;
            const rank_str = iter.next() orelse continue;
            const decoded_size = try std.base64.standard.Decoder.calcSizeForSlice(token_b64);
            const token = try allocator.alloc(u8, decoded_size);
            errdefer allocator.free(token);
            _ = try std.base64.standard.Decoder.decode(token, token_b64);
            const rank = try std.fmt.parseInt(u32, rank_str, 10);
            try vocab.put(token, rank);
            try id_to_token.put(rank, token);
        }
        const special_start = @as(u32, @intCast(vocab.count()));
        for (specials, 0..) |special, i| {
            const token = try allocator.dupe(u8, special);
            const rank = special_start + @as(u32, @intCast(i));
            try vocab.put(token, rank);
            try id_to_token.put(rank, token);
        }
        const special_regex = try createSpecialRegex(allocator, specials);
        return Tokenizer{
            .allocator = allocator,
            .pattern_regex = pattern_regex,
            .special_regex = special_regex,
            .vocab = vocab,
            .id_to_token = id_to_token,
            .specials = specials,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        var it = self.vocab.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.vocab.deinit();
        self.id_to_token.deinit();
        if (self.special_regex) |*regex| {
            regex.deinit();
        }
        self.pattern_regex.deinit();
        self.allocator.destroy(self);
    }

    fn loadVocabulary(self: *Tokenizer, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        var buf_reader = std.io.bufferedReader(file.reader());
        var in_stream = buf_reader.reader();
        var line_buf: [1024]u8 = undefined;
        var line_num: usize = 0;
        while (try in_stream.readUntilDelimiterOrEof(&line_buf, '\n')) |line| {
            if (line.len == 0) continue;
            var iter = std.mem.tokenize(u8, line, " ");
            const token_b64 = iter.next() orelse continue;
            const rank_str = iter.next() orelse continue;
            const decoded_size = try std.base64.standard.Decoder.calcSizeForSlice(token_b64);
            const token = try self.allocator.alloc(u8, decoded_size);
            errdefer self.allocator.free(token);
            _ = try std.base64.standard.Decoder.decode(token, token_b64);
            const rank = try std.fmt.parseInt(u32, rank_str, 10);
            try self.vocab.put(token, rank);
            try self.id_to_token.put(rank, token);
            line_num += 1;
        }
        const special_start = @as(u32, @intCast(self.vocab.count()));
        for (self.specials, 0..) |special, i| {
            const token = try self.allocator.dupe(u8, special);
            errdefer self.allocator.free(token);
            const rank = special_start + @as(u32, @intCast(i));
            try self.vocab.put(token, rank);
            try self.id_to_token.put(rank, token);
        }
    }

    fn createSpecialRegex(allocator: std.mem.Allocator, specials: []const []const u8) !?Regex {
        if (specials.len == 0) return null;
        var pattern = std.ArrayList(u8).init(allocator);
        defer pattern.deinit();
        var first = true;
        for (specials) |special| {
            if (!first) {
                try pattern.append('|');
            } else {
                first = false;
            }
            for (special) |char| {
                if (std.mem.indexOfScalar(u8, "\\^$.|?*+()[{", char) != null) {
                    try pattern.append('\\');
                }
                try pattern.append(char);
            }
        }
        if (pattern.items.len == 0) return null;
        return try Regex.init(pattern.items);
    }

    fn splitWithSpecials(self: *const Tokenizer, text: []const u8) !std.ArrayList([]const u8) {
        var result = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (result.items) |item| {
                self.allocator.free(item);
            }
            result.deinit();
        }
        if (self.special_regex == null) {
            try result.append(try self.allocator.dupe(u8, text));
            return result;
        }
        var last_end: usize = 0;
        var start: usize = 0;
        while (start < text.len) {
            const match_result = try self.special_regex.?.match(text, start);
            if (match_result == null) {
                break;
            }
            const match = match_result.?;
            if (match.start > last_end) {
                const non_match = try self.allocator.dupe(u8, text[last_end..match.start]);
                try result.append(non_match);
            }
            const matched_special = try self.allocator.dupe(u8, text[match.start..match.end]);
            try result.append(matched_special);
            last_end = match.end;
            start = match.end;
        }
        if (last_end < text.len) {
            const remaining = try self.allocator.dupe(u8, text[last_end..]);
            try result.append(remaining);
        }
        return result;
    }

    fn splitWithPattern(self: *const Tokenizer, text: []const u8) !std.ArrayList([]const u8) {
        var result = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (result.items) |item| {
                self.allocator.free(item);
            }
            result.deinit();
        }
        var start: usize = 0;
        while (start < text.len) {
            const match_result = try self.pattern_regex.match(text, start);
            if (match_result == null) {
                // Handle remaining text if no match is found
                if (start < text.len) {
                    const remaining = try self.allocator.dupe(u8, text[start..]);
                    try result.append(remaining);
                }
                break;
            }
            const match = match_result.?;
            const matched_text = try self.allocator.dupe(u8, text[match.start..match.end]);
            try result.append(matched_text);
            start = match.end;
        }
        return result;
    }

    fn bpeMerges(self: *Tokenizer, token: []const u8) !std.ArrayList(u32) {
        var result = std.ArrayList(u32).init(self.allocator);
        errdefer result.deinit();
        if (self.vocab.get(token)) |id| {
            try result.append(id);
            return result;
        }
        if (token.len == 0) {
            return result;
        }
        var boundaries = std.ArrayList(usize).init(self.allocator);
        defer boundaries.deinit();
        try boundaries.append(0);
        for (1..token.len + 1) |i| {
            try boundaries.append(i);
        }
        var did_merge = true;
        while (did_merge and boundaries.items.len > 2) {
            did_merge = false;
            var best_idx: ?usize = null;
            var best_rank: u32 = std.math.maxInt(u32);
            for (0..boundaries.items.len - 2) |i| {
                const start = boundaries.items[i];
                const end = boundaries.items[i + 2];
                const pair = token[start..end];
                if (self.vocab.get(pair)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }
            if (best_idx) |i| {
                did_merge = true;
                _ = boundaries.orderedRemove(i + 1);
            }
        }
        const num_tokens = boundaries.items.len - 1;
        for (0..num_tokens) |i| {
            const start = boundaries.items[i];
            const end = boundaries.items[i + 1];
            const segment = token[start..end];
            if (self.vocab.get(segment)) |id| {
                try result.append(id);
            } else {
                std.debug.print("Warning: No ID found for segment: '{s}'\n", .{segment});
                try result.append(0);
            }
        }
        return result;
    }

    pub fn encode(self: *Tokenizer, text: []const u8) ![]const u32 {
        var result = std.ArrayList(u32).init(self.allocator);
        errdefer result.deinit();
        var parts = try self.splitWithSpecials(text);
        defer {
            for (parts.items) |part| {
                self.allocator.free(part);
            }
            parts.deinit();
        }
        for (parts.items) |part| {
            if (self.vocab.get(part)) |id| {
                try result.append(id);
                continue;
            }
            var tokens = try self.splitWithPattern(part);
            defer {
                for (tokens.items) |token| {
                    self.allocator.free(token);
                }
                tokens.deinit();
            }
            for (tokens.items) |token| {
                var token_ids = try self.bpeMerges(token);
                defer token_ids.deinit();
                try result.appendSlice(token_ids.items);
            }
        }
        return result.toOwnedSlice();
    }

    pub fn decode(self: *Tokenizer, token_ids: []const u32) ![]const u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        errdefer result.deinit();
        for (token_ids) |id| {
            if (self.id_to_token.get(id)) |token| {
                try result.appendSlice(token);
            } else {
                std.debug.print("Warning: Unknown token ID: {d}\n", .{id});
            }
        }
        return result.toOwnedSlice();
    }

    pub fn encodeChat(self: *Tokenizer, comptime chat_format: ?[]const u8, system_message: []const u8, user_message: []const u8) ![]const u32 {
        const format = chat_format orelse
            \\<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            \\
            \\Cutting Knowledge Date: December 2023
            \\Today Date: 26 Jul 2024
            \\
            \\{s}<|eot_id|><|start_header_id|>user<|end_header_id|>
            \\
            \\{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            \\
            \\
        ;
        const formatted = try std.fmt.allocPrint(self.allocator, format, .{ system_message, user_message });
        defer self.allocator.free(formatted);
        return self.encode(formatted);
    }
};

fn formatRange(comptime format: []const u8, comptime start: usize, comptime end: usize) [end - start][]const u8 {
    var result: [end - start][]const u8 = undefined;
    inline for (&result, 0..) |*ptr, i| {
        ptr.* = std.fmt.comptimePrint(format, .{start + i});
    }
    return result;
}

test "Tokenizer round-trip" {
    std.debug.print("\n=== TOKENIZER.ZIG ===\n\n", .{});
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    const specials = [_][]const u8{
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|finetune_right_pad_id|>",
        "<|step_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",
        "<|eot_id|>",
        "<|python_tag|>",
    } ++ comptime formatRange("<|reserved_special_token_{d}|>", 2, 247);
    var tokenizer = try Tokenizer.initFromTikToken(allocator, pattern, "tokenizer.model", &specials);
    defer tokenizer.deinit();
    const text =
        \\<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        \\
        \\Cutting Knowledge Date: December 2023
        \\Today Date: 26 Jul 2024
        \\
        \\s<|eot_id|><|start_header_id|>user<|end_header_id|>
        \\
        \\s<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        \\
        \\
    ;
    std.debug.print("Original text: \"{s}\"\n\n", .{text});
    std.debug.print("1. Encoding text to token IDs...\n", .{});
    const token_ids = try tokenizer.encode(text);
    defer allocator.free(token_ids);
    std.debug.print("   Result: {d} tokens\n", .{token_ids});
    for (token_ids, 0..) |id, i| {
        if (tokenizer.id_to_token.get(id)) |token| {
            std.debug.print("   Token {d}: ID {d} = '{s}'\n", .{ i, id, token });
        } else {
            std.debug.print("   Token {d}: ID {d} = UNKNOWN\n", .{ i, id });
        }
    }
    std.debug.print("\n2. Decoding token IDs back to text...\n", .{});
    const decoded_text = try tokenizer.decode(token_ids);
    defer allocator.free(decoded_text);
    std.debug.print("   Result: \"{s}\"\n", .{decoded_text});
    std.debug.print("\n3. Verifying round-trip accuracy...\n", .{});
    const match = std.mem.eql(u8, text, decoded_text);
    std.debug.print("   Original and decoded text match: {}\n", .{match});
    try std.testing.expect(match);
    if (!match) {
        var i: usize = 0;
        const min_len = @min(text.len, decoded_text.len);
        while (i < min_len and text[i] == decoded_text[i]) {
            i += 1;
        }
        if (i < min_len) {
            const start = if (i > 10) i - 10 else 0;
            const end = @min(i + 10, min_len);
            std.debug.print("\nFirst difference at position {d}:\n", .{i});
            std.debug.print("  Original: ...{s}[{c}]{s}...\n", .{ text[start..i], text[i], text[i + 1 .. end] });
            std.debug.print("  Decoded:  ...{s}[{c}]{s}...\n", .{ decoded_text[start..i], decoded_text[i], decoded_text[i + 1 .. end] });
        } else if (text.len != decoded_text.len) {
            if (text.len > decoded_text.len) {
                std.debug.print("\nDecoded text is missing: '{s}'\n", .{text[min_len..]});
            } else {
                std.debug.print("\nDecoded text has extra: '{s}'\n", .{decoded_text[min_len..]});
            }
        }
    }
}
