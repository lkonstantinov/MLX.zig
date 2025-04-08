# MLX.zig

A [Zig](https://ziglang.org/) language binding for [MLX](https://github.com/ml-explore/mlx), Apple's array framework for machine learning on Apple Silicon.

## Prerequisites

- Apple Silicon Mac
- Zig v0.13.0
- CMake

## Getting Started

Clone the repository:
```fish
git clone https://github.com/jaco-bro/MLX.zig.git
cd MLX.zig
```

## Build Options

Build all executables:
```fish
zig build
```

This creates the following executables in `zig-out/bin/`:
- `whisper` - Whisper speech-to-text
- `llm` - LLM models (Qwen-2.5, Olympic-Coder, ...) for chat/code generation
- `phi` - Phi-4 text generation
- `llama` - Llama-3.2 chat
- `main` - Basic info

## Speech-to-Text with Whisper-Turbo-Large

Run using the build system:
```fish
zig build run-whisper
```

Run with custom audio file:
```fish
zig build run-whisper -- alive.mp3
```

Or run the executable directly:
```fish
zig-out/bin/whisper alive.mp3
```

Output:
```text
87.00 tokens-per-second (1408 tokens in 16184.00 ms)

Transcription of alive.mp3: Oh, hello, you're alive. Great, welcome to the world. Have a seat because you can't walk yet. You're gonna spend the next few years in the psychedelic world of colours that make no sense, but it's alright because you can pee yourself and somebody will probably sort it out for you. But don't get used to it, soon they'll be expecting you to use a toilet, so make the most of it. And not long after that you'll have to go to a building where they'll make you learn stuff and prove you know it, like times tables and the alphabet and whatnot. Maybe you still think you're the centre of the world and you can probably get away with that for a while. Some people do their entire lives. But eventually you're gonna start pissing other kids off, so you're probably gonna have to learn some humility soon too. Got it? Good, alright. Then you're gonna go to another building where the tests are a bit harder and the subjects are more intense. They'll try to teach you stuff like trigonometry and iambic pentameter without ever actually explaining what you can use it for, but don't worry, just memorise it and spit it out and forget it the second you walk out of the exam hall. By now you're probably getting weird urges to do stuff to your classmates that you never really wanted to do before, and now you're gonna have to play a game for the rest of your life where you really want this kind of closeness with people, but sometimes not everybody feels mutually, so you're going to have to hide it. Welcome to the world of dating, or body language and sex. Yeah, you're gonna like the last one. It's going to dictate your life and most of the films you watch and books you read for some time to come, whether you realise it or not. Oh, you're finished spitting out all that rote memorisation. Well, great, let's go to university. You'll need to if you want to earn lots of money, which is obviously very important because, well, it just is. Shut up. Look, everyone's happy when they're rich. Pick a subject. Not the humanities, you idiot. Something real like law or maths. I didn't spend 18 years raising a fucking philosophy major. Cogito ergo broke all the time. "Oh, you're finished? Great. Well, it's off to the companies for you then." Tell them you're a people person and you have excellent organizational skills and you work well in a team. Don't mention your actual passions for landscape gardening or music. They don't give a shit. Just come off as generic as possible. Stick it out for about 30 years. You'll make good money in time. Only the sex thing is probably getting a little empty by now and you're craving some kind of actual connection with the opposite sex or same sex if that's your thing. Jesus, you thought getting people to take their clothes off was difficult. You try finding a partner to fall in love with. And even then, what if they get bored or you get bored or they go off with the milkman or something? Well, sorry, you're just gonna have to risk it like everyone else. Like life, actually. Some people are dead by your rage, but you're not. No, you're still sat in a pit of your own mediocrity, feeling dull and stepped on by life. Standing on a rock that's spinning at 9,000 miles an hour around a gas giant in an infinite universe, the product of 13 billion years of cosmic evolution, but no, no, definitely you carry on being bored and feeling like crap. Only now you feel worse, because you know how great you should feel about everything, amazed and happy all the time, and yet you still feel like shit. Well, that's biology. Well, maybe your friends are getting rich, or getting married, or getting pregnant, or something, and you're poor and single, and maybe you don't want kids. It doesn't matter what Carl Sagan says. You don't feel any sense of wonder at all. You feel like shit. You don't want eloquent prose about how beautiful the cosmos is. You want money to live comfortably on. You want to be in love, and maybe you want children. Try books. There's quite a few dead guys who are willing to claim they can explain what you're doing here and how you can be happy, but loads of them just contradict each other, and to be honest, it all comes down to you. You're going to have to decide whether you believe in God, or want to eat meat, or support abortion, or feel that life has intrinsic meaning. And whatever you do, people will shit on your opinions and tell you you're delusional. Sorry, it's a game with no winners. And now you're old and maybe you've got money and maybe you haven't. Same with a partner and child. And now you're two steps from death. And you spend a lot of time thinking about what you could have done. And Jennifer Smith in the fields behind your parents' house when you were both 17. And how you should have said, I love you. And instead you said, look, I'm sorry, I'm just not in the best place right now. Come on, it's getting cold. Well, no use thinking about it now. Jennifer's probably old and doddery just like you are. Not much time left. Well, I guess I'll just do it all again differently the next time. Oh, there isn't a next time? Oh, that was it. Shit, I wish I'd known. Because if I'd known that this was the one chance I had to live as a talking monkey in space at the best point in history as the smartest species on the planet, using fucking magic on a daily basis like the internet and jet planes and smartphones, with access to all human knowledge at my fingertips and the chance to talk about how cool being alive is, I might have not worried so much about what other people thought and their shitty lives. And I might have just spent one little time there was making good art or doing good science or falling in love or just not being a dick. Oh well. If only I'd known. Which I did. But I just didn't really want to. want to think about it. Ho hum, so it goes. Gracias. Продолжение следует...
```

Code:
```zig
// Load transcriber with model name
var transcriber = try Transcriber.init(allocator, "whisper-large-v3-turbo");
defer transcriber.deinit();

// Transcribe audio file
const transcription = try transcriber.transcribe("test.mp3");
defer allocator.free(transcription);
```

## Large Language Models - Qwen-2.5, Olympic-Coder, ...

The `llm` executable provides a unified interface to easily switch between different LLM models via a consistent command-line interface.

Run using the build system:
```fish
zig build run-llm
```

Or run the executable directly with various options:
```fish
zig build
zig-out/bin/llm [options] [input]
```

### Options:
```
--model-type=TYPE       Model type: llama, phi, qwen, olympic (default: llama)
--model-name=NAME       Model name to download/use
--system-prompt=PROMPT  System prompt for the model
--num-tokens=N          Number of tokens to generate
--help                  Show this help
```

### Examples:

```fish
# Use Llama (default)
zig-out/bin/llm

# Use Qwen Coder
zig-out/bin/llm --model-type=qwen

# Use Olympic Coder model for programming tasks
zig-out/bin/llm --model-type=olympic "Write a python program to calculate the 10th Fibonacci number"

# Use Phi model with custom prompt
zig-out/bin/llm --model-type=phi --system-prompt="You are a helpful assistant" "How should I explain the Internet?"
```

## Phi-4 Text Generation

Run using the build system:
```fish
zig build run-phi
```

Run the executable directly with optional prompt:
```fish
zig-out/bin/phi "How should I explain the Internet?"
```

Output:
```text
Prompt:     100.56 tokens-per-second (36 tokens in 358.00 ms)
Generation: 37.78 tokens-per-second (100 tokens in 2646.00 ms)

Input: How should I explain the Internet?

Output: Certainly! Here's how you might explain the concept of the Internet to someone from a medieval perspective:

### Explanation:

**1. Basic Concept:**
   - **Internet**: Imagine the Internet as a vast library where knowledge and communication flow like water through interconnected channels. It's like a grand bazaar where people from distant lands can exchange ideas and goods without ever leaving their homes.

**2. Communication:**
   - **Messages and Letters**: Think of email and instant messages as modern letters. They
```

Code:
```zig
// Load tokenizer
var tokenizer = try Tokenizer.init(allocator, model_name);
defer tokenizer.deinit();

// Load transformer
var transformer = try Transformer.init(allocator, model_name);
defer transformer.deinit();

// Encode input string to token IDs
const input_ids = try tokenizer.encodeChat(chat_format, sys_prompt, user_input);
defer allocator.free(input_ids);

// Generate new tokens
const output_ids = try transformer.generate(input_ids, n_toks);
defer allocator.free(output_ids);
```

## Chat with Llama-3.2

Run using the build system:
```fish
zig build run-llama
```

Or run the executable directly:
```fish
zig-out/bin/llama
```

```text
Enter your message: How to get length of ArrayList in Zig?
```

Output:
~~~text
Prompt:     322.50 tokens-per-second (50 tokens in 155.00 ms)
Generation: 77.70 tokens-per-second (100 tokens in 1287.00 ms)

Input: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

How to get length of ArrayList in Zig?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Output: In Zig, you can get the length of an `ArrayList` using the `len()` function. Here's an example:

```zig
const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GpaAllocator();
    defer gpa.deinit();

    var array = try gpa.allocator().create([10] int);
    defer array.deinit();

    try array.allocator().allocate(10, .{});
    try array.
~~~

## Acknowledgements

This project was inspired by Erik Kaunismäki's [zig-build-mlx](https://github.com/ErikKaum/zig-build-mlx).

## License

This project is licensed under the [Apache License 2.0](LICENSE)
