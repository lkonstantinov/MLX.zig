# MLX.zig

A [Zig](https://ziglang.org/) language binding for [MLX](https://github.com/ml-explore/mlx), Apple's array framework for machine learning on Apple Silicon.

## Features

1. **Build System**: Compiles MLX (C++), MLX-C, and PCRE2 from source using Zig's build system (no CMake or other build tools needed)
2. **Transformer**: Implements a Llama-3.2 language model with attention mechanisms and key-value caching
3. **Tokenizer**: Uses PCRE2 for efficient regex-based text processing, handling complex patterns and special tokens
4. **Transcriber**: Supports Whisper audio transcription with OpenAI's speech-to-text models

## Prerequisites

- Apple Silicon Mac
- Zig v0.13.0

## Getting Started

1. Clone the repository:
```
git clone https://github.com/jaco-bro/MLX.zig.git
cd MLX.zig
```

2. Run one of the available models:

### Speech-to-Text with Whisper-Turbo-Large
```
zig build run-whisper
```

This will transcribe an audio file (default is "alive.mp3"):

```text
Transcription of alive.mp3: Oh, hello, you're alive. Great, welcome to the world. Have a seat because you can't walk yet. You're gonna spend the next few years in the psychedelic world of colours that make no sense, but it's alright because you can pee yourself and somebody will probably sort it out for you. But don't get used to it, soon they'll be expecting you to use a toilet, so make the most of it. And not long after that you'll have to go to a building where they'll make you learn stuff and prove you know it, like times tables and the alphabet and whatnot. Maybe you still think you're the centre of the world and you can probably get away with that for a while. Some people do their entire lives. But eventually you're gonna start pissing other kids off, so you're probably gonna have to learn some humility soon too. Got it? Good, alright. Then you're gonna go to another building where the tests are a bit harder and the subjects are more intense. They'll try to teach you stuff like trigonometry and iambic pentameter without ever actually explaining what you can use it for, but don't worry, just memorise it and spit it out and forget it the second you walk out of the exam hall. By now you're probably getting weird urges to do stuff to your classmates that you never really wanted to do before, and now you're gonna have to play a game for the rest of your life where you really want this kind of closeness with people, but sometimes not everybody feels mutually, so you're going to have to hide it. Welcome to the world of dating, or body language and sex. Yeah, you're gonna like the last one. It's going to dictate your life and most of the films you watch and books you read for some time to come, whether you realise it or not. Oh, you're finished spitting out all that rote memorisation. Well, great, let's go to university. You'll need to if you want to earn lots of money, which is obviously very important because, well, it just is. Shut up. Look, everyone's happy when they're rich. Pick a subject. Not the humanities, you idiot. Something real like law or maths. I didn't spend 18 years raising a fucking philosophy major. Cogito ergo broke all the time. "Oh, you're finished? Great. Well, it's off to the companies for you then." Tell them you're a people person and you have excellent organizational skills and you work well in a team. Don't mention your actual passions for landscape gardening or music. They don't give a shit. Just come off as generic as possible. Stick it out for about 30 years. You'll make good money in time. Only the sex thing is probably getting a little empty by now and you're craving some kind of actual connection with the opposite sex or same sex if that's your thing. Jesus, you thought getting people to take their clothes off was difficult. You try finding a partner to fall in love with. And even then, what if they get bored or you get bored or they go off with the milkman or something? Well, sorry, you're just gonna have to risk it like everyone else. Like life, actually. Some people are dead by your rage, but you're not. No, you're still sat in a pit of your own mediocrity, feeling dull and stepped on by life. Standing on a rock that's spinning at 9,000 miles an hour around a gas giant in an infinite universe, the product of 13 billion years of cosmic evolution, but no, no, definitely you carry on being bored and feeling like crap. Only now you feel worse, because you know how great you should feel about everything, amazed and happy all the time, and yet you still feel like shit. Well, that's biology. Well, maybe your friends are getting rich, or getting married, or getting pregnant, or something, and you're poor and single, and maybe you don't want kids. It doesn't matter what Carl Sagan says. You don't feel any sense of wonder at all. You feel like shit. You don't want eloquent prose about how beautiful the cosmos is. You want money to live comfortably on. You want to be in love, and maybe you want children. Try books. There's quite a few dead guys who are willing to claim they can explain what you're doing here and how you can be happy, but loads of them just contradict each other, and to be honest, it all comes down to you. You're going to have to decide whether you believe in God, or want to eat meat, or support abortion, or feel that life has intrinsic meaning. And whatever you do, people will shit on your opinions and tell you you're delusional. Sorry, it's a game with no winners. And now you're old and maybe you've got money and maybe you haven't. Same with a partner and child. And now you're two steps from death. And you spend a lot of time thinking about what you could have done. And Jennifer Smith in the fields behind your parents' house when you were both 17. And how you should have said, I love you. And instead you said, look, I'm sorry, I'm just not in the best place right now. Come on, it's getting cold. Well, no use thinking about it now. Jennifer's probably old and doddery just like you are. Not much time left. Well, I guess I'll just do it all again differently the next time. Oh, there isn't a next time? Oh, that was it. Shit, I wish I'd known. Because if I'd known that this was the one chance I had to live as a talking monkey in space at the best point in history as the smartest species on the planet, using fucking magic on a daily basis like the internet and jet planes and smartphones, with access to all human knowledge at my fingertips and the chance to talk about how cool being alive is, I might have not worried so much about what other people thought and their shitty lives. And I might have just spent one little time there was making good art or doing good science or falling in love or just not being a dick. Oh well. If only I'd known. Which I did. But I just didn't really want to. want to think about it. Ho hum, so it goes. Gracias. Продолжение следует...⏎  
```

### Chat with Llama-3.2-1B-Instruct
```
zig build run-llama
```

This will compile MLX from source and launch an interactive prompt where you can chat with the model:

```text
Enter your message:
```

<details>
<summary>Click to expand Llama chat example</summary>

```text
Enter your message: Hi, how have you been?

Input IDs: { 128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 271, 2675, 527, 264, 11190, 18328, 13, 128009, 128006, 882, 128007, 271, 13347, 11, 1268, 617, 499, 1027, 30, 128009, 128006, 78191, 128007, 271 }

Generated token 1/100: 40
Generated token 2/100: 2846
Generated token 3/100: 1120
Generated token 4/100: 264
Generated token 5/100: 4221
Generated token 6/100: 1646
Generated token 7/100: 11
Generated token 8/100: 358
Generated token 9/100: 1541
Generated token 10/100: 956
Generated token 11/100: 617
Generated token 12/100: 21958
Generated token 13/100: 477
Generated token 14/100: 16024
Generated token 15/100: 1093
Generated token 16/100: 12966
Generated token 17/100: 656
Generated token 18/100: 11
Generated token 19/100: 719
Generated token 20/100: 358
Generated token 21/100: 2846
Generated token 22/100: 31301
Generated token 23/100: 10489
Generated token 24/100: 323
Generated token 25/100: 5644
Generated token 26/100: 311
Generated token 27/100: 1520
Generated token 28/100: 449
Generated token 29/100: 904
Generated token 30/100: 4860
Generated token 31/100: 477
Generated token 32/100: 9256
Generated token 33/100: 499
Generated token 34/100: 1253
Generated token 35/100: 617
Generated token 36/100: 13
Generated token 37/100: 1102
Generated token 38/100: 596
Generated token 39/100: 2294
Generated token 40/100: 311
Generated token 41/100: 6865
Generated token 42/100: 430
Generated token 43/100: 499
Generated token 44/100: 2351
Generated token 45/100: 10371
Generated token 46/100: 1268
Generated token 47/100: 358
Generated token 48/100: 3077
Generated token 49/100: 1027
Generated token 50/100: 0
Generated token 51/100: 358
Generated token 52/100: 2846
Generated token 53/100: 3815
Generated token 54/100: 1664
Generated token 55/100: 11
Generated token 56/100: 9523
Generated token 57/100: 369
Generated token 58/100: 10371
Generated token 59/100: 13
Generated token 60/100: 358
Generated token 61/100: 3077
Generated token 62/100: 1027
Generated token 63/100: 16572
Generated token 64/100: 389
Generated token 65/100: 264
Generated token 66/100: 13057
Generated token 67/100: 3392
Generated token 68/100: 315
Generated token 69/100: 1495
Generated token 70/100: 828
Generated token 71/100: 11
Generated token 72/100: 902
Generated token 73/100: 6276
Generated token 74/100: 757
Generated token 75/100: 311
Generated token 76/100: 3493
Generated token 77/100: 11190
Generated token 78/100: 323
Generated token 79/100: 39319
Generated token 80/100: 14847
Generated token 81/100: 311
Generated token 82/100: 701
Generated token 83/100: 20126
Generated token 84/100: 13
Generated token 85/100: 2650
Generated token 86/100: 922
Generated token 87/100: 499
Generated token 88/100: 30
Generated token 89/100: 2650
Generated token 90/100: 596
Generated token 91/100: 701
Generated token 92/100: 1938
Generated token 93/100: 2133
Generated token 94/100: 779
Generated token 95/100: 3117
Generated token 96/100: 30
Generated token 97/100: 128009
EOS token reached after 97 tokens

Output IDs: { 40, 2846, 1120, 264, 4221, 1646, 11, 358, 1541, 956, 617, 21958, 477, 16024, 1093, 12966, 656, 11, 719, 358, 2846, 31301, 10489, 323, 5644, 311, 1520, 449, 904, 4860, 477, 9256, 499, 1253, 617, 13, 1102, 596, 2294, 311, 6865, 430, 499, 2351, 10371, 1268, 358, 3077, 1027, 0, 358, 2846, 3815, 1664, 11, 9523, 369, 10371, 13, 358, 3077, 1027, 16572, 389, 264, 13057, 3392, 315, 1495, 828, 11, 902, 6276, 757, 311, 3493, 11190, 323, 39319, 14847, 311, 701, 20126, 13, 2650, 922, 499, 30, 2650, 596, 701, 1938, 2133, 779, 3117, 30, 128009 }

Input: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi, how have you been?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Output: I'm just a language model, I don't have emotions or feelings like humans do, but I'm functioning properly and ready to help with any questions or tasks you may have. It's great to hear that you're asking how I've been! I'm doing well, thanks for asking. I've been trained on a vast amount of text data, which allows me to provide helpful and informative responses to your queries. How about you? How's your day going so far?<|eot_id|>⏎   
```

</details>

*The Llama example uses 4-bit quantization to reduce the Llama 3.2 model size from 2.7GB (bfloat16) to under 0.7GB.*

## Examples

### Llama Language Model

```zig
// Load tokenizer
var tokenizer = try Tokenizer.init(allocator, model_name);
defer tokenizer.deinit();

// Load transformer
var transformer = try Transformer.init(allocator, model_name);
defer transformer.deinit();

// Encode input string to token IDs (chat format)
const input_ids = try tokenizer.encodeChat(null, sys_prompt, user_input);
defer allocator.free(input_ids);

// Generate new tokens
const output_ids = try transformer.generate(input_ids, n_toks);
defer allocator.free(output_ids);
```

### Whisper Transcription

```zig
// Load transcriber with model name
var transcriber = try Transcriber.init(allocator, "whisper-large-v3-turbo");
defer transcriber.deinit();

// Transcribe audio file
const transcription = try transcriber.transcribe("test.mp3");
defer allocator.free(transcription);
```

## Acknowledgements

This project's build system is based on Erik Kaunismäki's [zig-build-mlx](https://github.com/ErikKaum/zig-build-mlx).

## License

This project is licensed under the [Apache License 2.0](LICENSE)
