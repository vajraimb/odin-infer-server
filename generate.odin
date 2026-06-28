/* Token generation loop shared by /api/generate and /api/chat */

package main

import infer "infer:infer"
import sampler "sampler:sampler"
import tokenizer "tokenizer:tokenizer"

import "core:fmt"
import "core:strings"
import "core:sync"
import "core:time"

EOS_TOKEN_ID :: 151645

Gen_Options :: struct {
	temperature: f32,
	top_p:       f32,
	max_tokens:  int,
	think:       bool,
}

Gen_State :: struct {
	engine:     ^infer.Engine,
	tok:        ^tokenizer.Tokenizer,
	samp:       ^sampler.Sampler,
	mu:         ^sync.Mutex,
	model_name: string,
	model_path: string,
}

Stream_Callback :: #type proc(chunk: string, done: bool, user_data: rawptr)

time_in_ms :: proc() -> i64 {
	return time.to_unix_nanoseconds(time.now()) / 1_000_000
}

build_chat_prompt :: proc(
	messages: []Chat_Message,
	think: bool,
	allocator := context.allocator,
) -> string {
	b := strings.builder_make(allocator)
	defer strings.builder_destroy(&b)

	for msg in messages {
		switch msg.role {
		case "system":
			fmt.sbprintf(&b, "<|im_start|>system\n%s<|im_end|>\n", msg.content)
		case "user":
			fmt.sbprintf(&b, "<|im_start|>user\n%s<|im_end|>\n", msg.content)
		case "assistant":
			fmt.sbprintf(&b, "<|im_start|>assistant\n%s<|im_end|>\n", msg.content)
		}
	}
	fmt.sbprint(&b, "<|im_start|>assistant\n")
	if !think {
		fmt.sbprint(&b, "<think>\n\n</think>\n")
	}
	return strings.clone(strings.to_string(b), allocator)
}

generate_tokens :: proc(
	state: ^Gen_State,
	prompt: string,
	opts: Gen_Options,
	stream: Stream_Callback = nil,
	stream_user: rawptr = nil,
) -> (
	response: string,
	prompt_tokens: int,
	gen_tokens: int,
	ok: bool,
) {
	sync.mutex_lock(state.mu)
	defer sync.mutex_unlock(state.mu)

	cfg := infer.engine_config(state.engine)
	samp := state.samp^
	samp.temperature = opts.temperature
	samp.topp = opts.top_p

	encoded, err := tokenizer.encode(state.tok, prompt)
	if err != nil {
		return "", 0, 0, false
	}
	defer delete(encoded)

	prompt_ids := encoded
	if len(prompt_ids) >= cfg.seq_len {
		prompt_ids = prompt_ids[:cfg.seq_len - 1]
	}

	num_prompt := len(prompt_ids)
	max_gen := opts.max_tokens
	if max_gen <= 0 {
		max_gen = cfg.seq_len - num_prompt
	}
	if max_gen < 0 do max_gen = 0

	pos := 0
	next := 0
	gen_count := 0
	builder := strings.builder_make()
	defer strings.builder_destroy(&builder)

	for {
		token: int
		if pos < num_prompt {
			token = prompt_ids[pos]
		} else {
			token = next
		}

		if pos >= cfg.seq_len do break

		logits := infer.engine_forward(state.engine, token, pos)
		next = sampler.sample(&samp, logits)
		pos += 1

		if pos < num_prompt do continue

		if next == EOS_TOKEN_ID do break
		if gen_count >= max_gen do break

		piece := tokenizer.decode_token_id(state.tok, next)
		if len(piece) > 0 {
			if stream != nil {
				stream(piece, false, stream_user)
			}
			strings.write_string(&builder, piece)
		}
		delete(piece)
		gen_count += 1
	}

	if stream != nil {
		stream("", true, stream_user)
	}

	return strings.clone(strings.to_string(builder)), num_prompt, gen_count, true
}
