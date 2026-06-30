/* Token generation loop shared by /api/generate and /api/chat. Dispatches to
   the Qwen3 or Qwen3.5 (Ornith) engine+tokenizer based on Gen_State.kind. */

package main

import ggml "ggml:ggml"
import infer "infer:infer"
import q35 "qwen3_5:qwen3_5"
import sampler "sampler:sampler"
import tokenizer "tokenizer:tokenizer"
import tok35 "qwen3_5_tokenizer:qwen3_5_tokenizer"

import "core:fmt"
import "core:mem"
import "core:strings"
import "core:sync"
import "core:time"

EOS_QWEN3   :: 151645  // <|im_end|>
EOS_QWEN3_5 :: 248046  // <|im_end|>

Model_Kind :: enum { Qwen3, Qwen3_5 }

Gen_Options :: struct {
	temperature: f32,
	top_p:       f32,
	max_tokens:  int,
	think:       bool,
}

Gen_State :: struct {
	kind:         Model_Kind,
	engine_q3:    ^infer.Engine,
	engine_q35:   ^q35.Engine,
	tok_q3:       ^tokenizer.Tokenizer,
	tok_q35:      ^tok35.Tokenizer,
	samp:         ^sampler.Sampler,
	mu:           ^sync.Mutex,
	model_name:   string,
	model_path:   string,
	rep_penalty:  f32, // server-wide repetition penalty (1.0 = off)
}

Stream_Callback :: #type proc(chunk: string, done: bool, user_data: rawptr)

time_in_ms :: proc() -> i64 {
	return time.to_unix_nanoseconds(time.now()) / 1_000_000
}

// Peek the GGUF metadata to decide which engine/tokenizer to use.
detect_arch :: proc(path: string) -> Model_Kind {
	g: ggml.GGUF_File
	ggml.parse_gguf(path, &g)
	defer ggml.free_gguf(&g)
	arch, _ := ggml.gguf_meta_str(&g, "general.architecture")
	switch arch {
	case "qwen35", "qwen3_5", "qwen3.5", "qwen3_5_text":
		return .Qwen3_5
	}
	return .Qwen3
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

	seq_len: int
	if state.kind == .Qwen3_5 {
		seq_len = q35.engine_config(state.engine_q35).seq_len
	} else {
		seq_len = infer.engine_config(state.engine_q3).seq_len
	}

	samp := state.samp^
	samp.temperature = opts.temperature
	samp.topp = opts.top_p
	// Per-request repetition penalty: clear the shared window so only this
	// request's tokens count. Requests are serialised by the mutex.
	rep := state.rep_penalty
	if rep > 1.0 && samp.last_tokens != nil {
		clear(&samp.last_tokens)
	}

	encoded: []int
	err: mem.Allocator_Error
	if state.kind == .Qwen3_5 {
		encoded, err = tok35.encode(state.tok_q35, prompt)
	} else {
		encoded, err = tokenizer.encode(state.tok_q3, prompt)
	}
	if err != nil {
		return "", 0, 0, false
	}
	defer delete(encoded)

	prompt_ids := encoded
	if len(prompt_ids) >= seq_len {
		prompt_ids = prompt_ids[:seq_len - 1]
	}
	if rep > 1.0 {
		for id in prompt_ids {
			sampler.record_token(&samp, id)
		}
	}

	num_prompt := len(prompt_ids)
	max_gen := opts.max_tokens
	if max_gen <= 0 {
		max_gen = seq_len - num_prompt
	}
	if max_gen < 0 do max_gen = 0

	eos := state.kind == .Qwen3_5 ? EOS_QWEN3_5 : EOS_QWEN3

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

		if pos >= seq_len do break

		logits: []f32
		if state.kind == .Qwen3_5 {
			logits = q35.engine_forward(state.engine_q35, token, pos)
		} else {
			logits = infer.engine_forward(state.engine_q3, token, pos)
		}
		next = sampler.sample(&samp, logits)
		pos += 1

		if pos < num_prompt do continue

		if next == eos do break
		if gen_count >= max_gen do break

		piece: string
		if state.kind == .Qwen3_5 {
			piece = tok35.decode_token_id(state.tok_q35, next)
		} else {
			piece = tokenizer.decode_token_id(state.tok_q3, next)
		}
		if len(piece) > 0 {
			if stream != nil {
				stream(piece, false, stream_user)
			}
			strings.write_string(&builder, piece)
		}
		delete(piece)
		if rep > 1.0 {
			sampler.record_token(&samp, next)
		}
		gen_count += 1
	}

	if stream != nil {
		stream("", true, stream_user)
	}

	return strings.clone(strings.to_string(builder)), num_prompt, gen_count, true
}
