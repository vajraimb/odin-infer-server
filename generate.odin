/* Token generation loop shared by /api/generate and /api/chat. Dispatches to
   the Qwen3 or Qwen3.5 (Ornith) engine+tokenizer based on Gen_State.kind. */

package main

import ggml "ggml:ggml"
import infer "infer:infer"
import q35 "qwen3_5:qwen3_5"
import sampler "sampler:sampler"
import tokenizer "tokenizer:tokenizer"
import tok35 "qwen3_5_tokenizer:qwen3_5_tokenizer"

import "core:encoding/json"
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
	rep_penalty:  f32,            // server-wide repetition penalty (1.0 = off)
	cache_tokens: [dynamic]int,   // token ids currently loaded into engine state (prompt + generated)
	cache_len:    int,            // valid prefix length in cache_tokens (== positions filled)
}

Stream_Callback :: #type proc(chunk: string, done: bool, user_data: rawptr)

// Length of the leading token-identical prefix between two sequences.
common_prefix_len :: proc(a: []int, b: []int) -> int {
	n := min(len(a), len(b))
	for i in 0 ..< n {
		if a[i] != b[i] do return i
	}
	return n
}

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
	tools: []Tool_Def,
	think: bool,
	allocator := context.allocator,
) -> string {
	b := strings.builder_make(allocator)
	defer strings.builder_destroy(&b)

	tools_section := ""
	if len(tools) > 0 {
		tools_section = build_tools_section(tools, allocator)
	}
	defer {
		if len(tools_section) > 0 do delete(tools_section)
	}

	tools_injected := false

	// If no system message exists, prepend a tools system message so tool
	// definitions appear before the user's question (standard chat order).
	has_system := false
	if len(tools) > 0 {
		for msg in messages {
			if msg.role == "system" {
				has_system = true
				break
			}
		}
		if !has_system {
			fmt.sbprintf(&b, "<|im_start|>system\n%s<|im_end|>\n", tools_section)
			tools_injected = true
		}
	}

	for msg in messages {
		switch msg.role {
		case "system":
			if len(tools) > 0 && !tools_injected {
				fmt.sbprintf(&b, "<|im_start|>system\n%s%s<|im_end|>\n", msg.content, tools_section)
				tools_injected = true
			} else {
				fmt.sbprintf(&b, "<|im_start|>system\n%s<|im_end|>\n", msg.content)
			}
		case "user":
			fmt.sbprintf(&b, "<|im_start|>user\n%s<|im_end|>\n", msg.content)
		case "assistant":
			// Reconstruct past assistant turns the same way they were generated:
			// when think is off, an empty <think></think> block was injected
			// before the model's answer, so it must be present here too -- this
			// keeps the token sequence identical across turns and lets the
			// prefix cache hit (otherwise every turn diverges and resets).
			if !think {
				fmt.sbprintf(&b, "<|im_start|>assistant\n<think>\n\n</think>\n%s<|im_end|>\n", msg.content)
			} else {
				fmt.sbprintf(&b, "<|im_start|>assistant\n%s<|im_end|>\n", msg.content)
			}
		case "tool":
			// Ornith/Qwen3.5 chat template wraps tool results in <tool_response>
			// inside a user turn, not a separate "tool" role turn.
			fmt.sbprintf(&b, "<|im_start|>user\n<tool_response>\n%s\n</tool_response><|im_end|>\n", msg.content)
		}
	}

	if len(tools) > 0 && !tools_injected {
		fmt.sbprintf(&b, "<|im_start|>system\n%s<|im_end|>\n", tools_section)
	}

	fmt.sbprint(&b, "<|im_start|>assistant\n")
	if !think {
		fmt.sbprint(&b, "<think>\n\n</think>\n")
	}
	return strings.clone(strings.to_string(b), allocator)
}

// build_tools_section renders tool definitions in the Qwen3/Ornith tool-calling
// format. Appended to the system prompt so the model knows what tools exist and
// how to invoke them via <tool_call> XML tags.
build_tools_section :: proc(tools: []Tool_Def, allocator := context.allocator) -> string {
	b := strings.builder_make(allocator)
	defer strings.builder_destroy(&b)

	fmt.sbprint(&b, "\n\n# Tools\n\n")
	fmt.sbprint(&b, "You may call one or more functions to assist with the user query.\n\n")
	fmt.sbprint(&b, "You are provided with function signatures within <tools></tools> XML tags:\n")
	fmt.sbprint(&b, "<tools>\n")

	arena: mem.Dynamic_Arena
	mem.dynamic_arena_init(&arena, alignment = 64, block_allocator = allocator)
	defer mem.dynamic_arena_destroy(&arena)
	tmp := mem.dynamic_arena_allocator(&arena)

	for tool in tools {
		data, err := json.marshal(tool, allocator = tmp)
		if err != nil {
			continue
		}
		strings.write_string(&b, string(data))
		strings.write_string(&b, "\n")
	}

	fmt.sbprint(&b, "</tools>\n\n")
	fmt.sbprint(&b, "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n")
	fmt.sbprint(&b, "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>")

	return strings.clone(strings.to_string(b), allocator)
}

// extract_tool_calls scans model output for <tool_call>...</tool_call> blocks,
// parses each block (XML format <function=...><parameter=...> or JSON fallback),
// and returns the clean text plus parsed tool call entries.
extract_tool_calls :: proc(text: string) -> (clean_text: string, tool_calls: []Tool_Call_Entry) {
	open_tag := "<tool_call>"
	close_tag := "</tool_call>"

	clean_builder := strings.builder_make()
	defer strings.builder_destroy(&clean_builder)

	results := make([dynamic]Tool_Call_Entry)

	remaining := text
	call_index := 0
	for {
		open_pos := strings.index(remaining, open_tag)
		if open_pos < 0 {
			strings.write_string(&clean_builder, remaining)
			break
		}
		strings.write_string(&clean_builder, remaining[:open_pos])

		after_open := open_pos + len(open_tag)
		rest := remaining[after_open:]
		close_pos := strings.index(rest, close_tag)
		if close_pos < 0 {
			strings.write_string(&clean_builder, remaining[open_pos:])
			break
		}

		content := strings.trim_space(rest[:close_pos])

		// Try XML format first: <function=NAME>...<parameter=KEY>VALUE</parameter>...</function>
		name, args, xml_ok := parse_xml_tool_call(content)
		if xml_ok {
			append(&results, Tool_Call_Entry{
				index = call_index,
				id = fmt.tprintf("call_%d", call_index),
				function = Tool_Call_Function{
					name = name,
					arguments = args,
				},
			})
			call_index += 1
		} else {
			// Fallback: JSON format {"name": "...", "arguments": {...}}
			parsed: struct {
				name:      string     `json:"name"`,
				arguments: json.Value `json:"arguments"`,
			}
			if err := json.unmarshal_string(content, &parsed); err == nil {
				args2: json.Value = parsed.arguments
				if args2 == nil {
					args2 = json.Value(json.Object{})
				}
				append(&results, Tool_Call_Entry{
					index = call_index,
					id = fmt.tprintf("call_%d", call_index),
					function = Tool_Call_Function{
						name = parsed.name,
						arguments = args2,
					},
				})
				call_index += 1
			}
		}

		remaining = rest[close_pos + len(close_tag):]
	}

	clean_text = strings.clone(strings.to_string(clean_builder))
	if len(results) > 0 {
		return clean_text, results[:]
	}
	delete(results)
	return clean_text, nil
}

// Parse XML-format tool call content:
//   <function=NAME>
//   <parameter=KEY>
//   VALUE
//   </parameter>
//   </function>
// Returns (name, arguments as json.Object, ok). All strings are cloned so they
// outlive the input buffer.
parse_xml_tool_call :: proc(content: string) -> (name: string, args: json.Value, ok: bool) {
	func_tag := "<function="
	idx := strings.index(content, func_tag)
	if idx < 0 do return "", nil, false

	name_start := idx + len(func_tag)
	name_end := strings.index(content[name_start:], ">")
	if name_end < 0 do return "", nil, false
	name = strings.clone(strings.trim_space(content[name_start : name_start + name_end]))

	obj := make(json.Object)
	param_tag := "<parameter="
	scan := content[name_start + name_end + 1:]
	for {
		p_idx := strings.index(scan, param_tag)
		if p_idx < 0 do break
		key_start := p_idx + len(param_tag)
		key_end := strings.index(scan[key_start:], ">")
		if key_end < 0 do break
		key := strings.clone(strings.trim_space(scan[key_start : key_start + key_end]))

		val_start := key_start + key_end + 1
		val_end := strings.index(scan[val_start:], "</parameter>")
		if val_end < 0 do break
		val := strings.clone(strings.trim_space(scan[val_start : val_start + val_end]))

		obj[key] = json.Value(val)
		scan = scan[val_start + val_end + len("</parameter>"):]
	}

	args = json.Value(obj)
	return name, args, true
}

// free_tool_calls frees the heap-owned fields of a []Tool_Call_Entry returned by
// extract_tool_calls. Safe to call with nil.
free_tool_calls :: proc(tool_calls: []Tool_Call_Entry) {
	if len(tool_calls) == 0 do return
	for tc in tool_calls {
		delete(tc.id)
		delete(tc.function.name)
		json.destroy_value(tc.function.arguments)
	}
	delete(tool_calls)
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

	// ---- prefix caching: resume from the longest point the engine state allows ----
	// The engine's evolving state (KV cache + conv/recurrent state) is kept
	// between requests. Find how much of the new prompt is already loaded and
	// only prefill the rest. For agent multi-turn (each turn extends the last)
	// this turns O(history) prefill into O(new tokens).
	cache := state.cache_tokens[:state.cache_len]
	L := common_prefix_len(cache, prompt_ids)
	resume_at: int
	if state.kind == .Qwen3_5 {
		// Gated-delta recurrent state is not rewindable: only safe to resume if
		// the new prompt extends the cache exactly (L == cache_len). Otherwise
		// reset and recompute from scratch.
		if L == state.cache_len && L > 0 {
			resume_at = L
		} else {
			resume_at = 0
			if state.cache_len > 0 {
				q35.engine_reset_state(state.engine_q35)
				state.cache_len = 0
			}
		}
	} else {
		// Qwen3 KV cache is random-access: resume at the common-prefix length;
		// stale slots beyond L get overwritten as we prefill.
		resume_at = L
	}
	if resume_at > num_prompt do resume_at = num_prompt

	max_gen := opts.max_tokens
	if max_gen <= 0 {
		max_gen = seq_len - num_prompt
	}
	if max_gen < 0 do max_gen = 0

	eos := state.kind == .Qwen3_5 ? EOS_QWEN3_5 : EOS_QWEN3

	pos := resume_at
	next := 0
	gen_count := 0
	builder := strings.builder_make()
	defer strings.builder_destroy(&builder)
	gen_ids: [dynamic]int
	defer delete(gen_ids)
	t0 := time_in_ms()

	// prefill the tail [resume_at .. num_prompt). Qwen3.5 uses the batched MMQ
	// path (Stage 1b: all projections batched); Qwen3 stays per-token. The last
	// new token runs per-token so its sampled `next` is the first generated
	// token (same semantics as the old per-token loop).
	if state.kind == .Qwen3_5 {
		to_prefill := prompt_ids[resume_at:num_prompt]
		nt := len(to_prefill)
		if nt > 1 {
			q35.engine_forward_batch(state.engine_q35, to_prefill[:nt - 1], resume_at)
			logits := q35.engine_forward(state.engine_q35, to_prefill[nt - 1], resume_at + nt - 1)
			next = sampler.sample(&samp, logits)
			pos = num_prompt
		} else if nt == 1 {
			logits := q35.engine_forward(state.engine_q35, to_prefill[0], resume_at)
			next = sampler.sample(&samp, logits)
			pos = num_prompt
		}
	} else {
		for pos < num_prompt {
			if pos >= seq_len do break
			logits := infer.engine_forward(state.engine_q3, prompt_ids[pos], pos)
			next = sampler.sample(&samp, logits)
			pos += 1
		}
	}
	t_prefill := time_in_ms() - t0

	// generation
	utf8_buf: [dynamic]u8  // buffers incomplete UTF-8 from multi-byte tokens
	defer delete(utf8_buf)
	for {
		if pos >= seq_len do break
		if next == eos do break
		if gen_count >= max_gen do break

		append(&gen_ids, next)
		piece: string
		if state.kind == .Qwen3_5 {
			piece = tok35.decode_token_id(state.tok_q35, next)
		} else {
			piece = tokenizer.decode_token_id(state.tok_q3, next)
		}
		if len(piece) > 0 {
			strings.write_string(&builder, piece)
			if stream != nil {
				// Buffer decoded bytes and only emit complete UTF-8 characters.
				// A single emoji token may produce partial bytes (e.g. \xf0\x9f)
				// that aren't valid UTF-8 on their own; json.marshal would escape
				// them as \xNN (invalid JSON). Holding them until the next token
				// completes the sequence fixes streaming emoji.
				// NOTE: must use transmute([]u8) — `for c in string` iterates
				// runes (codepoints), not bytes, and u8(c) would truncate.
				raw_bytes := transmute([]u8)piece
				for b in raw_bytes do append(&utf8_buf, b)
				emit_len := valid_utf8_prefix_len(utf8_buf[:])
				if emit_len > 0 {
					stream(string(utf8_buf[:emit_len]), false, stream_user)
					remaining := len(utf8_buf) - emit_len
					if remaining > 0 {
						copy(utf8_buf[:remaining], utf8_buf[emit_len:])
					}
					resize(&utf8_buf, remaining)
				}
			}
		}
		delete(piece)
		if rep > 1.0 {
			sampler.record_token(&samp, next)
		}
		gen_count += 1

		logits: []f32
		if state.kind == .Qwen3_5 {
			logits = q35.engine_forward(state.engine_q35, next, pos)
		} else {
			logits = infer.engine_forward(state.engine_q3, next, pos)
		}
		next = sampler.sample(&samp, logits)
		pos += 1
	}

	// flush any remaining buffered bytes (should always be complete UTF-8 at EOS)
	if stream != nil && len(utf8_buf) > 0 {
		stream(string(utf8_buf[:]), false, stream_user)
	}
	if stream != nil {
		stream("", true, stream_user)
	}
	t_gen := time_in_ms() - t0 - t_prefill

	// per-request diagnostics on stderr: shows cache-hit point + where time goes
	fmt.eprintf(
		"[gen] prompt=%d resume=@%d (cached %d) new_prefill=%d gen=%d  %dms/%dms\n",
		num_prompt, resume_at, resume_at, num_prompt - resume_at, gen_count, t_prefill, t_gen,
	)

	// update cache: cache_tokens = prompt_ids + generated ids, so the next
	// request that extends this conversation can resume from pos.
	clear(&state.cache_tokens)
	append(&state.cache_tokens, ..prompt_ids)
	append(&state.cache_tokens, ..gen_ids[:])
	state.cache_len = pos

	return strings.clone(strings.to_string(builder)), num_prompt, gen_count, true
}

// Returns the length (in bytes) of the longest valid UTF-8 prefix of buf.
// Incomplete trailing sequences (e.g. the first 2 bytes of a 4-byte emoji) are
// excluded so the caller can buffer them for the next token.
valid_utf8_prefix_len :: proc(buf: []u8) -> int {
	i := 0
	for i < len(buf) {
		b := buf[i]
		if b < 0x80 {
			i += 1
		} else if b < 0xC0 {
			break // stray continuation byte
		} else if b < 0xE0 {
			if i + 2 > len(buf) do break
			if buf[i + 1] & 0xC0 != 0x80 do break
			i += 2
		} else if b < 0xF0 {
			if i + 3 > len(buf) do break
			if buf[i + 1] & 0xC0 != 0x80 do break
			if buf[i + 2] & 0xC0 != 0x80 do break
			i += 3
		} else {
			if i + 4 > len(buf) do break
			if buf[i + 1] & 0xC0 != 0x80 do break
			if buf[i + 2] & 0xC0 != 0x80 do break
			if buf[i + 3] & 0xC0 != 0x80 do break
			i += 4
		}
	}
	return i
}
