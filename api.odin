/* Ollama-compatible HTTP API handlers */

package main

import "core:encoding/json"
import "core:fmt"
import "core:net"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"

Chat_Message :: struct {
	role:       string            `json:"role"`,
	content:    string            `json:"content,omitempty"`,
	tool_calls: []Tool_Call_Entry `json:"tool_calls,omitempty"`,
}

Tool_Function :: struct {
	name:        string     `json:"name"`,
	description: string     `json:"description"`,
	parameters:  json.Value `json:"parameters"`,
}

Tool_Def :: struct {
	type:     string        `json:"type"`,
	function: Tool_Function `json:"function"`,
}

Tool_Call_Function :: struct {
	name:      string     `json:"name"`,
	arguments: json.Value `json:"arguments"`,
}

Tool_Call_Entry :: struct {
	index:    int                `json:"index"`,
	id:       string             `json:"id"`,
	function: Tool_Call_Function `json:"function"`,
}

Generate_Options :: struct {
	temperature: f32 `json:"temperature"`,
	top_p:       f32 `json:"top_p"`,
	num_predict: int   `json:"num_predict"`,
	seed:        int   `json:"seed"`,
}

Generate_Request :: struct {
	model:   string           `json:"model"`,
	prompt:  string           `json:"prompt"`,
	stream:  bool             `json:"stream"`,
	options: Generate_Options `json:"options"`,
}

Chat_Request :: struct {
	model:    string           `json:"model"`,
	messages: []Chat_Message   `json:"messages"`,
	stream:   bool             `json:"stream"`,
	tools:    []Tool_Def       `json:"tools,omitempty"`,
	options:  Generate_Options `json:"options"`,
}

Generate_Response :: struct {
	model:              string `json:"model"`,
	created_at:         string `json:"created_at"`,
	response:           string `json:"response,omitempty"`,
	done:               bool   `json:"done"`,
	prompt_eval_count:  int    `json:"prompt_eval_count,omitempty"`,
	eval_count:         int    `json:"eval_count,omitempty"`,
}

Chat_Response :: struct {
	model:             string       `json:"model"`,
	created_at:        string       `json:"created_at"`,
	message:           Chat_Message `json:"message,omitempty"`,
	done:              bool         `json:"done"`,
	done_reason:       string       `json:"done_reason,omitempty"`,
	prompt_eval_count: int          `json:"prompt_eval_count,omitempty"`,
	eval_count:        int          `json:"eval_count,omitempty"`,
}

Tags_Response :: struct {
	models: []Model_Info `json:"models"`,
}

Model_Info :: struct {
	name:        string `json:"name"`,
	model:       string `json:"model"`,
	modified_at: string `json:"modified_at"`,
	size:        i64    `json:"size"`,
	digest:      string `json:"digest"`,
}

Version_Response :: struct {
	version: string `json:"version"`,
}

now_rfc3339 :: proc() -> string {
	t := time.now()
	y := time.year(t)
	m := int(time.month(t))
	d := time.day(t)
	h, min, s := time.clock(t)
	return fmt.tprintf("%04d-%02d-%02dT%02d:%02d:%02dZ", y, m, d, h, min, s)
}

json_marshal :: proc(v: any, allocator := context.allocator) -> (string, bool) {
	b, err := json.marshal(v, allocator = allocator)
	if err != nil do return "", false
	return string(b), true
}

parse_gen_options :: proc(req_options: Generate_Options, defaults: Gen_Options) -> Gen_Options {
	opts := defaults
	if req_options.temperature > 0 {
		opts.temperature = req_options.temperature
	}
	if req_options.top_p > 0 {
		opts.top_p = req_options.top_p
	}
	if req_options.num_predict > 0 {
		opts.max_tokens = req_options.num_predict
	}
	return opts
}

Stream_Writer :: struct {
	sock:        net.TCP_Socket,
	model:       string,
	tool_calls:  []Tool_Call_Entry,
	done_reason: string,
}

stream_write_chunk :: proc(chunk: string, done: bool, user_data: rawptr) {
	w := (^Stream_Writer)(user_data)
	resp: Generate_Response
	resp.model = w.model
	resp.created_at = now_rfc3339()
	resp.response = chunk
	resp.done = done
	if body, ok := json_marshal(resp); ok {
		http_write_ndjson(w.sock, body)
	}
}

stream_write_chat_chunk :: proc(chunk: string, done: bool, user_data: rawptr) {
	w := (^Stream_Writer)(user_data)
	resp: Chat_Response
	resp.model = w.model
	resp.created_at = now_rfc3339()
	resp.message = Chat_Message{role = "assistant", content = chunk}
	resp.done = done
	if done {
		resp.done_reason = w.done_reason
		if len(w.tool_calls) > 0 {
			resp.message.tool_calls = w.tool_calls
		}
	}
	if body, ok := json_marshal(resp); ok {
		http_write_ndjson(w.sock, body)
		delete(body)
	}
}

handle_generate :: proc(state: ^Gen_State, body: string, sock: net.TCP_Socket) {
	req: Generate_Request
	if err := json.unmarshal(transmute([]u8)body, &req); err != nil {
		http_respond(sock, 400, "application/json", `{"error":"invalid JSON"}`)
		return
	}
	if len(req.prompt) == 0 {
		http_respond(sock, 400, "application/json", `{"error":"prompt required"}`)
		return
	}

	opts := parse_gen_options(req.options, Gen_Options{
		temperature = 0.6,
		top_p       = 0.95,
		max_tokens  = 256,
	})

	if req.stream {
		http_respond_headers(sock, 200, "application/x-ndjson")
		writer := Stream_Writer{sock = sock, model = state.model_name}
		text, n_prompt, n_gen, ok := generate_tokens(
			state,
			req.prompt,
			opts,
			stream_write_chunk,
			&writer,
		)
		delete(text)
		if !ok {
			http_write_ndjson(sock, `{"error":"generation failed"}`)
		} else {
			_ = n_prompt
			_ = n_gen
		}
		return
	}

	text, n_prompt, n_gen, ok := generate_tokens(state, req.prompt, opts)
	if !ok {
		http_respond(sock, 500, "application/json", `{"error":"generation failed"}`)
		return
	}
	defer delete(text)

	resp := Generate_Response{
		model             = state.model_name,
		created_at        = now_rfc3339(),
		response          = text,
		done              = true,
		prompt_eval_count = n_prompt,
		eval_count        = n_gen,
	}
	if body_json, ok2 := json_marshal(resp); ok2 {
		http_respond(sock, 200, "application/json", body_json)
		delete(body_json)
	} else {
		http_respond(sock, 500, "application/json", `{"error":"marshal failed"}`)
	}
}

handle_chat :: proc(state: ^Gen_State, body: string, sock: net.TCP_Socket) {
	req: Chat_Request
	if err := json.unmarshal(transmute([]u8)body, &req); err != nil {
		http_respond(sock, 400, "application/json", `{"error":"invalid JSON"}`)
		return
	}
	if len(req.messages) == 0 {
		http_respond(sock, 400, "application/json", `{"error":"messages required"}`)
		return
	}

	defer {
		for t in req.tools {
			json.destroy_value(t.function.parameters)
		}
		if req.tools != nil do delete(req.tools)
	}

	prompt := build_chat_prompt(req.messages, req.tools, false)
	defer delete(prompt)

	opts := parse_gen_options(req.options, Gen_Options{
		temperature = 0.6,
		top_p       = 0.95,
		max_tokens  = 256,
	})

	has_tools := len(req.tools) > 0

	if req.stream {
		http_respond_headers(sock, 200, "application/x-ndjson")

		if has_tools {
			text, _, _, gen_ok := generate_tokens(state, prompt, opts)
			if !gen_ok {
				http_write_ndjson(sock, `{"error":"generation failed"}`)
				return
			}
			clean_text, tool_calls := extract_tool_calls(text)
			delete(text)

			writer := Stream_Writer{
				sock        = sock,
				model       = state.model_name,
				tool_calls  = tool_calls,
				done_reason = len(tool_calls) > 0 ? "tool_calls" : "stop",
			}
			if len(clean_text) > 0 {
				stream_write_chat_chunk(clean_text, false, &writer)
			}
			stream_write_chat_chunk("", true, &writer)

			delete(clean_text)
			free_tool_calls(tool_calls)
		} else {
			writer := Stream_Writer{sock = sock, model = state.model_name, done_reason = "stop"}
			text, _, _, gen_ok := generate_tokens(
				state,
				prompt,
				opts,
				stream_write_chat_chunk,
				&writer,
			)
			delete(text)
			if !gen_ok {
				http_write_ndjson(sock, `{"error":"generation failed"}`)
			}
		}
		return
	}

	text, n_prompt, n_gen, ok := generate_tokens(state, prompt, opts)
	if !ok {
		http_respond(sock, 500, "application/json", `{"error":"generation failed"}`)
		return
	}

	clean_text := text
	tool_calls: []Tool_Call_Entry = nil
	if has_tools {
		clean_text, tool_calls = extract_tool_calls(text)
		delete(text)
	}

	done_reason := len(tool_calls) > 0 ? "tool_calls" : "stop"
	resp := Chat_Response{
		model      = state.model_name,
		created_at = now_rfc3339(),
		message = Chat_Message{
			role       = "assistant",
			content    = clean_text,
			tool_calls = tool_calls,
		},
		done              = true,
		done_reason       = done_reason,
		prompt_eval_count = n_prompt,
		eval_count        = n_gen,
	}

	if body_json, ok2 := json_marshal(resp); ok2 {
		http_respond(sock, 200, "application/json", body_json)
		delete(body_json)
	} else {
		http_respond(sock, 500, "application/json", `{"error":"marshal failed"}`)
	}

	delete(clean_text)
	free_tool_calls(tool_calls)
}

handle_tags :: proc(state: ^Gen_State, sock: net.TCP_Socket) {
	size: i64 = 0
	if st, err := os.stat(state.model_path, context.temp_allocator); err == os.ERROR_NONE {
		size = st.size
	}
	resp := Tags_Response{
		models = {
			{
				name        = state.model_name,
				model       = state.model_name,
				modified_at = now_rfc3339(),
				size        = size,
				digest      = "sha256:local",
			},
		},
	}
	if body, ok := json_marshal(resp); ok {
		http_respond(sock, 200, "application/json", body)
		delete(body)
	} else {
		http_respond(sock, 500, "application/json", `{"error":"marshal failed"}`)
	}
}

handle_version :: proc(sock: net.TCP_Socket) {
	resp := Version_Response{version = "0.1.0"}
	if body, ok := json_marshal(resp); ok {
		http_respond(sock, 200, "application/json", body)
		delete(body)
	} else {
		http_respond(sock, 500, "application/json", `{"error":"marshal failed"}`)
	}
}

handle_root :: proc(state: ^Gen_State, sock: net.TCP_Socket) {
	body := fmt.tprintf(
		"odin-infer-server\nmodel: %s\nendpoints: GET /api/tags, GET /api/version, POST /api/generate, POST /api/chat\n",
		state.model_name,
	)
	http_respond(sock, 200, "text/plain; charset=utf-8", body)
}

handle_health :: proc(sock: net.TCP_Socket) {
	http_respond(sock, 200, "application/json", `{"status":"ok"}`)
}

parse_content_length :: proc(headers: string) -> int {
	for line in strings.split_lines(headers, context.temp_allocator) {
		lower := strings.to_lower(line, context.temp_allocator)
		if strings.has_prefix(lower, "content-length:") {
			val := strings.trim_space(line[strings.index(line, ":") + 1:])
			if n, ok := strconv.parse_int(val); ok {
				return n
			}
		}
	}
	return 0
}
