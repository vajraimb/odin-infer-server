/* odin-infer-server — local HTTP inference service (Ollama-compatible API).
   Auto-detects Qwen3 vs Qwen3.5 (Ornith) from the GGUF and dispatches to the
   matching engine + tokenizer. */

package main

import infer "infer:infer"
import q35 "qwen3_5:qwen3_5"
import sampler "sampler:sampler"
import tokenizer "tokenizer:tokenizer"
import tok35 "qwen3_5_tokenizer:qwen3_5_tokenizer"

import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strconv"
import "core:strings"
import "core:sync"
import "core:time"

error_usage :: proc() {
	fmt.eprintln("Usage: odin-infer-server <GGUF model> [options]")
	fmt.eprintln("")
	fmt.eprintln("Options:")
	fmt.eprintln("  --host <addr>   bind address (default 127.0.0.1)")
	fmt.eprintln("  --port <int>    port (default 11435)")
	fmt.eprintln("  --name <str>    model name in API responses (default: filename)")
	fmt.eprintln("  -c <int>        max context length (default 4096)")
	fmt.eprintln("  -g <int>        Metal GPU: 0=off, 1=on (default 1 on macOS)")
	fmt.eprintln("  -j <int>        matmul threads (default: CPU cores)")
	fmt.eprintln("  -t <float>      default temperature (default 0.6)")
	fmt.eprintln("  -p <float>      default top-p (default 0.95)")
	fmt.eprintln("  -x <float>      repetition penalty, 1.0 = off (default); try 1.1-1.3")
	fmt.eprintln("")
	fmt.eprintln("Examples:")
	fmt.eprintln("  ./odin-infer-server Qwen3-0.6B-Q4_K_M.gguf --port 11435 -g 1")
	fmt.eprintln("  ./odin-infer-server ornith-1.0-9b-Q4_K_M.gguf -g 1 -c 4096 -x 1.15")
	fmt.eprintln("Auto-detects Qwen3 vs Qwen3.5 (Ornith) from the GGUF architecture.")
	os.exit(1)
}

main :: proc() {
	args := os.args
	if len(args) < 2 {
		error_usage()
	}

	model_path := args[1]
	host := "127.0.0.1"
	port := 11435
	model_name := filepath.base(model_path)
	max_ctx := infer.DEFAULT_MAX_CONTEXT
	use_metal := true
	num_threads := os.get_processor_core_count()
	default_temp: f32 = 0.6
	default_topp: f32 = 0.95
	rep_penalty: f32 = 1.0

	i := 2
	for i < len(args) {
		arg := args[i]
		if arg == "--host" && i + 1 < len(args) {
			host = args[i + 1]
			i += 2
			continue
		}
		if arg == "--port" && i + 1 < len(args) {
			if val, ok := strconv.parse_int(args[i + 1]); ok {
				port = val
			}
			i += 2
			continue
		}
		if arg == "--name" && i + 1 < len(args) {
			model_name = args[i + 1]
			i += 2
			continue
		}
		if len(arg) == 2 && arg[0] == '-' {
			if i + 1 >= len(args) do error_usage()
			switch arg[1] {
			case 'c':
				if val, ok := strconv.parse_int(args[i + 1]); ok { max_ctx = val }
			case 'g':
				use_metal = args[i + 1] == "1"
			case 'j':
				if val, ok := strconv.parse_int(args[i + 1]); ok { num_threads = val }
			case 't':
				if val, ok := strconv.parse_f64(args[i + 1]); ok { default_temp = f32(val) }
			case 'p':
				if val, ok := strconv.parse_f64(args[i + 1]); ok { default_topp = f32(val) }
			case 'x':
				if val, ok := strconv.parse_f64(args[i + 1]); ok { rep_penalty = f32(val) }
			case:
				error_usage()
			}
			i += 2
			continue
		}
		error_usage()
	}

	if rep_penalty < 1.0 do rep_penalty = 1.0

	kind := detect_arch(model_path)
	fmt.printf("Loading model: %s (architecture: %s)\n", model_path, kind == .Qwen3_5 ? "qwen3_5 (Ornith)" : "qwen3")

	mu: sync.Mutex
	samp: sampler.Sampler
	state := Gen_State{
		kind = kind,
		mu = &mu,
		samp = &samp,
		model_name = model_name,
		rep_penalty = rep_penalty,
	}
	state.cache_tokens = make([dynamic]int, 0, 4096)
	defer delete(state.cache_tokens)
	state.model_path = strings.clone(model_path)
	defer delete(state.model_path)

	// Load the matching engine + tokenizer for the detected architecture.
	if kind == .Qwen3_5 {
		qe: q35.Engine
		defer q35.engine_destroy(&qe)
		qe, _ = q35.engine_load(model_path, q35.Engine_Opts{max_ctx = max_ctx, use_metal = use_metal, num_threads = num_threads})
		state.engine_q35 = &qe

		t: tok35.Tokenizer
		defer tok35.free_tokenizer(&t)
		tok35.build_tokenizer(&t)
		state.tok_q35 = &t

		if q35.metal_ready() {
			fmt.println("Metal GPU: enabled")
		} else {
			fmt.println("Metal GPU: disabled (CPU)")
		}
		cfg := q35.engine_config(&qe)
		sampler.build_sampler(&samp, int(cfg.vocab_size), default_temp, default_topp, u64(time.time_to_unix(time.now())))
		defer sampler.free_sampler(&samp)
		if rep_penalty > 1.0 {
			sampler.enable_repeat_penalty(&samp, rep_penalty)
		}

		fmt.printf("think=%s rep=%s T=%.2f P=%.2f\n", "server-controlled", rep_penalty > 1.0 ? "on" : "off", default_temp, default_topp)
		if !run_server(&state, host, port) {
			os.exit(1)
		}
		q35.destroy_matmul_pool()
	} else {
		qe: infer.Engine
		defer infer.engine_destroy(&qe)
		qe, _ = infer.engine_load(model_path, infer.Engine_Opts{max_ctx = max_ctx, use_metal = use_metal, num_threads = num_threads})
		state.engine_q3 = &qe

		t: tokenizer.Tokenizer
		defer tokenizer.free_tokenizer(&t)
		tokenizer.build_tokenizer(&t)
		state.tok_q3 = &t
		if !tokenizer.verify_tokenizer(&t) {
			fmt.eprintln("Tokenizer self-check failed.")
			os.exit(1)
		}

		if infer.engine_metal_ready(&qe) {
			fmt.println("Metal GPU: enabled")
		} else {
			fmt.println("Metal GPU: disabled (CPU)")
		}
		cfg := infer.engine_config(&qe)
		sampler.build_sampler(&samp, int(cfg.vocab_size), default_temp, default_topp, u64(time.time_to_unix(time.now())))
		defer sampler.free_sampler(&samp)
		if rep_penalty > 1.0 {
			sampler.enable_repeat_penalty(&samp, rep_penalty)
		}

		fmt.printf("rep=%s T=%.2f P=%.2f\n", rep_penalty > 1.0 ? "on" : "off", default_temp, default_topp)
		if !run_server(&state, host, port) {
			os.exit(1)
		}
		infer.destroy_matmul_pool()
	}
}
