/* odin-infer-server — local HTTP inference service (Ollama-compatible API) */

package main

import infer "infer:infer"
import sampler "sampler:sampler"
import tokenizer "tokenizer:tokenizer"

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
	fmt.eprintln("")
	fmt.eprintln("Example:")
	fmt.eprintln("  ./odin-infer-server Qwen3-0.6B-Q4_K_M.gguf --port 11435 -g 1")
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
				if val, ok := strconv.parse_int(args[i + 1]); ok {
					max_ctx = val
				}
			case 'g':
				use_metal = args[i + 1] == "1"
			case 'j':
				if val, ok := strconv.parse_int(args[i + 1]); ok {
					num_threads = val
				}
			case 't':
				if val, ok := strconv.parse_f64(args[i + 1]); ok {
					default_temp = f32(val)
				}
			case 'p':
				if val, ok := strconv.parse_f64(args[i + 1]); ok {
					default_topp = f32(val)
				}
			case:
				error_usage()
			}
			i += 2
			continue
		}
		error_usage()
	}

	fmt.printf("Loading model: %s\n", model_path)
	engine, _ := infer.engine_load(model_path, infer.Engine_Opts{
		max_ctx     = max_ctx,
		use_metal   = use_metal,
		num_threads = num_threads,
	})
	defer infer.engine_destroy(&engine)

	if infer.engine_metal_ready(&engine) {
		fmt.println("Metal GPU: enabled")
	} else {
		fmt.println("Metal GPU: disabled (CPU)")
	}

	tok: tokenizer.Tokenizer
	tokenizer.build_tokenizer(&tok)
	defer tokenizer.free_tokenizer(&tok)
	if !tokenizer.verify_tokenizer(&tok) {
		fmt.eprintln("Tokenizer self-check failed.")
		os.exit(1)
	}

	cfg := infer.engine_config(&engine)
	samp: sampler.Sampler
	sampler.build_sampler(&samp, int(cfg.vocab_size), default_temp, default_topp, u64(time.time_to_unix(time.now())))
	defer sampler.free_sampler(&samp)

	mu: sync.Mutex
	state := Gen_State{
		engine     = &engine,
		tok        = &tok,
		samp       = &samp,
		mu         = &mu,
		model_name = model_name,
	}
	state.model_path = strings.clone(model_path)
	defer delete(state.model_path)

	if !run_server(&state, host, port) {
		os.exit(1)
	}

	infer.destroy_matmul_pool()
}
