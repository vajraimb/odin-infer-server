/* Minimal HTTP/1.1 server for local inference */

package main

import "core:fmt"
import "core:net"
import "core:strings"

HTTP_Request :: struct {
	method:  string,
	path:    string,
	headers: string,
	body:    string,
}

send_all :: proc(sock: net.TCP_Socket, data: []u8) -> bool {
	off := 0
	for off < len(data) {
		n, err := net.send_tcp(sock, data[off:])
		if err != nil do return false
		off += n
	}
	return true
}

http_respond :: proc(sock: net.TCP_Socket, status: int, content_type, body: string) {
	status_text := "OK"
	switch status {
	case 400:
		status_text = "Bad Request"
	case 404:
		status_text = "Not Found"
	case 405:
		status_text = "Method Not Allowed"
	case 500:
		status_text = "Internal Server Error"
	}
	header := fmt.tprintf(
		"HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %d\r\nConnection: close\r\n\r\n",
		status,
		status_text,
		content_type,
		len(body),
	)
	_ = send_all(sock, transmute([]u8)header)
	_ = send_all(sock, transmute([]u8)body)
}

http_respond_headers :: proc(sock: net.TCP_Socket, status: int, content_type: string) {
	status_text := status == 200 ? "OK" : "Error"
	header := fmt.tprintf(
		"HTTP/1.1 %d %s\r\nContent-Type: %s\r\nConnection: close\r\n\r\n",
		status,
		status_text,
		content_type,
	)
	send_all(sock, transmute([]u8)header)
}

http_write_ndjson :: proc(sock: net.TCP_Socket, line: string) {
	send_all(sock, transmute([]u8)line)
	send_all(sock, []u8{'\n'})
}

read_http_request :: proc(sock: net.TCP_Socket, allocator := context.allocator) -> (HTTP_Request, bool) {
	buf: [65536]u8
	total := 0

	for total < len(buf) {
		n, err := net.recv_tcp(sock, buf[total:])
		if err != nil || n == 0 do break
		total += n
		s := string(buf[:total])
		if strings.contains(s, "\r\n\r\n") do break
	}

	if total == 0 do return {}, false

	raw := string(buf[:total])
	header_end := strings.index(raw, "\r\n\r\n")
	if header_end < 0 do return {}, false

	header_part := raw[:header_end]
	body_part := raw[header_end + 4:]

	lines := strings.split_lines(header_part, context.temp_allocator)
	if len(lines) == 0 do return {}, false

	parts := strings.split(lines[0], " ", context.temp_allocator)
	if len(parts) < 2 do return {}, false

	method := parts[0]
	path := parts[1]
	headers := strings.join(lines[1:], "\n", context.temp_allocator)

	content_len := parse_content_length(headers)
	body := body_part
	if content_len > len(body) {
		remaining := content_len - len(body)
		extra := make([]u8, remaining, allocator)
		defer delete(extra)
		got := 0
		for got < remaining {
			n, err := net.recv_tcp(sock, extra[got:])
			if err != nil || n == 0 do break
			got += n
		}
		body = strings.concatenate([]string{body, string(extra[:got])}, allocator)
	}

	return HTTP_Request{
		method  = strings.clone(method, allocator),
		path    = strings.clone(path, allocator),
		headers = strings.clone(headers, allocator),
		body    = strings.clone(body, allocator),
	}, true
}

route_request :: proc(state: ^Gen_State, req: ^HTTP_Request, sock: net.TCP_Socket) {
	switch req.method {
	case "GET":
		switch req.path {
		case "/", "":
			handle_root(state, sock)
		case "/api/tags":
			handle_tags(state, sock)
		case "/api/version":
			handle_version(sock)
		case "/v1/models":
			handle_v1_models(state, sock)
		case "/health":
			handle_health(sock)
		case:
			http_respond(sock, 404, "application/json", `{"error":"not found"}`)
		}
	case "POST":
		switch req.path {
		case "/api/generate":
			handle_generate(state, req.body, sock)
		case "/api/chat":
			handle_chat(state, req.body, sock)
		case "/api/show":
			handle_show(state, sock)
		case "/v1/chat/completions":
			handle_v1_chat(state, req.body, sock)
		case:
			http_respond(sock, 404, "application/json", `{"error":"not found"}`)
		}
	case:
		http_respond(sock, 405, "application/json", `{"error":"method not allowed"}`)
	}
}

handle_connection :: proc(state: ^Gen_State, client: net.TCP_Socket) {
	defer net.close(client)

	req, ok := read_http_request(client)
	if !ok {
		http_respond(client, 400, "application/json", `{"error":"bad request"}`)
		return
	}
	defer {
		delete(req.method)
		delete(req.path)
		delete(req.headers)
		delete(req.body)
	}

	route_request(state, &req, client)
}

run_server :: proc(state: ^Gen_State, host: string, port: int) -> bool {
	addr: net.Address = net.IP4_Loopback
	if host == "0.0.0.0" {
		addr = net.IP4_Any
	}

	listener, err := net.listen_tcp(net.Endpoint{address = addr, port = port})
	if err != nil {
		fmt.eprintf("listen failed: %v\n", err)
		return false
	}
	defer net.close(listener)

	fmt.printf("odin-infer-server listening on http://%s:%d\n", host, port)
	fmt.printf("model: %s\n", state.model_name)

	for {
		client, _, acc_err := net.accept_tcp(listener)
		if acc_err != nil {
			fmt.eprintf("accept failed: %v\n", acc_err)
			continue
		}
		handle_connection(state, client)
	}
}
