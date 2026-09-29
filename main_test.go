package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"
)

// The stub upstream reports these token counts for every request, whichever
// dialect was used, so tests can assert on exact counter deltas.
const (
	stubPromptTokens    = 11
	stubGeneratedTokens = 7
	stubEvalDurationNs  = 700000000
	stubStreamedContent = 2 // content chunks per streamed response
	stubFirstChunkDelay = 20 * time.Millisecond
	stubBetweenChunkGap = 10 * time.Millisecond
)

// recordedRequest is what the stub upstream saw, so tests can assert on how the
// proxy rewrote the request.
type recordedRequest struct {
	path          string
	body          []byte
	contentLength int64
	header        http.Header
}

type stubUpstream struct {
	*httptest.Server
	mu       sync.Mutex
	requests []recordedRequest
}

func (s *stubUpstream) lastRequest(t *testing.T, path string) recordedRequest {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(s.requests) - 1; i >= 0; i-- {
		if s.requests[i].path == path {
			return s.requests[i]
		}
	}
	t.Fatalf("upstream never received a request for %s", path)
	return recordedRequest{}
}

// sseEvents are the streamed chunks of an OpenAI-compatible response, in order.
// The usage chunk is the one Ollama only sends when include_usage is set.
func sseEvents(model string, includeUsage bool) []string {
	events := []string{
		fmt.Sprintf(`{"id":"1","object":"chat.completion.chunk","model":%q,"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`, model),
		fmt.Sprintf(`{"id":"1","object":"chat.completion.chunk","model":%q,"choices":[{"index":0,"delta":{"content":"Hi"}}]}`, model),
		fmt.Sprintf(`{"id":"1","object":"chat.completion.chunk","model":%q,"choices":[{"index":0,"delta":{"content":" there"}}]}`, model),
		fmt.Sprintf(`{"id":"1","object":"chat.completion.chunk","model":%q,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`, model),
	}
	if includeUsage {
		events = append(events, fmt.Sprintf(
			`{"id":"1","object":"chat.completion.chunk","model":%q,"choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
			model, stubPromptTokens, stubGeneratedTokens, stubPromptTokens+stubGeneratedTokens))
	}
	return append(events, "[DONE]")
}

// sseBody renders the events as they appear on the wire.
func sseBody(model string, includeUsage bool) string {
	var b strings.Builder
	for _, e := range sseEvents(model, includeUsage) {
		fmt.Fprintf(&b, "data: %s\n\n", e)
	}
	return b.String()
}

// nativeChatBody is the NDJSON stream of a native /api/chat response.
func nativeChatBody(model string) string {
	return strings.Join([]string{
		fmt.Sprintf(`{"model":%q,"message":{"role":"assistant","content":"Hi"},"done":false}`, model),
		fmt.Sprintf(`{"model":%q,"message":{"role":"assistant","content":" there"},"done":false}`, model),
		fmt.Sprintf(`{"model":%q,"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":%d,"eval_count":%d,"eval_duration":%d,"total_duration":1200000000}`,
			model, stubPromptTokens, stubGeneratedTokens, stubEvalDurationNs),
	}, "\n") + "\n"
}

// anthropicStream is a /v1/messages stream as Ollama sends it: message_start
// announces a placeholder input count, and message_delta carries the real one.
func anthropicStream(model string) string {
	events := []struct{ name, data string }{
		{"message_start", fmt.Sprintf(`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":%q,"content":[],"usage":{"input_tokens":2,"output_tokens":0}}}`, model)},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", fmt.Sprintf(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":%d,"cache_read_input_tokens":0,"output_tokens":%d}}`, stubPromptTokens, stubGeneratedTokens)},
		{"message_stop", `{"type":"message_stop"}`},
	}
	var b strings.Builder
	for _, e := range events {
		fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", e.name, e.data)
	}
	return b.String()
}

// anthropicBody is a non-streamed /v1/messages reply. Part of the prompt was a
// cache hit, which input_tokens leaves out.
func anthropicBody(model string) string {
	return fmt.Sprintf(`{"id":"msg_1","type":"message","role":"assistant","model":%q,"content":[{"type":"text","text":"Hi there"}],"usage":{"input_tokens":%d,"cache_read_input_tokens":%d,"output_tokens":%d}}`,
		model, stubPromptTokens-6, 6, stubGeneratedTokens)
}

func newStubUpstream(t *testing.T) *stubUpstream {
	t.Helper()
	stub := &stubUpstream{}

	mux := http.NewServeMux()

	mux.HandleFunc("/api/ps", func(w http.ResponseWriter, r *http.Request) {
		stub.record(r)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"models":[{"name":"stub-loaded","size":1048576,"processor":"cpu"}]}`)
	})

	mux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		body := stub.record(r)
		model, _ := requestField(body, "model").(string)
		w.Header().Set("Content-Type", "application/x-ndjson")
		writeChunked(w, strings.SplitAfter(nativeChatBody(model), "\n"))
	})

	completions := func(w http.ResponseWriter, r *http.Request) {
		body := stub.record(r)
		model, _ := requestField(body, "model").(string)
		stream, _ := requestField(body, "stream").(bool)

		gzipReply := strings.Contains(r.Header.Get("Accept-Encoding"), "gzip")

		if !stream {
			var payload string
			if r.URL.Path == "/v1/completions" {
				payload = fmt.Sprintf(`{"id":"1","object":"text_completion","model":%q,"choices":[{"index":0,"text":"Hi there","finish_reason":"stop"}],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
					model, stubPromptTokens, stubGeneratedTokens, stubPromptTokens+stubGeneratedTokens)
			} else {
				payload = fmt.Sprintf(`{"id":"1","object":"chat.completion","model":%q,"choices":[{"index":0,"message":{"role":"assistant","content":"Hi there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
					model, stubPromptTokens, stubGeneratedTokens, stubPromptTokens+stubGeneratedTokens)
			}
			w.Header().Set("Content-Type", "application/json")
			if gzipReply {
				// Mirrors real upstreams: compressed only because the request
				// asked for it. If the proxy forwards a client's
				// Accept-Encoding, Go stops decompressing for it and parsing
				// silently yields nothing.
				w.Header().Set("Content-Encoding", "gzip")
				zw := gzip.NewWriter(w)
				io.WriteString(zw, payload)
				zw.Close()
				return
			}
			io.WriteString(w, payload)
			return
		}

		includeUsage := false
		if opts, ok := requestField(body, "stream_options").(map[string]interface{}); ok {
			includeUsage, _ = opts["include_usage"].(bool)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		var events []string
		for _, e := range sseEvents(model, includeUsage) {
			events = append(events, fmt.Sprintf("data: %s\n\n", e))
		}
		writeChunked(w, events)
	}
	mux.HandleFunc("/v1/chat/completions", completions)
	mux.HandleFunc("/v1/completions", completions)

	mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		body := stub.record(r)
		model, _ := requestField(body, "model").(string)
		if stream, _ := requestField(body, "stream").(bool); !stream {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, anthropicBody(model))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		writeChunked(w, strings.SplitAfter(anthropicStream(model), "\n\n"))
	})

	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		stub.record(r)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"models":[{"name":"stub-loaded","model":"stub-loaded","size_vram":1048576}]}`)
	})

	stub.Server = httptest.NewServer(mux)
	t.Cleanup(stub.Close)
	return stub
}

func (s *stubUpstream) record(r *http.Request) []byte {
	body, _ := io.ReadAll(r.Body)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, recordedRequest{
		path:          r.URL.Path,
		body:          body,
		contentLength: r.ContentLength,
		header:        r.Header.Clone(),
	})
	return body
}

// writeChunked writes each piece separately and flushes, so the proxy sees a
// real stream arriving over time rather than one buffered write.
func writeChunked(w http.ResponseWriter, pieces []string) {
	flusher, _ := w.(http.Flusher)
	first := true
	for _, p := range pieces {
		if p == "" {
			continue
		}
		if first {
			time.Sleep(stubFirstChunkDelay)
			first = false
		} else {
			time.Sleep(stubBetweenChunkGap)
		}
		io.WriteString(w, p)
		if flusher != nil {
			flusher.Flush()
		}
	}
}

func requestField(body []byte, key string) interface{} {
	var parsed map[string]interface{}
	if json.Unmarshal(body, &parsed) != nil {
		return nil
	}
	return parsed[key]
}

// newProxy starts the sidecar in front of a stub upstream.
func newProxy(t *testing.T) (*httptest.Server, *stubUpstream) {
	t.Helper()
	stub := newStubUpstream(t)
	proxy := httptest.NewServer(newMux(stub.URL))
	t.Cleanup(proxy.Close)
	return proxy, stub
}

func newProxyWithOverrides(t *testing.T, overrides map[string]interface{}) (*httptest.Server, *stubUpstream) {
	t.Helper()
	stub := newStubUpstream(t)
	proxy := httptest.NewServer(newMuxWithOverrides(stub.URL, overrides))
	t.Cleanup(proxy.Close)
	return proxy, stub
}

func newProxyWithConfig(t *testing.T, defaults, overrides map[string]interface{}, sanitizeUTF8Responses bool) (*httptest.Server, *stubUpstream) {
	t.Helper()
	stub := newStubUpstream(t)
	proxy := httptest.NewServer(newMuxWithConfig(stub.URL, proxyConfig{
		requestDefaults:       defaults,
		requestOverrides:      overrides,
		sanitizeUTF8Responses: sanitizeUTF8Responses,
		promoteReasoning:      true,
	}))
	t.Cleanup(proxy.Close)
	return proxy, stub
}

func TestRequestOverridesConfiguration(t *testing.T) {
	t.Run("absent", func(t *testing.T) {
		t.Setenv("OLLAMA_PROXY_REQUEST_OVERRIDES", "")
		got, err := parseRequestOverrides()
		if err != nil || len(got) != 0 {
			t.Fatalf("parseRequestOverrides() = %#v, %v; want empty object", got, err)
		}
	})
	t.Run("valid object", func(t *testing.T) {
		t.Setenv("OLLAMA_PROXY_REQUEST_OVERRIDES", `{"reasoning_effort":"none","temperature":0}`)
		got, err := parseRequestOverrides()
		if err != nil || got["reasoning_effort"] != "none" {
			t.Fatalf("parseRequestOverrides() = %#v, %v", got, err)
		}
	})
	for _, value := range []string{"not json", `[]`, `null`} {
		t.Run("invalid "+value, func(t *testing.T) {
			t.Setenv("OLLAMA_PROXY_REQUEST_OVERRIDES", value)
			if _, err := parseRequestOverrides(); err == nil {
				t.Fatal("parseRequestOverrides() succeeded; want an error")
			}
		})
	}
}

func TestRequestDefaultsConfiguration(t *testing.T) {
	t.Setenv("OLLAMA_PROXY_REQUEST_DEFAULTS", `{"reasoning_effort":"none"}`)
	got, err := parseRequestDefaults()
	if err != nil || got["reasoning_effort"] != "none" {
		t.Fatalf("parseRequestDefaults() = %#v, %v", got, err)
	}
}

func TestRequestOverridesClientValuesAndFields(t *testing.T) {
	proxy, stub := newProxyWithOverrides(t, map[string]interface{}{
		"reasoning_effort": "none",
		"temperature":      float64(0),
	})
	body := `{"model":"override-model","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function"}],"stream":false,"temperature":0.9}`
	post(t, proxy, "/v1/chat/completions", body, http.Header{"X-Test": []string{"preserve-me"}})
	up := stub.lastRequest(t, "/v1/chat/completions")
	var got map[string]interface{}
	if err := json.Unmarshal(up.body, &got); err != nil {
		t.Fatal(err)
	}
	if got["reasoning_effort"] != "none" || got["temperature"] != float64(0) {
		t.Errorf("overrides not applied: %s", up.body)
	}
	for _, key := range []string{"model", "messages", "tools", "stream"} {
		if _, ok := got[key]; !ok {
			t.Errorf("unrelated field %q was lost: %s", key, up.body)
		}
	}
	if up.header.Get("X-Test") != "preserve-me" {
		t.Errorf("header was not preserved: %v", up.header)
	}
}

func TestRequestDefaultsYieldToClientValues(t *testing.T) {
	proxy, stub := newProxyWithConfig(t, map[string]interface{}{
		"reasoning_effort": "none",
		"temperature":      float64(0),
	}, map[string]interface{}{}, true)
	body := `{"model":"default-model","messages":[],"temperature":0.9}`
	post(t, proxy, "/v1/chat/completions", body, nil)

	var got map[string]interface{}
	up := stub.lastRequest(t, "/v1/chat/completions")
	if err := json.Unmarshal(up.body, &got); err != nil {
		t.Fatal(err)
	}
	if got["reasoning_effort"] != "none" {
		t.Errorf("default missing from upstream request: %s", up.body)
	}
	if got["temperature"] != float64(0.9) {
		t.Errorf("client value was overwritten by default: %s", up.body)
	}
}

// TestRequestOverridesBeatDefaultsAndClient pins the precedence the two
// variables exist to express: an override wins over everything, and a default
// loses to anything the client sent.
func TestRequestOverridesBeatDefaultsAndClient(t *testing.T) {
	proxy, stub := newProxyWithConfig(t,
		map[string]interface{}{"reasoning_effort": "none", "top_p": float64(0.1)},
		map[string]interface{}{"reasoning_effort": "high"},
		true)
	post(t, proxy, "/v1/chat/completions", `{"model":"m","messages":[],"reasoning_effort":"low","top_p":0.9}`, nil)

	var got map[string]interface{}
	up := stub.lastRequest(t, "/v1/chat/completions")
	if err := json.Unmarshal(up.body, &got); err != nil {
		t.Fatal(err)
	}
	if got["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v, want the override to win: %s", got["reasoning_effort"], up.body)
	}
	if got["top_p"] != float64(0.9) {
		t.Errorf("top_p = %v, want the client value to beat the default: %s", got["top_p"], up.body)
	}
}

func TestMalformedChatRequestIsRejected(t *testing.T) {
	proxy, stub := newProxyWithOverrides(t, map[string]interface{}{"reasoning_effort": "none"})
	resp, err := proxy.Client().Post(proxy.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"messages":`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	stub.mu.Lock()
	defer stub.mu.Unlock()
	for _, request := range stub.requests {
		if request.path == "/v1/chat/completions" {
			t.Fatal("malformed request reached upstream")
		}
	}
}

func TestRequestOverridesDoNotAffectOtherEndpoints(t *testing.T) {
	proxy, stub := newProxyWithOverrides(t, map[string]interface{}{"reasoning_effort": "none"})
	nativeBody := `{"model":"native","messages":[],"stream":false}`
	post(t, proxy, "/api/chat", nativeBody, nil)
	post(t, proxy, "/v1/completions", `{"model":"legacy","prompt":"hi"}`, nil)
	for _, path := range []string{"/api/chat", "/v1/completions"} {
		up := stub.lastRequest(t, path)
		if strings.Contains(string(up.body), "reasoning_effort") {
			t.Errorf("override leaked into %s: %s", path, up.body)
		}
	}
}

// post sends a request through the proxy and returns the client-visible body.
func post(t *testing.T, proxy *httptest.Server, path, body string, header http.Header) string {
	t.Helper()
	got, _ := postFull(t, proxy, path, body, header)
	return got
}

// postFull also returns the client-visible response headers.
func postFull(t *testing.T, proxy *httptest.Server, path, body string, header http.Header) (string, http.Header) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, proxy.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatalf("building request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for name, vals := range header {
		for _, v := range vals {
			req.Header.Add(name, v)
		}
	}
	resp, err := proxy.Client().Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	defer resp.Body.Close()
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading response: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST %s: status %d, body %s", path, resp.StatusCode, got)
	}
	return string(got), resp.Header.Clone()
}

// scrape fetches the proxy's own /metrics output.
func scrape(t *testing.T, proxy *httptest.Server) string {
	t.Helper()
	resp, err := proxy.Client().Get(proxy.URL + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading /metrics: %v", err)
	}
	return string(body)
}

// sample returns the value of the first series whose name matches and whose
// label set contains every given label fragment (e.g. `model="x:latest"`).
func sample(t *testing.T, metrics, name string, labels ...string) (float64, bool) {
	t.Helper()
	for _, line := range strings.Split(metrics, "\n") {
		if !strings.HasPrefix(line, name+"{") {
			continue
		}
		matched := true
		for _, l := range labels {
			if !strings.Contains(line, l) {
				matched = false
				break
			}
		}
		if !matched {
			continue
		}
		fields := strings.Fields(line)
		v, err := strconv.ParseFloat(fields[len(fields)-1], 64)
		if err != nil {
			t.Fatalf("parsing value of %q: %v", line, err)
		}
		return v, true
	}
	return 0, false
}

func mustSample(t *testing.T, metrics, name string, labels ...string) float64 {
	t.Helper()
	v, ok := sample(t, metrics, name, labels...)
	if !ok {
		t.Fatalf("metric %s%v not found in /metrics output", name, labels)
	}
	return v
}

func requireAbsent(t *testing.T, metrics, name string, labels ...string) {
	t.Helper()
	if v, ok := sample(t, metrics, name, labels...); ok {
		t.Fatalf("metric %s%v unexpectedly present with value %v", name, labels, v)
	}
}

// TestNativeChatIsUnchanged pins the pre-existing /api/chat behaviour: the
// NDJSON stream reaches the client verbatim and the counts Ollama reports are
// recorded, now alongside a time-to-first-token observation.
func TestNativeChatIsUnchanged(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "native-chat"
	tagged := model + ":latest"

	got := post(t, proxy, "/api/chat", fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}],"stream":true}`, model), nil)

	if want := nativeChatBody(model); got != want {
		t.Errorf("client saw a modified stream:\n got: %q\nwant: %q", got, want)
	}

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_prompt_tokens_total", `model="`+tagged+`"`); v != stubPromptTokens {
		t.Errorf("prompt tokens = %v, want %v", v, stubPromptTokens)
	}
	if v := mustSample(t, metrics, "ollama_generated_tokens_total", `model="`+tagged+`"`); v != stubGeneratedTokens {
		t.Errorf("generated tokens = %v, want %v", v, stubGeneratedTokens)
	}
	// The endpoint label is normalised and renamed to api_endpoint.
	if v := mustSample(t, metrics, "ollama_request_duration_seconds_count", `api_endpoint="chat"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("request duration count = %v, want 1", v)
	}
	// Ollama reports eval_duration itself, so seconds-per-token is exact.
	wantPerToken := float64(stubEvalDurationNs) / 1e9 / stubGeneratedTokens
	if v := mustSample(t, metrics, "ollama_time_per_token_seconds_sum", `model="`+tagged+`"`); v != wantPerToken {
		t.Errorf("time per token sum = %v, want %v", v, wantPerToken)
	}
	if v := mustSample(t, metrics, "ollama_time_to_first_token_seconds_count", `api_endpoint="chat"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("time to first token count = %v, want 1", v)
	}
}

// TestNativeChatNonStreaming checks that stream:false still yields token counts
// but no time-to-first-token, which only a stream can measure.
func TestNativeChatNonStreaming(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "native-chat-blocking"
	tagged := model + ":latest"

	post(t, proxy, "/api/chat", fmt.Sprintf(`{"model":%q,"messages":[],"stream":false}`, model), nil)

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_generated_tokens_total", `model="`+tagged+`"`); v != stubGeneratedTokens {
		t.Errorf("generated tokens = %v, want %v", v, stubGeneratedTokens)
	}
	requireAbsent(t, metrics, "ollama_time_to_first_token_seconds_count", `model="`+tagged+`"`)
}

// TestOpenAIStreamingInjectsAndHidesUsage is the core of the fix: usage is asked
// for on the client's behalf and the extra chunk never reaches the client.
func TestOpenAIStreamingInjectsAndHidesUsage(t *testing.T) {
	proxy, stub := newProxy(t)
	const model = "v1-stream-injected"
	tagged := model + ":latest"

	got := post(t, proxy, "/v1/chat/completions",
		fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":1000000}`, model), nil)

	// The forwarded request asked for usage even though the client did not.
	up := stub.lastRequest(t, "/v1/chat/completions")
	opts, ok := requestField(up.body, "stream_options").(map[string]interface{})
	if !ok {
		t.Fatalf("upstream request had no stream_options: %s", up.body)
	}
	if include, _ := opts["include_usage"].(bool); !include {
		t.Errorf("stream_options.include_usage not injected: %s", up.body)
	}
	// Injection must not mangle the rest of the body: re-marshalling through
	// interface{} would otherwise turn large integers into 1e+06, which Ollama's
	// integer fields reject.
	if !strings.Contains(string(up.body), `"max_tokens":1000000`) {
		t.Errorf("max_tokens literal was rewritten: %s", up.body)
	}
	if up.contentLength != int64(len(up.body)) {
		t.Errorf("upstream Content-Length = %d, want %d (the rewritten body length)", up.contentLength, len(up.body))
	}

	// The client sees exactly the stream it would have got without injection.
	if want := sseBody(model, false); got != want {
		t.Errorf("client saw the injected usage chunk:\n got: %q\nwant: %q", got, want)
	}
	if strings.Contains(got, "usage") {
		t.Errorf("client-visible stream mentions usage: %q", got)
	}

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_prompt_tokens_total", `model="`+tagged+`"`); v != stubPromptTokens {
		t.Errorf("prompt tokens = %v, want %v", v, stubPromptTokens)
	}
	if v := mustSample(t, metrics, "ollama_generated_tokens_total", `model="`+tagged+`"`); v != stubGeneratedTokens {
		t.Errorf("generated tokens = %v, want %v", v, stubGeneratedTokens)
	}
	if v := mustSample(t, metrics, "ollama_request_duration_seconds_count", `api_endpoint="v1/chat/completions"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("request duration count = %v, want 1", v)
	}
	if v := mustSample(t, metrics, "ollama_time_to_first_token_seconds_count", `api_endpoint="v1/chat/completions"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("time to first token count = %v, want 1", v)
	}
	// Measured across the generation window, so it must be well under the whole
	// request duration rather than derived from it.
	if v := mustSample(t, metrics, "ollama_time_per_token_seconds_count", `model="`+tagged+`"`); v != 1 {
		t.Errorf("time per token count = %v, want 1", v)
	}
	window := mustSample(t, metrics, "ollama_time_per_token_seconds_sum", `model="`+tagged+`"`)
	total := mustSample(t, metrics, "ollama_request_duration_seconds_sum", `api_endpoint="v1/chat/completions"`, `model="`+tagged+`"`)
	if window <= 0 {
		t.Errorf("time per token = %v, want a positive generation-window measurement", window)
	}
	if window*stubGeneratedTokens >= total {
		t.Errorf("time per token (%v) looks derived from total duration (%v)", window, total)
	}
}

// TestOpenAIStreamingKeepsClientRequestedUsage covers the other half: a client
// that asks for usage itself must still receive the chunk.
func TestOpenAIStreamingKeepsClientRequestedUsage(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "v1-stream-requested"
	tagged := model + ":latest"

	got := post(t, proxy, "/v1/chat/completions",
		fmt.Sprintf(`{"model":%q,"messages":[],"stream":true,"stream_options":{"include_usage":true}}`, model), nil)

	if want := sseBody(model, true); got != want {
		t.Errorf("client-requested usage chunk was not forwarded:\n got: %q\nwant: %q", got, want)
	}

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_generated_tokens_total", `model="`+tagged+`"`); v != stubGeneratedTokens {
		t.Errorf("generated tokens = %v, want %v", v, stubGeneratedTokens)
	}
}

// TestOpenAINonStreaming reads usage straight from the body, and deliberately
// records no seconds-per-token: with a large prompt and a short answer, prompt
// evaluation would dominate the request and skew the metric.
func TestOpenAINonStreaming(t *testing.T) {
	proxy, stub := newProxy(t)
	const model = "v1-blocking"
	tagged := model + ":latest"

	got := post(t, proxy, "/v1/chat/completions", fmt.Sprintf(`{"model":%q,"messages":[]}`, model), nil)
	if !strings.Contains(got, `"total_tokens":18`) {
		t.Errorf("non-streaming body was not forwarded intact: %q", got)
	}

	// Nothing to inject when the client is not streaming.
	up := stub.lastRequest(t, "/v1/chat/completions")
	if requestField(up.body, "stream_options") != nil {
		t.Errorf("stream_options injected into a non-streaming request: %s", up.body)
	}

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_prompt_tokens_total", `model="`+tagged+`"`); v != stubPromptTokens {
		t.Errorf("prompt tokens = %v, want %v", v, stubPromptTokens)
	}
	if v := mustSample(t, metrics, "ollama_request_duration_seconds_count", `api_endpoint="v1/chat/completions"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("request duration count = %v, want 1", v)
	}
	requireAbsent(t, metrics, "ollama_time_per_token_seconds_count", `model="`+tagged+`"`)
	requireAbsent(t, metrics, "ollama_time_to_first_token_seconds_count", `model="`+tagged+`"`)
}

// TestOpenAILegacyCompletions checks the second /v1 path, whose streamed content
// lives in choices[].text rather than a delta.
func TestOpenAILegacyCompletions(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "v1-legacy"
	tagged := model + ":latest"

	post(t, proxy, "/v1/completions", fmt.Sprintf(`{"model":%q,"prompt":"hi"}`, model), nil)

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_generated_tokens_total", `model="`+tagged+`"`); v != stubGeneratedTokens {
		t.Errorf("generated tokens = %v, want %v", v, stubGeneratedTokens)
	}
	if v := mustSample(t, metrics, "ollama_request_duration_seconds_count", `api_endpoint="v1/completions"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("request duration count = %v, want 1", v)
	}
}

// TestAnthropicStreaming covers /v1/messages as Claude Code uses it: the stream
// reaches the client verbatim, and the counts come from message_delta rather
// than message_start's placeholder.
func TestAnthropicStreaming(t *testing.T) {
	proxy, stub := newProxy(t)
	const model = "anthropic-stream"
	tagged := model + ":latest"

	got := post(t, proxy, "/v1/messages?beta=true",
		fmt.Sprintf(`{"model":%q,"max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`, model), nil)

	if want := anthropicStream(model); got != want {
		t.Errorf("client saw a modified stream:\n got: %q\nwant: %q", got, want)
	}
	// Nothing is injected into this dialect.
	up := stub.lastRequest(t, "/v1/messages")
	if requestField(up.body, "stream_options") != nil {
		t.Errorf("stream_options injected into /v1/messages: %s", up.body)
	}

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_prompt_tokens_total", `model="`+tagged+`"`); v != stubPromptTokens {
		t.Errorf("prompt tokens = %v, want %v (message_start's placeholder must not win)", v, stubPromptTokens)
	}
	if v := mustSample(t, metrics, "ollama_generated_tokens_total", `model="`+tagged+`"`); v != stubGeneratedTokens {
		t.Errorf("generated tokens = %v, want %v", v, stubGeneratedTokens)
	}
	if v := mustSample(t, metrics, "ollama_request_duration_seconds_count", `api_endpoint="v1/messages"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("request duration count = %v, want 1", v)
	}
	if v := mustSample(t, metrics, "ollama_time_to_first_token_seconds_count", `api_endpoint="v1/messages"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("time to first token count = %v, want 1", v)
	}
	if v := mustSample(t, metrics, "ollama_time_per_token_seconds_count", `model="`+tagged+`"`); v != 1 {
		t.Errorf("time per token count = %v, want 1", v)
	}
}

// TestAnthropicNonStreaming reads usage from the body, counting cached prompt
// tokens too, and records no stream-only timings.
func TestAnthropicNonStreaming(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "anthropic-blocking"
	tagged := model + ":latest"

	got := post(t, proxy, "/v1/messages", fmt.Sprintf(`{"model":%q,"max_tokens":100,"messages":[]}`, model), nil)
	if want := anthropicBody(model); got != want {
		t.Errorf("non-streaming body was not forwarded intact:\n got: %q\nwant: %q", got, want)
	}

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_prompt_tokens_total", `model="`+tagged+`"`); v != stubPromptTokens {
		t.Errorf("prompt tokens = %v, want %v (input + cache read)", v, stubPromptTokens)
	}
	if v := mustSample(t, metrics, "ollama_generated_tokens_total", `model="`+tagged+`"`); v != stubGeneratedTokens {
		t.Errorf("generated tokens = %v, want %v", v, stubGeneratedTokens)
	}
	requireAbsent(t, metrics, "ollama_time_per_token_seconds_count", `model="`+tagged+`"`)
	requireAbsent(t, metrics, "ollama_time_to_first_token_seconds_count", `model="`+tagged+`"`)
}

// TestObserveAnthropicKeepsInputWhenDeltaOmitsIt covers Anthropic's own
// convention, where message_delta reports output only: the input count from
// message_start must survive it.
func TestObserveAnthropicKeepsInputWhenDeltaOmitsIt(t *testing.T) {
	var st requestStats
	for _, ev := range []string{
		`{"type":"message_start","message":{"model":"m","usage":{"input_tokens":40,"cache_read_input_tokens":2,"output_tokens":1}}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}`,
	} {
		var parsed anthropicEvent
		if err := json.Unmarshal([]byte(ev), &parsed); err != nil {
			t.Fatal(err)
		}
		st.observeAnthropic(&parsed)
	}
	if st.model != "m:latest" || st.promptCount != 42 || st.generatedCount != 9 {
		t.Errorf("got model=%q prompt=%d generated=%d, want m:latest 42 9", st.model, st.promptCount, st.generatedCount)
	}
}

// TestGzipRequestStillParses guards the Accept-Encoding bug: forwarding the
// client's header disables Go's transparent decompression, leaving the proxy
// parsing gzip bytes and silently counting nothing.
func TestGzipRequestStillParses(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "v1-gzip"
	tagged := model + ":latest"

	header := http.Header{"Accept-Encoding": []string{"gzip"}}
	got, respHeader := postFull(t, proxy, "/v1/chat/completions", fmt.Sprintf(`{"model":%q,"messages":[]}`, model), header)

	metrics := scrape(t, proxy)
	if v, ok := sample(t, metrics, "ollama_prompt_tokens_total", `model="`+tagged+`"`); !ok || v != stubPromptTokens {
		t.Errorf("prompt tokens = %v (present=%v), want %v: the upstream body arrived gzipped and parsed to nothing",
			v, ok, stubPromptTokens)
	}

	// Because the client's Accept-Encoding never reaches upstream, the body comes
	// back uncompressed and the client can read it without negotiating anything.
	if enc := respHeader.Get("Content-Encoding"); enc != "" {
		t.Errorf("response Content-Encoding = %q, want none", enc)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal([]byte(got), &decoded); err != nil {
		t.Errorf("client could not decode body %q: %v", got, err)
	}
}

// TestHistogramSeriesAppearOnlyAfterTraffic documents that the _bucket/_count/
// _sum series are missing until a request creates them: they are lazily created
// *Vec children, not a broken exporter.
func TestHistogramSeriesAppearOnlyAfterTraffic(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "v1-lazy-child"
	tagged := model + ":latest"
	labels := []string{`api_endpoint="v1/chat/completions"`, `model="` + tagged + `"`}

	before := scrape(t, proxy)
	for _, suffix := range []string{"_bucket", "_count", "_sum"} {
		requireAbsent(t, before, "ollama_request_duration_seconds"+suffix, labels...)
		requireAbsent(t, before, "ollama_time_to_first_token_seconds"+suffix, labels...)
	}

	post(t, proxy, "/v1/chat/completions", fmt.Sprintf(`{"model":%q,"messages":[],"stream":true}`, model), nil)

	after := scrape(t, proxy)
	for _, suffix := range []string{"_bucket", "_count", "_sum"} {
		mustSample(t, after, "ollama_request_duration_seconds"+suffix, labels...)
		mustSample(t, after, "ollama_time_to_first_token_seconds"+suffix, labels...)
		mustSample(t, after, "ollama_time_per_token_seconds"+suffix, `model="`+tagged+`"`)
	}
}

// TestLoadedModelGauges covers the /api/ps interception and the refresh the
// /metrics handler does on every scrape.
func TestLoadedModelGauges(t *testing.T) {
	proxy, _ := newProxy(t)

	resp, err := proxy.Client().Get(proxy.URL + "/api/ps")
	if err != nil {
		t.Fatalf("GET /api/ps: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(body), "stub-loaded") {
		t.Errorf("/api/ps body not forwarded: %q", body)
	}

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_model_loaded", `model="stub-loaded:latest"`); v != 1 {
		t.Errorf("model_loaded = %v, want 1", v)
	}
	if v := mustSample(t, metrics, "ollama_model_ram_mb", `model="stub-loaded:latest"`); v != 1 {
		t.Errorf("model_ram_mb = %v, want 1", v)
	}
}

// TestPassthroughEndpoint checks an unparsed endpoint still gets a duration
// observation, labelled from the request body's model.
func TestPassthroughEndpoint(t *testing.T) {
	proxy, _ := newProxy(t)
	const model = "passthrough"
	tagged := model + ":latest"

	post(t, proxy, "/api/tags", fmt.Sprintf(`{"model":%q}`, model), nil)

	metrics := scrape(t, proxy)
	if v := mustSample(t, metrics, "ollama_request_duration_seconds_count", `api_endpoint="tags"`, `model="`+tagged+`"`); v != 1 {
		t.Errorf("request duration count = %v, want 1", v)
	}
}

func TestEndpointLabel(t *testing.T) {
	cases := map[string]string{
		"/api/chat":            "chat",
		"/api/generate":        "generate",
		"/api/ps":              "ps",
		"/v1/chat/completions": "v1/chat/completions",
		"/v1/completions":      "v1/completions",
		"/":                    "/",
	}
	for path, want := range cases {
		if got := endpointLabel(path); got != want {
			t.Errorf("endpointLabel(%q) = %q, want %q", path, got, want)
		}
	}
}

// TestInjectIncludeUsage covers the request-rewriting decision table on its own.
func TestInjectIncludeUsage(t *testing.T) {
	cases := []struct {
		name         string
		body         string
		wantInjected bool
	}{
		{"non-streaming is left alone", `{"model":"m"}`, false},
		{"explicit stream:false is left alone", `{"model":"m","stream":false}`, false},
		{"streaming gets usage injected", `{"model":"m","stream":true}`, true},
		{"empty stream_options gets usage injected", `{"model":"m","stream":true,"stream_options":{}}`, true},
		{"null stream_options gets usage injected", `{"model":"m","stream":true,"stream_options":null}`, true},
		{"client-requested usage is left alone", `{"model":"m","stream":true,"stream_options":{"include_usage":true}}`, false},
		{"unparseable body is left alone", `not json`, false},
		{"empty body is left alone", ``, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := []byte(tc.body)
			got, injected := injectIncludeUsage(parseRequestJSON(body), body)
			if injected != tc.wantInjected {
				t.Fatalf("injected = %v, want %v", injected, tc.wantInjected)
			}
			if !injected {
				if !bytes.Equal(got, body) {
					t.Errorf("body was rewritten to %q, want it untouched", got)
				}
				return
			}
			opts, ok := requestField(got, "stream_options").(map[string]interface{})
			if !ok {
				t.Fatalf("rewritten body has no stream_options object: %s", got)
			}
			if include, _ := opts["include_usage"].(bool); !include {
				t.Errorf("rewritten body does not set include_usage: %s", got)
			}
		})
	}
}

// TestSanitizeUTF8 covers the repair itself. The point is not tidiness: an
// invalid byte in a reply costs the whole report downstream, because
// Sympozium's runner ships it to its controller over gRPC and protobuf refuses
// to marshal a string field that is not valid UTF-8 - dropping the result while
// the run still reports success.
func TestSanitizeUTF8(t *testing.T) {
	valid := []byte(`{"content":"SRE Sentinel · homelab 中文 ✓"}`)
	if got := sanitizeUTF8(valid); !bytes.Equal(got, valid) {
		t.Errorf("valid input was modified:\n got: %q\nwant: %q", got, valid)
	}
	// The fast path must not allocate a copy for the common case.
	if got := sanitizeUTF8(valid); &got[0] != &valid[0] {
		t.Error("valid input was copied; expected the same backing array")
	}

	for _, tc := range []struct {
		name string
		in   []byte
		want string
	}{
		// A lone continuation byte: what a truncated multi-byte emit looks like.
		{"lone continuation", []byte{'a', 0xb7, 'b'}, "a�b"},
		// A lead byte with its continuation missing - the U+00B7 case that
		// actually happened, where the model emitted half of "\xc2\xb7".
		{"truncated lead", []byte{'a', 0xc2, 'b'}, "a�b"},
		// Surrounding valid multi-byte text must survive intact.
		{"keeps neighbours", []byte("中\xff·"), "中�·"},
		{"trailing lead byte", []byte{'x', 0xe4}, "x�"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := string(sanitizeUTF8(tc.in))
			if got != tc.want {
				t.Errorf("sanitizeUTF8(%q) = %q, want %q", tc.in, got, tc.want)
			}
			if !utf8.ValidString(got) {
				t.Errorf("output is still not valid UTF-8: %q", got)
			}
		})
	}
}

func TestSanitizeUTF8ResponsesEnabled(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  bool
	}{
		{"", true},
		{"true", true},
		{"false", false},
		// A value ParseBool cannot read is a typo, not a request to turn the
		// protection off, so it keeps the safe default.
		{"yes", true},
		{"invalid", true},
	} {
		t.Run(strconv.Quote(tc.value), func(t *testing.T) {
			t.Setenv("OLLAMA_SANITIZE_UTF8_RESPONSE", tc.value)
			if got := sanitizeUTF8ResponsesEnabled(); got != tc.want {
				t.Errorf("sanitizeUTF8ResponsesEnabled() = %t, want %t", got, tc.want)
			}
		})
	}
}

// TestInvalidUTF8IsRepairedInFlight drives a real /api/chat stream whose upstream
// emits a broken byte, and checks the client sees valid UTF-8 with the
// surrounding characters untouched. Uses its own upstream rather than the shared
// stub, which only ever serves well-formed bodies.
func TestInvalidUTF8IsRepairedInFlight(t *testing.T) {
	line := append([]byte(`{"model":"m","message":{"role":"assistant","content":"ok `), 0xc2)
	line = append(line, []byte(` · end"},"done":false}`+"\n")...)
	done := []byte(`{"model":"m","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1,"eval_duration":1000000}` + "\n")

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Write(line)
		w.Write(done)
	}))
	defer upstream.Close()

	proxy := httptest.NewServer(newMux(upstream.URL))
	defer proxy.Close()

	res, err := http.Post(proxy.URL+"/api/chat", "application/json",
		strings.NewReader(`{"model":"m","messages":[],"stream":true}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer res.Body.Close()
	got, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	if !utf8.Valid(got) {
		t.Fatalf("client still received invalid UTF-8: %q", got)
	}
	if !strings.Contains(string(got), "�") {
		t.Errorf("expected the bad byte replaced by U+FFFD, got %q", got)
	}
	// The characters either side of the damage must be intact.
	for _, want := range []string{"ok ", " · end", `"done":true`} {
		if !strings.Contains(string(got), want) {
			t.Errorf("lost %q from the stream: %q", want, got)
		}
	}
}

func TestInvalidUTF8CanBeForwardedUnchanged(t *testing.T) {
	body := []byte{'x', 0xc2, 'y'}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(body)
	}))
	defer upstream.Close()

	proxy := httptest.NewServer(newMuxWithConfig(upstream.URL, proxyConfig{
		requestDefaults:       map[string]interface{}{},
		requestOverrides:      map[string]interface{}{},
		sanitizeUTF8Responses: false,
		promoteReasoning:      true,
	}))
	defer proxy.Close()

	res, err := http.Post(proxy.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"m","messages":[]}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer res.Body.Close()
	got, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, body) {
		t.Errorf("body = %q, want unchanged invalid UTF-8 %q", got, body)
	}
}

// TestPromoteReasoningToContent pins the repair against the shape that lost a
// scheduled report: qwen3.5:4b ended its terminal turn with a complete analysis
// in "reasoning", an empty "content" and finish_reason "stop", and the caller,
// which reads only content, delivered nothing while reporting success.
func TestPromoteReasoningToContent(t *testing.T) {
	body := []byte(`{"id":"chatcmpl-530","object":"chat.completion","model":"qwen3.5:4b",` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant",` +
		`"content":"","reasoning":"Let me analyze the results: Postgres is healthy."}}],` +
		`"usage":{"prompt_tokens":5541,"completion_tokens":266,"total_tokens":5807}}`)

	got, promoted := promoteReasoningToContent(body)
	if promoted != 1 {
		t.Fatalf("promoted = %d, want 1", promoted)
	}
	var res struct {
		Usage   map[string]json.Number `json:"usage"`
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content   string `json:"content"`
				Reasoning string `json:"reasoning"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(got, &res); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	want := "Let me analyze the results: Postgres is healthy."
	if res.Choices[0].Message.Content != want {
		t.Errorf("content = %q, want %q", res.Choices[0].Message.Content, want)
	}
	// Copied, not moved: a client reading both fields still sees what was sent.
	if res.Choices[0].Message.Reasoning != want {
		t.Errorf("reasoning = %q, want it left in place", res.Choices[0].Message.Reasoning)
	}
	if res.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", res.Choices[0].FinishReason)
	}
	// Integer literals must survive the re-marshal.
	if v := res.Usage["total_tokens"].String(); v != "5807" {
		t.Errorf("total_tokens = %s, want 5807", v)
	}
}

// TestPromoteReasoningLeavesToolCallTurnsAlone is the guard that matters most:
// every intermediate turn of an agent loop has empty content by design, and
// rewriting one would put prose where the caller expects a tool call.
func TestPromoteReasoningLeavesToolCallTurnsAlone(t *testing.T) {
	body := []byte(`{"object":"chat.completion","choices":[{"message":{"role":"assistant",` +
		`"content":"","reasoning":"I should call the health tool.",` +
		`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"health","arguments":"{}"}}]}}]}`)

	got, promoted := promoteReasoningToContent(body)
	if promoted != 0 {
		t.Fatalf("promoted = %d, want 0 for a tool-call turn", promoted)
	}
	if !bytes.Equal(got, body) {
		t.Errorf("body was rewritten:\n got %s\nwant %s", got, body)
	}
}

// TestPromoteReasoningSkipsHealthyResponses covers the cases that must stay
// byte-identical: real content present, and no reasoning to promote.
func TestPromoteReasoningSkipsHealthyResponses(t *testing.T) {
	cases := map[string]string{
		"content present":   `{"choices":[{"message":{"content":"## Headline\nAll clear","reasoning":"thinking"}}]}`,
		"no reasoning":      `{"choices":[{"message":{"content":""}}]}`,
		"whitespace only":   `{"choices":[{"message":{"content":"","reasoning":"   "}}]}`,
		"not a completion":  `{"models":[{"name":"m"}]}`,
		"malformed":         `{"choices":`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			got, promoted := promoteReasoningToContent([]byte(body))
			if promoted != 0 {
				t.Fatalf("promoted = %d, want 0", promoted)
			}
			if string(got) != body {
				t.Errorf("body = %s, want it unchanged", got)
			}
		})
	}
}

// TestPromoteReasoningToContentEnabled pins the default-on behaviour and the
// refusal to disable the repair on a typo.
func TestPromoteReasoningToContentEnabled(t *testing.T) {
	cases := []struct {
		raw  string
		set  bool
		want bool
	}{
		{set: false, want: true},
		{raw: "true", set: true, want: true},
		{raw: "false", set: true, want: false},
		{raw: "0", set: true, want: false},
		{raw: "yes-please", set: true, want: true},
	}
	for _, tc := range cases {
		if tc.set {
			t.Setenv("OLLAMA_PROMOTE_REASONING_TO_CONTENT", tc.raw)
		} else {
			os.Unsetenv("OLLAMA_PROMOTE_REASONING_TO_CONTENT")
		}
		if got := promoteReasoningToContentEnabled(); got != tc.want {
			t.Errorf("OLLAMA_PROMOTE_REASONING_TO_CONTENT=%q: got %v, want %v", tc.raw, got, tc.want)
		}
	}
}

// TestPromotedResponseReachesTheClient exercises the whole proxy path, since
// the promotion has to survive header copying and the metrics parse.
func TestPromotedResponseReachesTheClient(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"object":"chat.completion","model":"qwen3.5:4b",`+
			`"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"","reasoning":"The report."}}],`+
			`"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}`)
	}))
	defer upstream.Close()

	proxy := httptest.NewServer(newMuxWithConfig(upstream.URL, proxyConfig{
		requestDefaults:       map[string]interface{}{},
		requestOverrides:      map[string]interface{}{},
		sanitizeUTF8Responses: true,
		promoteReasoning:      true,
	}))
	defer proxy.Close()

	res, err := http.Post(proxy.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"qwen3.5:4b","messages":[]}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer res.Body.Close()
	got, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatalf("unmarshal %s: %v", got, err)
	}
	if parsed.Choices[0].Message.Content != "The report." {
		t.Errorf("content = %q, want %q", parsed.Choices[0].Message.Content, "The report.")
	}
}
