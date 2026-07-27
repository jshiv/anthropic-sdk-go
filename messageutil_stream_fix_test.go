package anthropic_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

// Exercises the full streaming replay path a multi-turn agent takes: stream a
// turn containing a web_search_tool_result plus a text block citing it,
// Accumulate() every event, then ToParam() the message onto the next turn's
// conversation. Without the ContentBlockStopEvent carve-out in Accumulate the
// re-marshal collapses the result block to web_search_tool_result_error and
// every search result is silently dropped from the replayed turn.
func TestStreamedWebSearchTurnRoundTrip(t *testing.T) {
	rawEvents := []string{
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"go sdk\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","title":"Go SDK","url":"https://example.com/go","encrypted_content":"ENC_ABC","page_age":"2 days ago"}]}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"The SDK is here."}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","cited_text":"The SDK is here.","url":"https://example.com/go","title":"Go SDK","encrypted_index":"IDX_XYZ"}}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":20}}`,
		`{"type":"message_stop"}`,
	}

	msg := anthropic.Message{}
	for i, raw := range rawEvents {
		var ev anthropic.MessageStreamEventUnion
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			t.Fatalf("event %d unmarshal: %v", i, err)
		}
		if err := msg.Accumulate(ev); err != nil {
			t.Fatalf("event %d accumulate: %v", i, err)
		}
	}

	out, err := json.MarshalIndent(msg.ToParam(), "", "  ")
	if err != nil {
		t.Fatalf("marshal param: %v", err)
	}
	got := string(out)

	for _, want := range []string{
		"ENC_ABC",                // web_search_result.encrypted_content
		"https://example.com/go", // result url + citation url
		"IDX_XYZ",                // citation encrypted_index
		"Go SDK",                 // result title
	} {
		if !strings.Contains(got, want) {
			t.Errorf("replayed turn is missing %q\nToParam JSON:\n%s", want, got)
		}
	}
	if strings.Contains(got, "web_search_tool_result_error") {
		t.Errorf("replayed turn degraded to web_search_tool_result_error\nToParam JSON:\n%s", got)
	}
}

// The streaming carve-out must not affect blocks that legitimately accumulate
// from deltas: a text block's JSON.raw still has to reflect the accumulated
// text, not the empty string from content_block_start.
func TestStreamedTextBlockStillReMarshals(t *testing.T) {
	rawEvents := []string{
		`{"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_stop"}`,
	}

	msg := anthropic.Message{}
	for i, raw := range rawEvents {
		var ev anthropic.MessageStreamEventUnion
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			t.Fatalf("event %d unmarshal: %v", i, err)
		}
		if err := msg.Accumulate(ev); err != nil {
			t.Fatalf("event %d accumulate: %v", i, err)
		}
	}

	if got := msg.Content[0].RawJSON(); !strings.Contains(got, "hello world") {
		t.Errorf("text block JSON.raw was not updated with accumulated text: %s", got)
	}
	if got := msg.Content[0].AsText().Text; got != "hello world" {
		t.Errorf("AsText().Text = %q, want %q", got, "hello world")
	}
}
