package anthropic_test

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

func TestWebSearchToParamRoundTrip(t *testing.T) {
	// Simulate the API response for a web_search_tool_result
	apiResponse := []byte(`{
		"id": "msg_test",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-6",
		"stop_reason": "end_turn",
		"content": [
			{
				"type": "web_search_tool_result",
				"tool_use_id": "srvtoolu_123",
				"content": [
					{
						"type": "web_search_result",
						"url": "https://example.com/1",
						"title": "First Result",
						"encrypted_content": "ZW5jcnlwdGVk",
						"page_age": "1 day ago"
					},
					{
						"type": "web_search_result",
						"url": "https://example.com/2",
						"title": "Second Result",
						"encrypted_content": "ZW5jcnlwdGVkMg=="
					}
				]
			}
		],
		"usage": {"input_tokens": 100, "output_tokens": 50}
	}`)

	var msg anthropic.Message
	if err := json.Unmarshal(apiResponse, &msg); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	param := msg.ToParam()

	// Marshal the full MessageNewParams to see the exact request body
	req := anthropic.MessageNewParams{
		Model:    anthropic.ModelClaudeSonnet4_6,
		MaxTokens: 1024,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("test")),
			param,
			anthropic.NewUserMessage(anthropic.NewTextBlock("continue")),
		},
	}

	reqJSON, err := json.MarshalIndent(req, "", "  ")
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	t.Logf("Request body:\n%s", string(reqJSON))

	// Also marshal just the param to see the content shape
	paramJSON, err := json.MarshalIndent(param, "", "  ")
	if err != nil {
		t.Fatalf("marshal param: %v", err)
	}
	t.Logf("Param JSON:\n%s", string(paramJSON))

	// Verify the content field is present and is an array
	var raw map[string]json.RawMessage
	json.Unmarshal(paramJSON, &raw)
	var contentBlocks []json.RawMessage
	json.Unmarshal(raw["content"], &contentBlocks)
	if len(contentBlocks) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(contentBlocks))
	}
	var block map[string]json.RawMessage
	json.Unmarshal(contentBlocks[0], &block)
	content := block["content"]
	if content == nil || string(content) == "null" {
		t.Fatal("web_search_tool_result.content is nil or null")
	}
	// Should be an array
	var arr []json.RawMessage
	if err := json.Unmarshal(content, &arr); err != nil {
		t.Fatalf("web_search_tool_result.content is not a valid array: %v\nGot: %s", err, string(content))
	}
	if len(arr) != 2 {
		t.Fatalf("expected 2 search results, got %d", len(arr))
	}
}

func TestWebFetchToParamRoundTrip(t *testing.T) {
	apiResponse := []byte(`{
		"id": "msg_test2",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-6",
		"stop_reason": "end_turn",
		"content": [
			{
				"type": "web_fetch_tool_result",
				"tool_use_id": "srvtoolu_456",
				"content": {
					"type": "web_fetch_result",
					"url": "https://example.com",
					"retrieved_at": "2026-05-27T12:00:00Z",
					"content": {
						"type": "document",
						"citations": {"enabled": false},
						"source": {
							"type": "text",
							"media_type": "text/plain",
							"data": "Hello World"
						},
						"title": "Example Page"
					}
				}
			}
		],
		"usage": {"input_tokens": 100, "output_tokens": 50}
	}`)

	var msg anthropic.Message
	if err := json.Unmarshal(apiResponse, &msg); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	param := msg.ToParam()
	paramJSON, err := json.MarshalIndent(param, "", "  ")
	if err != nil {
		t.Fatalf("marshal param: %v", err)
	}
	t.Logf("Param JSON:\n%s", string(paramJSON))

	var raw map[string]json.RawMessage
	json.Unmarshal(paramJSON, &raw)
	var contentBlocks []json.RawMessage
	json.Unmarshal(raw["content"], &contentBlocks)
	if len(contentBlocks) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(contentBlocks))
	}
	var block map[string]json.RawMessage
	json.Unmarshal(contentBlocks[0], &block)
	content := block["content"]
	if content == nil || string(content) == "null" || string(content) == "" {
		t.Fatalf("BUG: web_fetch_tool_result.content is missing\nFull block: %s", string(contentBlocks[0]))
	}
	t.Logf("web_fetch_tool_result.content = %s", string(content))
}
