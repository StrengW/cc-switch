use serde_json::Value;

const PROVIDER_STATE_ITEM_TYPES: &[&str] = &[
    "message",
    "reasoning",
    "function_call",
    "function_call_output",
    "custom_tool_call",
    "custom_tool_call_output",
    "tool_search_call",
    "tool_search_call_output",
];

const PROVIDER_ITEM_ID_PREFIXES: &[&str] =
    &["resp_", "msg_", "rs_", "fc_", "fco_", "ctc_", "ctco_"];

const PREVIOUS_RESPONSE_REJECTION_TERMS: &[&str] = &[
    "invalid",
    "not found",
    "unknown",
    "unsupported",
    "does not exist",
    "expired",
    "cannot",
];

/// Match only Responses-state failures that the guarded sanitizer can repair.
pub(crate) fn is_cross_provider_responses_portability_error(
    status: u16,
    body: Option<&str>,
) -> bool {
    if !matches!(status, 400 | 422) {
        return false;
    }
    let Some(body) = body else {
        return false;
    };
    let body = body.to_ascii_lowercase();
    let names_input = body.contains("input[");

    (names_input && body.contains("expected an id that begins"))
        || (body.contains("encrypted content") && body.contains("could not be verified"))
        || (names_input && body.contains("array too long"))
        || (body.contains("previous_response_id")
            && PREVIOUS_RESPONSE_REJECTION_TERMS
                .iter()
                .any(|term| body.contains(term)))
}

/// Remove only provider-private state from an in-memory Responses request copy.
///
/// Visible messages and complete plain tool call/output pairs are preserved. The
/// function is deterministic and idempotent.
pub(crate) fn sanitize_cross_provider_responses_request(body: &mut Value) -> bool {
    let Some(object) = body.as_object_mut() else {
        return false;
    };
    let mut changed = object.remove("previous_response_id").is_some();
    changed |= sanitize_include(object);

    let Some(input) = object.get_mut("input") else {
        return changed;
    };
    match input {
        Value::Array(items) => {
            for item in items {
                changed |= sanitize_input_item(item);
            }
        }
        Value::Object(_) => {
            changed |= sanitize_input_item(input);
        }
        _ => {}
    }
    changed
}

fn sanitize_include(object: &mut serde_json::Map<String, Value>) -> bool {
    let Some(include) = object.get("include") else {
        return false;
    };
    let replacement = match include {
        Value::String(value) if is_encrypted_reasoning_include(value) => None,
        Value::Array(items) => {
            let filtered = items
                .iter()
                .filter(|item| !item.as_str().is_some_and(is_encrypted_reasoning_include))
                .cloned()
                .collect::<Vec<_>>();
            if filtered.len() == items.len() {
                return false;
            }
            if filtered.is_empty() {
                None
            } else {
                Some(Value::Array(filtered))
            }
        }
        _ => return false,
    };

    match replacement {
        Some(value) => {
            object.insert("include".to_string(), value);
        }
        None => {
            object.remove("include");
        }
    }
    true
}

fn is_encrypted_reasoning_include(value: &str) -> bool {
    value
        .trim()
        .eq_ignore_ascii_case("reasoning.encrypted_content")
}

fn sanitize_input_item(item: &mut Value) -> bool {
    let Some(object) = item.as_object_mut() else {
        return false;
    };
    let item_type = object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let mut changed = false;

    let provider_owned_item = PROVIDER_STATE_ITEM_TYPES.contains(&item_type.as_str());
    let provider_owned_id = object
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(is_provider_item_id);
    if provider_owned_item || provider_owned_id {
        changed |= object.remove("id").is_some();
    }

    if item_type == "reasoning" {
        changed |= object.remove("encrypted_content").is_some();
        if object.get("content").is_some_and(Value::is_array) {
            object.remove("content");
            changed = true;
        }
    }

    changed
}

fn is_provider_item_id(value: &str) -> bool {
    let value = value.trim();
    PROVIDER_ITEM_ID_PREFIXES
        .iter()
        .any(|prefix| value.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matcher_accepts_confirmed_cross_provider_state_errors() {
        let cases = [
            (
                400,
                r#"{"error":{"message":"Invalid 'input[6].id': 'resp_abc'. Expected an ID that begins with 'msg'."}}"#,
            ),
            (
                400,
                r#"{"error":{"message":"The encrypted content for item rs_1 could not be verified."}}"#,
            ),
            (
                400,
                r#"{"error":{"message":"Invalid 'input[105].content': array too long."}}"#,
            ),
            (
                422,
                r#"{"error":{"message":"previous_response_id is invalid for this endpoint"}}"#,
            ),
        ];

        for (status, body) in cases {
            assert!(
                is_cross_provider_responses_portability_error(status, Some(body)),
                "expected status={status} body={body} to trigger portability retry"
            );
        }
    }

    #[test]
    fn matcher_rejects_unrelated_400_and_422_errors() {
        let cases = [
            (
                400,
                r#"{"error":{"message":"model deepseek-v4-flash is not supported"}}"#,
            ),
            (400, r#"{"error":{"message":"invalid api key"}}"#),
            (422, r#"{"error":{"message":"invalid temperature value"}}"#),
            (429, r#"{"error":{"message":"rate limit exceeded"}}"#),
            (500, r#"{"error":{"message":"internal server error"}}"#),
            (
                400,
                r#"{"error":{"message":"input[0].temperature is invalid"}}"#,
            ),
        ];

        for (status, body) in cases {
            assert!(
                !is_cross_provider_responses_portability_error(status, Some(body)),
                "expected status={status} body={body} to remain unchanged"
            );
        }
    }

    #[test]
    fn sanitizer_leaves_normal_requests_byte_identical() {
        let original = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "shell",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok"
                }
            ],
            "store": true
        });
        let mut body = original.clone();

        let changed = sanitize_cross_provider_responses_request(&mut body);

        assert!(!changed);
        assert_eq!(body, original);
    }

    #[test]
    fn sanitizer_removes_provider_private_state_but_preserves_visible_content() {
        let mut body = json!({
            "previous_response_id": "resp_foreign",
            "include": ["reasoning.encrypted_content", "message.output_text.logprobs"],
            "input": [
                {"type": "message", "id": "msg_user", "role": "user", "content": "keep user text"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "keep summary"}],
                    "content": [{"type": "reasoning_text", "text": "private"}],
                    "encrypted_content": "opaque"
                },
                {
                    "type": "message",
                    "id": "msg_assistant",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "keep assistant text"}]
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "shell",
                    "arguments": "{\"cmd\":\"pwd\"}"
                },
                {
                    "type": "function_call_output",
                    "id": "fco_1",
                    "call_id": "call_1",
                    "output": "kept tool output"
                }
            ]
        });

        assert!(sanitize_cross_provider_responses_request(&mut body));
        assert!(body.get("previous_response_id").is_none());
        assert_eq!(body["include"], json!(["message.output_text.logprobs"]));
        assert!(body["input"][0].get("id").is_none());
        assert_eq!(body["input"][0]["content"], "keep user text");
        assert!(body["input"][1].get("id").is_none());
        assert!(body["input"][1].get("encrypted_content").is_none());
        assert!(body["input"][1].get("content").is_none());
        assert_eq!(body["input"][1]["summary"][0]["text"], "keep summary");
        assert!(body["input"][2].get("id").is_none());
        assert_eq!(
            body["input"][2]["content"][0]["text"],
            "keep assistant text"
        );
        assert!(body["input"][3].get("id").is_none());
        assert_eq!(body["input"][3]["call_id"], "call_1");
        assert_eq!(body["input"][3]["arguments"], "{\"cmd\":\"pwd\"}");
        assert!(body["input"][4].get("id").is_none());
        assert_eq!(body["input"][4]["call_id"], "call_1");
        assert_eq!(body["input"][4]["output"], "kept tool output");
    }

    #[test]
    fn sanitizer_is_deterministic_and_idempotent() {
        let mut body = json!({
            "previous_response_id": "resp_foreign",
            "include": ["reasoning.encrypted_content"],
            "input": [{
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "opaque",
                "content": [{"type": "reasoning_text", "text": "private"}]
            }]
        });

        assert!(sanitize_cross_provider_responses_request(&mut body));
        let once = body.clone();
        assert!(!sanitize_cross_provider_responses_request(&mut body));
        assert_eq!(body, once);
    }

    #[test]
    fn sanitizer_keeps_non_provider_ids_on_unknown_items() {
        let mut body = json!({
            "input": [{
                "type": "custom_item",
                "id": "user-owned-id",
                "content": "keep"
            }]
        });

        assert!(!sanitize_cross_provider_responses_request(&mut body));
        assert_eq!(body["input"][0]["id"], "user-owned-id");
    }
}
