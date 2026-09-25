//! Terminal response-usage parsing for fair-share settlement.
//!
//! This parser is intentionally owned by the scheduler completion path. It
//! does not call or depend on adaptive admission's predictor.

fn value_u32(value: Option<&serde_json::Value>) -> Option<u32> {
    value?.as_u64().and_then(|value| u32::try_from(value).ok())
}

fn output_tokens_from_value(value: &serde_json::Value) -> Option<u32> {
    [
        "/usage/completion_tokens",
        "/usage/output_tokens",
        "/meta_info/completion_tokens",
    ]
    .into_iter()
    .find_map(|pointer| value_u32(value.pointer(pointer)))
}

fn output_tokens_from_truncated_json_tail(body: &[u8]) -> Option<u32> {
    [b"\"usage\"".as_slice(), b"\"meta_info\"".as_slice()]
        .into_iter()
        .find_map(|key| {
            let offset = body.windows(key.len()).rposition(|window| window == key)? + key.len();
            let remainder = &body[offset..];
            let object = &remainder[remainder.iter().position(|byte| *byte == b':')? + 1..];
            let value = serde_json::Deserializer::from_slice(object)
                .into_iter::<serde_json::Value>()
                .next()?
                .ok()?;
            value_u32(value.get("completion_tokens"))
                .or_else(|| value_u32(value.get("output_tokens")))
        })
}

pub(crate) fn observed_output_tokens(body: &[u8]) -> Option<u32> {
    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) {
        if let Some(tokens) = output_tokens_from_value(&value) {
            return Some(tokens);
        }
    }
    if let Some(tokens) = output_tokens_from_truncated_json_tail(body) {
        return Some(tokens);
    }

    body.split(|byte| *byte == b'\n')
        .filter_map(|line| {
            let line = line
                .strip_suffix(b"\r")
                .unwrap_or(line)
                .strip_prefix(b"data:")?;
            let line = line.strip_prefix(b" ").unwrap_or(line);
            if line == b"[DONE]" {
                return None;
            }
            serde_json::from_slice::<serde_json::Value>(line)
                .ok()
                .and_then(|value| output_tokens_from_value(&value))
        })
        .next_back()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_anthropic_and_sglang_shapes() {
        assert_eq!(
            observed_output_tokens(br#"{"usage":{"completion_tokens":42}}"#),
            Some(42)
        );
        assert_eq!(
            observed_output_tokens(br#"{"usage":{"output_tokens":13}}"#),
            Some(13)
        );
        assert_eq!(
            observed_output_tokens(br#"{"meta_info":{"completion_tokens":7}}"#),
            Some(7)
        );
    }

    #[test]
    fn parses_terminal_stream_usage() {
        assert_eq!(
            observed_output_tokens(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":17}}\n\ndata: [DONE]\n\n"
            ),
            Some(17)
        );
    }

    #[test]
    fn missing_usage_is_not_invented() {
        assert_eq!(observed_output_tokens(br#"{"choices":[]}"#), None);
    }
}
