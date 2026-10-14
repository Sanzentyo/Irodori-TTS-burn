//! Independent generated-shader cache selection for controlled comparisons.

pub(super) fn spirv_cache_requested() -> bool {
    match std::env::var("CUBECL_WGPU_SPIRV_CACHE") {
        Ok(value) => match parse_switch(Some(&value)) {
            Some(enabled) => enabled,
            None => {
                log::warn!("CUBECL_WGPU_SPIRV_CACHE expects 0 or 1; disabling SPIR-V persistence");
                false
            }
        },
        Err(std::env::VarError::NotPresent) => true,
        Err(error) => {
            log::warn!("Invalid CUBECL_WGPU_SPIRV_CACHE: {error}; disabling SPIR-V persistence");
            false
        }
    }
}

fn parse_switch(value: Option<&str>) -> Option<bool> {
    match value {
        None | Some("1") => Some(true),
        Some("0") => Some(false),
        Some(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spirv_cache_switch_rejects_ambiguous_values() {
        assert_eq!(parse_switch(None), Some(true));
        assert_eq!(parse_switch(Some("1")), Some(true));
        assert_eq!(parse_switch(Some("0")), Some(false));
        for value in ["", "true", "false", " 1", "2"] {
            assert_eq!(parse_switch(Some(value)), None);
        }
    }
}
