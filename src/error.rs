use thiserror::Error;

#[derive(Debug, Error)]
pub enum IrodoriError {
    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Shape mismatch: {0}")]
    Shape(String),

    #[error("Missing required input: {0}")]
    MissingInput(String),

    #[error("Unsupported mode: {0}")]
    UnsupportedMode(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// A required tensor key was not found in the checkpoint.
    #[error("Missing weight tensor: {0}")]
    Weight(String),

    /// A tensor had an unexpected number of dimensions.
    #[error("Wrong rank for tensor '{0}': expected {1}, got {2}")]
    WrongDim(String, usize, usize),

    /// A tensor had an unsupported dtype.
    #[error("Unsupported dtype for tensor '{0}': {1}")]
    Dtype(String, String),

    /// The safetensors file was malformed or could not be read.
    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    /// The checkpoint is missing the required `config_json` metadata key.
    #[error("Checkpoint is missing the 'config_json' metadata key")]
    NoConfig,

    /// Tokenizer error (tokenizers crate).
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// HuggingFace Hub download error.
    #[error("HF Hub error: {0}")]
    HfHub(String),

    /// Dataset loading or manifest parsing error.
    #[error("Dataset error: {0}")]
    Dataset(String),

    /// Checkpoint save/load error.
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    /// Training-related error (resume, validation, etc.).
    #[error("Training error: {0}")]
    Training(String),
}

pub type Result<T> = std::result::Result<T, IrodoriError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_error_display() {
        let e = IrodoriError::Config("bad value".into());
        assert_eq!(e.to_string(), "Invalid configuration: bad value");
    }

    #[test]
    fn training_error_display() {
        let e = IrodoriError::Training("missing checkpoint".into());
        assert_eq!(e.to_string(), "Training error: missing checkpoint");
    }

    #[test]
    fn io_error_from_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let e: IrodoriError = io_err.into();
        assert!(matches!(e, IrodoriError::Io(_)));
        assert!(e.to_string().contains("file missing"));
    }

    #[test]
    fn json_error_from_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("{{bad").unwrap_err();
        let e: IrodoriError = json_err.into();
        assert!(matches!(e, IrodoriError::Json(_)));
    }

    #[test]
    fn all_variants_implement_debug() {
        let e = IrodoriError::Shape("mismatch".into());
        let debug = format!("{e:?}");
        assert!(debug.contains("Shape"));
    }

    #[test]
    fn result_type_alias_works() {
        let ok: Result<i32> = Ok(42);
        assert_eq!(*ok.as_ref().expect("should be Ok"), 42);

        let err: Result<i32> = Err(IrodoriError::MissingInput("x".into()));
        assert!(err.is_err());
    }
}
