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
}

pub type Result<T> = std::result::Result<T, IrodoriError>;
