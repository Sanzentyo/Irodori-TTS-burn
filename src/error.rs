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
}

pub type Result<T> = std::result::Result<T, IrodoriError>;
