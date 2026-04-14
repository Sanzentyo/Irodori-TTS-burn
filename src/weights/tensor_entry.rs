//! Per-tensor metadata and raw byte handling from safetensors checkpoints.

use burn::tensor::TensorData;
use half::{bf16, f16};
use safetensors::Dtype;

use crate::error::{IrodoriError, Result};

/// Raw bytes and metadata for a single safetensors tensor.
pub(super) struct TensorEntry {
    /// Little-endian raw bytes as stored in the safetensors file.
    pub(super) bytes: Vec<u8>,
    /// Element dtype of the stored bytes.
    pub(super) dtype: Dtype,
    /// Shape dimensions (innermost-last / row-major).
    pub(super) shape: Vec<usize>,
}

impl TensorEntry {
    pub(super) fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub(super) fn dtype_byte_size(dtype: Dtype) -> usize {
        match dtype {
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
            Dtype::I8 | Dtype::U8 | Dtype::BOOL => 1,
            _ => 4, // fallback; validated separately
        }
    }

    /// Validate that `bytes.len() == numel * dtype_byte_size`.
    pub(super) fn validate_byte_len(&self, key: &str) -> Result<()> {
        let expected = self.numel() * Self::dtype_byte_size(self.dtype);
        if self.bytes.len() != expected {
            return Err(IrodoriError::Weight(format!(
                "{key}: byte length mismatch — expected {expected}, got {}",
                self.bytes.len()
            )));
        }
        Ok(())
    }

    /// Decode the raw bytes into a [`TensorData`] using the checkpoint's native dtype.
    ///
    /// Burn will transparently convert to the backend's float element type when
    /// `Tensor::from_data` is called, so a bf16 checkpoint loaded into an f32
    /// backend incurs the expected bf16→f32 conversion there, not here.
    ///
    /// When the checkpoint dtype matches the backend dtype (e.g. bf16 checkpoint
    /// into bf16 backend), burn skips the conversion and does a direct copy.
    pub(super) fn to_tensor_data<const D: usize>(&self, key: &str) -> Result<TensorData> {
        if self.shape.len() != D {
            return Err(IrodoriError::WrongDim(key.to_string(), D, self.shape.len()));
        }
        let shape_arr: [usize; D] = self.shape[..].try_into().expect("D validated above");

        let td = match self.dtype {
            Dtype::F32 => {
                let data: Vec<f32> = self
                    .bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                TensorData::new(data, shape_arr)
            }
            Dtype::BF16 => {
                let data: Vec<bf16> = self
                    .bytes
                    .chunks_exact(2)
                    .map(|b| bf16::from_le_bytes([b[0], b[1]]))
                    .collect();
                TensorData::new(data, shape_arr)
            }
            Dtype::F16 => {
                let data: Vec<f16> = self
                    .bytes
                    .chunks_exact(2)
                    .map(|b| f16::from_le_bytes([b[0], b[1]]))
                    .collect();
                TensorData::new(data, shape_arr)
            }
            other => return Err(IrodoriError::Dtype(key.to_string(), format!("{other:?}"))),
        };
        Ok(td)
    }

    /// Decode raw bytes to `Vec<f32>` for arithmetic operations (e.g. LoRA merge).
    #[cfg(feature = "lora")]
    pub(super) fn to_f32_vec(&self, key: &str) -> Result<Vec<f32>> {
        match self.dtype {
            Dtype::F32 => Ok(self
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            Dtype::BF16 => Ok(self
                .bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()),
            Dtype::F16 => Ok(self
                .bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f16::from_bits(bits).to_f32()
                })
                .collect()),
            other => Err(IrodoriError::Dtype(key.to_string(), format!("{other:?}"))),
        }
    }
}

/// Re-encode a `Vec<f32>` back to the target safetensors `Dtype`.
#[cfg(feature = "lora")]
pub(super) fn encode_f32_to_dtype(data: &[f32], dtype: Dtype, key: &str) -> Result<Vec<u8>> {
    match dtype {
        Dtype::F32 => Ok(data.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Dtype::BF16 => Ok(data
            .iter()
            .flat_map(|v| {
                let bits = (v.to_bits() >> 16) as u16;
                bits.to_le_bytes()
            })
            .collect()),
        Dtype::F16 => Ok(data
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect()),
        other => Err(IrodoriError::Dtype(key.to_string(), format!("{other:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::test_helpers::*;
    use burn::{backend::NdArray, tensor::Tensor};
    use safetensors::Dtype;

    type B = NdArray<f32>;

    #[test]
    fn tensor_entry_validate_f32_ok() {
        let entry = TensorEntry {
            bytes: f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            dtype: Dtype::F32,
            shape: vec![2, 3],
        };
        entry.validate_byte_len("test").unwrap();
    }

    #[test]
    fn tensor_entry_validate_byte_mismatch() {
        let entry = TensorEntry {
            bytes: vec![0u8; 10], // wrong: 2*3*4=24 expected
            dtype: Dtype::F32,
            shape: vec![2, 3],
        };
        assert!(entry.validate_byte_len("test").is_err());
    }

    #[test]
    fn tensor_entry_to_tensor_data_f32() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let entry = TensorEntry {
            bytes: f32_bytes(&vals),
            dtype: Dtype::F32,
            shape: vec![2, 2],
        };
        let td = entry.to_tensor_data::<2>("test").unwrap();
        let t = Tensor::<B, 2>::from_data(td, &Default::default());
        let data = t.to_data();
        let result: Vec<f32> = data.to_vec().unwrap();
        assert_eq!(result, vals);
    }

    #[test]
    fn tensor_entry_to_tensor_data_bf16() {
        let vals = vec![1.0f32, -0.5, 3.125, 0.0];
        let entry = TensorEntry {
            bytes: bf16_bytes(&vals),
            dtype: Dtype::BF16,
            shape: vec![4],
        };
        let td = entry.to_tensor_data::<1>("test").unwrap();
        let t = Tensor::<B, 1>::from_data(td, &Default::default());
        let result: Vec<f32> = t.to_data().to_vec().unwrap();
        for (a, b) in result.iter().zip(vals.iter()) {
            assert!((a - b).abs() < 0.02, "bf16 decode: {a} vs {b}");
        }
    }

    #[test]
    fn tensor_entry_to_tensor_data_f16() {
        let vals = vec![0.25f32, -1.0, 2.0, 0.5];
        let entry = TensorEntry {
            bytes: f16_bytes(&vals),
            dtype: Dtype::F16,
            shape: vec![2, 2],
        };
        let td = entry.to_tensor_data::<2>("test").unwrap();
        let t = Tensor::<B, 2>::from_data(td, &Default::default());
        let result: Vec<f32> = t.to_data().to_vec().unwrap();
        for (a, b) in result.iter().zip(vals.iter()) {
            assert!((a - b).abs() < 0.01, "f16 decode: {a} vs {b}");
        }
    }

    #[test]
    fn tensor_entry_wrong_dim_error() {
        let entry = TensorEntry {
            bytes: f32_bytes(&[1.0, 2.0]),
            dtype: Dtype::F32,
            shape: vec![2],
        };
        // Requesting 2-D from a 1-D tensor should fail
        let err = entry.to_tensor_data::<2>("test");
        assert!(err.is_err());
    }

    // --- to_f32_vec + encode_f32_to_dtype roundtrip ---

    #[test]
    #[cfg(feature = "lora")]
    fn roundtrip_f32_encode_decode() {
        let vals = vec![1.5f32, -2.25, 0.0, 100.0];
        let entry = TensorEntry {
            bytes: f32_bytes(&vals),
            dtype: Dtype::F32,
            shape: vec![4],
        };
        let decoded = entry.to_f32_vec("test").unwrap();
        assert_eq!(decoded, vals);

        let re_encoded = encode_f32_to_dtype(&decoded, Dtype::F32, "test").unwrap();
        assert_eq!(re_encoded, entry.bytes);
    }

    #[test]
    #[cfg(feature = "lora")]
    fn roundtrip_bf16_encode_decode() {
        let vals = vec![1.0f32, -0.5, 3.125, 0.0];
        let entry = TensorEntry {
            bytes: bf16_bytes(&vals),
            dtype: Dtype::BF16,
            shape: vec![4],
        };
        let decoded = entry.to_f32_vec("test").unwrap();
        let re_encoded = encode_f32_to_dtype(&decoded, Dtype::BF16, "test").unwrap();
        assert_eq!(re_encoded, entry.bytes);
    }

    #[test]
    #[cfg(feature = "lora")]
    fn roundtrip_f16_encode_decode() {
        let vals = vec![0.25f32, -1.0, 2.0, 0.5];
        let entry = TensorEntry {
            bytes: f16_bytes(&vals),
            dtype: Dtype::F16,
            shape: vec![4],
        };
        let decoded = entry.to_f32_vec("test").unwrap();
        let re_encoded = encode_f32_to_dtype(&decoded, Dtype::F16, "test").unwrap();
        assert_eq!(re_encoded, entry.bytes);
    }

    #[test]
    #[cfg(feature = "lora")]
    fn encode_unsupported_dtype_errors() {
        let err = encode_f32_to_dtype(&[1.0], Dtype::I32, "test");
        assert!(err.is_err());
    }
}
