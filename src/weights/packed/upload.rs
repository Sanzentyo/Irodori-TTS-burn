//! Group allocations without changing archive layout or tensor boundaries.

use std::{collections::BTreeMap, num::NonZeroUsize};

use burn::{
    backend::{DispatchDevice, wgpu::CubeTensor},
    tensor::{Bytes, DType, Device, Shape, TensorData},
};
use cubecl::server::{MemoryLayoutDescriptor, MemoryLayoutStrategy};
use wgslpack::Descriptor;

use crate::{IrodoriError, Result};

pub(super) fn tensors<'a>(
    descriptors: impl Iterator<Item = Descriptor<'a>>,
    device: &Device,
    dtype: DType,
    limit: NonZeroUsize,
    mut read: impl FnMut(Descriptor<'a>) -> Result<Bytes>,
) -> Result<BTreeMap<String, CubeTensor>> {
    // The production feature graph admits only the raw Cube backend.
    let DispatchDevice::Cube(device) = device.as_dispatch();
    let client = device.client();
    let mut output = BTreeMap::new();
    let mut pending = Vec::new();
    let mut pending_bytes = 0usize;
    let flush = |pending: &mut Vec<(String, TensorData)>,
                 output: &mut BTreeMap<String, CubeTensor>| {
        let (identities, uploads): (Vec<_>, Vec<_>) = pending
            .drain(..)
            .map(|(name, data)| {
                let (bytes, shape, dtype) = data.into_parts();
                (
                    (name, shape.clone(), dtype),
                    (
                        MemoryLayoutDescriptor::new(
                            MemoryLayoutStrategy::Contiguous,
                            shape,
                            dtype.size(),
                        ),
                        bytes,
                    ),
                )
            })
            .unzip();
        let allocations = client.create_tensors(uploads);
        for ((name, shape, dtype), allocation) in identities.into_iter().zip(allocations) {
            output.insert(
                name,
                CubeTensor::new_contiguous(
                    client.clone(),
                    device.clone(),
                    shape,
                    allocation.memory,
                    dtype,
                ),
            );
        }
    };
    for descriptor in descriptors {
        let shape = Shape::from(
            descriptor
                .shape()
                .map(|dim| dim as usize)
                .collect::<Vec<_>>(),
        );
        let bytes = read(descriptor)?;
        let length = bytes.len();
        if !pending.is_empty() && length > limit.get().saturating_sub(pending_bytes) {
            flush(&mut pending, &mut output);
            pending_bytes = 0;
        }
        let data = TensorData::try_from_bytes(bytes, shape, dtype).map_err(|error| {
            IrodoriError::Weight(format!("packed {}: {error}", descriptor.name()))
        })?;
        pending.push((descriptor.name().to_owned(), data));
        pending_bytes += length;
        if pending_bytes >= limit.get() {
            flush(&mut pending, &mut output);
            pending_bytes = 0;
        }
    }
    if !pending.is_empty() {
        flush(&mut pending, &mut output);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use wgslpack::{Dtype, Layout, TensorSpec};

    #[test]
    fn grouped_upload_preserves_odd_tensor_boundaries_and_dtypes() {
        let device =
            crate::backend_config::strict_fp32_device(&burn::backend::wgpu::WgpuDevice::default())
                .unwrap();
        for (native, wire) in [(DType::F32, Dtype::F32), (DType::F16, Dtype::F16)] {
            let shapes = [[3], [17], [5]];
            let names = ["first", "oversized", "tail"];
            let specs = names
                .iter()
                .zip(&shapes)
                .map(|(name, shape)| TensorSpec {
                    name,
                    dtype: wire,
                    shape,
                })
                .collect::<Vec<_>>();
            let layout = Layout::new(&specs, b"").unwrap();
            let payloads = names
                .into_iter()
                .zip(shapes)
                .enumerate()
                .map(|(index, (name, shape))| {
                    let values = (0..shape[0])
                        .map(|i| (index * 100 + i as usize) as f32 * 0.25)
                        .collect::<Vec<_>>();
                    (name, values)
                })
                .collect::<BTreeMap<_, _>>();
            for budget in [1, 40, 4096] {
                let mut uploaded = tensors(
                    layout.index().tensors(),
                    &device,
                    native,
                    NonZeroUsize::new(budget).unwrap(),
                    |descriptor| {
                        let values = &payloads[descriptor.name()];
                        let bytes = if native == DType::F16 {
                            values
                                .iter()
                                .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
                                .collect::<Vec<_>>()
                        } else {
                            values
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>()
                        };
                        Ok(Bytes::from_elems(bytes))
                    },
                )
                .unwrap();
                for name in names {
                    let tensor = Tensor::<1>::from_primitive::<crate::WgpuRaw>(
                        uploaded.remove(name).unwrap(),
                    );
                    assert_eq!(tensor.dtype(), native);
                    assert_eq!(
                        tensor
                            .cast(DType::F32)
                            .into_data()
                            .try_to_vec::<f32>()
                            .unwrap(),
                        payloads[name]
                    );
                }
                assert!(uploaded.is_empty());
            }
        }
        device.sync().unwrap();
    }
}
