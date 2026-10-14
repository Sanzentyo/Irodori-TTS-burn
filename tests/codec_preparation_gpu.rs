//! Explicit device acceptance checks, intentionally not part of CPU validation.
#![cfg(all(feature = "streaming-graphs", feature = "packed-export"))]

use burn::{
    backend::wgpu::WgpuDevice,
    tensor::{Bytes, DType, Device, Tensor, TensorData},
};
use irodori_tts_burn::{
    WgpuFloatPrecision, backend_config,
    codec::{self, DecodeWindow, DecoderGraphLimits, StreamingDacVaeDecoder},
};
use std::{num::NonZeroUsize, ops::ControlFlow, path::Path};

fn latent(device: &Device, frames: usize, seed: usize) -> Tensor<3> {
    let values = (0..frames * 32)
        .map(|i| ((i + seed) as f32 * 0.13).sin() * 0.1)
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [1, frames, 32]), device)
}

fn bits(values: &[f32]) -> Vec<u32> {
    assert!(values.iter().all(|x| x.is_finite()));
    values.iter().map(|x| x.to_bits()).collect()
}

#[test]
#[ignore = "requires GPU and IRODORI_CODEC_TEST_CHECKPOINT; set IRODORI_PACK_TEST_INTEGRATED=1 for Intel"]
fn packed_codec_and_bounded_stream_graphs_match_ordinary_execution() {
    let checkpoint = std::env::var_os("IRODORI_CODEC_TEST_CHECKPOINT")
        .expect("set source FP32 codec checkpoint");
    let selected = if std::env::var_os("IRODORI_PACK_TEST_INTEGRATED").is_some() {
        cubecl::wgpu::WgpuDeviceKind::IntegratedGpu(0)
    } else {
        cubecl::wgpu::WgpuDeviceKind::DiscreteGpu(0)
    };
    let directory = tempfile::tempdir().unwrap();
    backend_config::configure_cubecl_persistent_cache_for_precision(
        directory.path().join("cache"),
        WgpuFloatPrecision::Fp32,
    )
    .unwrap();
    let device = backend_config::wgpu_device_with_precision(
        &WgpuDevice::new(selected),
        WgpuFloatPrecision::Fp32,
    )
    .unwrap();
    let packed = directory.path().join("codec.wgslpack");
    codec::packed::pack_decoder(Path::new(&checkpoint), &packed).unwrap();
    let mut ordinary =
        codec::load_decoder_with_float_dtype(Path::new(&checkpoint), &device, DType::F32).unwrap();
    ordinary.prepare_for_wgsl();
    assert_eq!(ordinary.context_frames(), 8);
    let input = latent(&device, 2, 1);
    let expected = ordinary
        .decode_wgsl(input.clone())
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    for owned in [false, true] {
        let bytes = std::fs::read(&packed).unwrap();
        let mut decoder = if owned {
            codec::packed::load_decoder_from_owned_bytes(
                Bytes::from_elems(bytes),
                &device,
                DType::F32,
            )
            .unwrap()
        } else {
            codec::packed::load_decoder_from_bytes(&bytes, &device, DType::F32).unwrap()
        };
        decoder.prepare_for_wgsl();
        let actual = decoder
            .decode_wgsl(input.clone())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(bits(&actual), bits(&expected));
    }
    let mut decoder = codec::load_decoder(&packed, &device).unwrap();
    decoder.prepare_for_wgsl();
    let mut captured = StreamingDacVaeDecoder::new(
        decoder,
        DecoderGraphLimits::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(25).unwrap(),
        )
        .unwrap(),
    );
    let window = DecodeWindow {
        first_frames: 3,
        subsequent_frames: 5,
    };
    // Change values, geometry and final crop; revisit a prior geometry after
    // eviction, so stale captured input/output cannot satisfy this comparison.
    for (frames, seed, trim) in [(2, 1, 0), (13, 3, 317), (40, 7, 1), (13, 11, 0)] {
        let input = latent(&device, frames, seed);
        let limit = frames * codec::DACVAE_HOP_LENGTH - trim;
        let whole = ordinary
            .decode_wgsl(input.clone())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let mut expected = Vec::new();
        ordinary
            .decode_stream(input.clone(), Some(limit), window, |chunk| {
                expected.extend_from_slice(chunk);
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        assert_eq!(bits(&expected), bits(&whole[..limit]));
        let mut actual = Vec::new();
        captured
            .decode_stream(input, Some(limit), window, |chunk| {
                actual.extend_from_slice(chunk);
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        assert_eq!(bits(&actual), bits(&expected));
        assert!(captured.cached_graphs() <= 2);
    }
    let report = captured
        .decode_stream(latent(&device, 40, 19), None, window, |_| {
            Ok(ControlFlow::Break(()))
        })
        .unwrap();
    assert!(report.cancelled);
    assert_eq!(report.chunks, 1);
    let mut emitted = false;
    assert!(
        captured
            .decode_stream(
                latent(&device, 40, 23),
                None,
                DecodeWindow::default(),
                |_| {
                    emitted = true;
                    Ok(ControlFlow::Continue(()))
                }
            )
            .is_err()
    );
    assert!(!emitted);
    cubecl_environment::environment::activate("codec-graph-generation-check");
    let input = latent(&device, 2, 31);
    let expected = ordinary
        .decode_wgsl(input.clone())
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let mut actual = Vec::new();
    captured
        .decode_stream(input, None, window, |chunk| {
            actual.extend_from_slice(chunk);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(bits(&actual), bits(&expected));
    assert_eq!(
        captured.cached_graphs(),
        1,
        "old environment graphs were released"
    );
    drop(captured);

    // The production 96-frame interior now decodes 112 input frames. Check this
    // geometry against whole decoding as well as the smaller windows above,
    // including a single latent frame at the beginning.
    let mut captured = StreamingDacVaeDecoder::new(
        ordinary.clone(),
        DecoderGraphLimits::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(96 + 2 * ordinary.context_frames()).unwrap(),
        )
        .unwrap(),
    );
    let input = latent(&device, 130, 37);
    let whole = ordinary
        .decode_wgsl(input.clone())
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    for (first_frames, trim) in [(1, 0), (12, 317)] {
        let limit = whole.len() - trim;
        let mut actual = Vec::new();
        captured
            .decode_stream(
                input.clone(),
                Some(limit),
                DecodeWindow {
                    first_frames,
                    subsequent_frames: 96,
                },
                |chunk| {
                    actual.extend_from_slice(chunk);
                    Ok(ControlFlow::Continue(()))
                },
            )
            .unwrap();
        assert_eq!(bits(&actual), bits(&whole[..limit]));
        assert!(captured.cached_graphs() <= 2);
    }
    device.sync().unwrap();
}
