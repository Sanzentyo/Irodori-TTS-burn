//! Shared capture owner for fixed-shape WGPU dispatch plans.
//! Only a borrowed output guard can read reusable output storage.
use crate::{WgpuRaw, error::IrodoriError};
use burn::tensor::{FloatDType, Tensor};
use cubecl::client::Graph;
use cubecl_environment::stream::StreamId;

#[derive(Debug)]
pub(crate) struct CapturedTensorGraph {
    // Release recorded work before its stable input/output owners.
    graph: Graph,
    stable_input: Tensor<3>,
    output: Tensor<3>,
    stream: StreamId,
    pub(crate) input_dims: [usize; 3],
    pub(crate) output_dims: [usize; 3],
}

/// One replayed graph output whose storage remains owned by its graph.
///
/// The mutable borrow of the captured session prevents another replay until
/// this value is consumed. Only an owned CPU result is exposed, so reusable
/// graph storage cannot escape into application state.
#[derive(Debug)]
pub struct CapturedTensorOutput<'graph> {
    output: &'graph Tensor<3>,
    stream: StreamId,
}

/// Preparation also changes the client's capture state. Abort it on every
/// early return or unwind, including failures before recording has started.
struct CaptureWindow<'client> {
    client: Option<&'client cubecl::client::Client>,
}

impl CaptureWindow<'_> {
    fn finish(mut self) -> crate::Result<Graph> {
        self.client
            .take()
            .expect("capture window is armed until finalization")
            .stop_capture()
            .map_err(|error| {
                IrodoriError::Config(format!("tensor graph finalization failed: {error}"))
            })
    }
}

impl Drop for CaptureWindow<'_> {
    fn drop(&mut self) {
        if let Some(client) = self.client.take()
            && client.is_capturing()
            && let Err(error) = client.stop_capture()
        {
            // Stopping a prepared-but-unrecorded window returns no graph.
            // Preserve the original model error/panic; report the discarded
            // incomplete capture rather than panicking again during unwind.
            tracing::debug!(%error, "discarded incomplete tensor graph capture");
        }
    }
}

impl CapturedTensorOutput<'_> {
    /// Fixed output geometry produced by this replay.
    pub fn dims(&self) -> [usize; 3] {
        self.output.dims()
    }

    /// Complete final readback and return owned contiguous F32 values.
    pub fn to_cpu_f32(self) -> crate::error::Result<Vec<f32>> {
        self.stream.executes(|| {
            self.output
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|error| {
                    IrodoriError::Dtype("captured graph output".to_owned(), error.to_string())
                })
        })
    }
}

impl CapturedTensorGraph {
    pub(crate) fn decode_to_cpu_f32(
        &mut self,
        latent: Tensor<3>,
    ) -> crate::error::Result<Vec<f32>> {
        self.enqueue(latent)?.to_cpu_f32()
    }

    pub(crate) fn enqueue(
        &mut self,
        latent: Tensor<3>,
    ) -> crate::error::Result<CapturedTensorOutput<'_>> {
        if latent.dims() != self.input_dims {
            return Err(IrodoriError::Shape(format!(
                "captured graph expects latent {:?}, got {:?}",
                self.input_dims,
                latent.dims()
            )));
        }
        self.stream.executes(|| {
            copy_into_stable_input(&latent, &self.stable_input)?;
            // SAFETY: model/request owners and the graph arena retain every
            // binding. Mutable borrowing serializes access to reusable storage.
            // Refresh, replay and guarded readback all use the captured stream,
            // even when the session moves to another thread or scoped stream.
            unsafe { self.graph.replay() }
                .map_err(|error| IrodoriError::Execution(error.to_string()))
        })?;
        Ok(CapturedTensorOutput {
            output: &self.output,
            stream: self.stream,
        })
    }
}

pub(crate) fn capture_tensor_graph<F>(
    example: &Tensor<3>,
    decode: F,
) -> crate::error::Result<CapturedTensorGraph>
where
    F: Fn(Tensor<3>) -> crate::Result<Tensor<3>>,
{
    capture_tensor_graph_impl(example, decode, |_, _| Ok(())).map(|(graph, ())| graph)
}

/// Return the completed priming result along with the recorded graph. Reading
/// it before recording avoids replaying the first request just for its output.
#[cfg(feature = "streaming-graphs")]
pub(crate) fn capture_tensor_graph_with_initial_output<F>(
    example: &Tensor<3>,
    decode: F,
) -> crate::Result<(CapturedTensorGraph, Vec<f32>)>
where
    F: Fn(Tensor<3>) -> crate::Result<Tensor<3>>,
{
    capture_tensor_graph_impl(example, decode, |output, stream| {
        CapturedTensorOutput {
            output: &output,
            stream,
        }
        .to_cpu_f32()
    })
}

fn capture_tensor_graph_impl<F, R>(
    example: &Tensor<3>,
    decode: F,
    consume_priming: impl FnOnce(Tensor<3>, StreamId) -> crate::Result<R>,
) -> crate::Result<(CapturedTensorGraph, R)>
where
    F: Fn(Tensor<3>) -> crate::Result<Tensor<3>>,
{
    validate_input_geometry(example.dims())?;
    let primitive = example
        .clone()
        .try_into_primitive::<WgpuRaw>()
        .map_err(|_| IrodoriError::UnsupportedMode("tensor graph requires WGPU".to_owned()))?;
    // Obtain the device's ordinary client instead of inheriting a caller's
    // explicitly pinned input client. Tensor operations resolve this stream.
    let client = primitive.device.client();
    let stream = StreamId::current();
    let stable_input = client.memory_persistent_allocation((), |()| {
        Tensor::<3>::zeros(example.dims(), (&example.device(), example.dtype()))
    });
    copy_into_stable_input(example, &stable_input)?;
    sync_client(&client, "tensor graph input initialization")?;

    client.graph_prepare().map_err(|error| {
        IrodoriError::Config(format!("tensor graph preparation failed: {error}"))
    })?;
    let window = CaptureWindow {
        client: Some(&client),
    };
    let priming_output = decode(stable_input.clone())?.cast(FloatDType::F32);
    sync_client(&client, "tensor graph priming")?;
    // Finish readback (if requested) and release the priming tensor before
    // recording. The capture allocator must be able to reuse that storage.
    let initial_output = consume_priming(priming_output, stream)?;
    client
        .start_capture()
        .map_err(|error| IrodoriError::Config(format!("tensor graph capture failed: {error}")))?;
    // Public consumers promise owned F32 CPU values. Capture the final
    // conversion as part of that consumer boundary so a replay is the entire
    // fixed device graph rather than a graph followed by one host-enqueued
    // dispatch per request.
    let output = decode(stable_input.clone()).map(|output| output.cast(FloatDType::F32));
    let graph = window.finish()?;
    // Always end capture before propagating a model error.
    let output = output?;
    let output_dims = output.dims();
    // Capture owns its live arena. Release only unused pages left in the
    // ordinary allocator by the priming run before another shape is captured.
    client
        .memory_cleanup()
        .map_err(|error| crate::IrodoriError::Execution(error.to_string()))?;
    sync_client(&client, "tensor graph allocator cleanup")?;

    Ok((
        CapturedTensorGraph {
            graph,
            stable_input,
            output,
            stream,
            input_dims: example.dims(),
            output_dims,
        },
        initial_output,
    ))
}

pub(crate) fn validate_input_geometry(input_dims: [usize; 3]) -> crate::error::Result<()> {
    if input_dims.into_iter().any(|dim| dim == 0) {
        return Err(IrodoriError::Shape(format!(
            "captured graph dimensions must be non-zero, got {input_dims:?}"
        )));
    }
    Ok(())
}

fn copy_into_stable_input(source: &Tensor<3>, destination: &Tensor<3>) -> crate::error::Result<()> {
    let mut source = source
        .clone()
        .try_into_primitive::<WgpuRaw>()
        .map_err(|_| {
            IrodoriError::UnsupportedMode("tensor graph source requires WGPU".to_owned())
        })?;
    let destination = destination
        .clone()
        .try_into_primitive::<WgpuRaw>()
        .map_err(|_| {
            IrodoriError::UnsupportedMode("tensor graph destination requires WGPU".to_owned())
        })?;
    // The stable destination's client follows our scoped stream. Do not inherit
    // an input producer's explicitly pinned client for this graph-owned write.
    source.client = destination.client.clone();
    crate::kernels::contiguous_copy::copy_contiguous_into_wgsl(source, destination).map_err(
        |reason| IrodoriError::Config(format!("tensor graph input copy rejected: {reason}")),
    )
}

fn sync_client(client: &cubecl::client::Client, stage: &str) -> crate::error::Result<()> {
    cubecl::future::block_on(client.sync())
        .map_err(|error| IrodoriError::Config(format!("{stage} failed: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "streaming-graphs")]
    #[test]
    fn priming_output_is_owned_and_later_requests_use_fresh_input() {
        let device =
            crate::backend_config::strict_fp32_device(&burn::backend::wgpu::WgpuDevice::default())
                .unwrap();
        let example = Tensor::<3>::ones([1, 2, 8], &device);
        let failed: crate::Result<(CapturedTensorGraph, ())> = capture_tensor_graph_impl(
            &example,
            |input| Ok(input * 3.0 + 1.0),
            |_, _| Err(IrodoriError::Execution("injected readback failure".into())),
        );
        assert!(failed.is_err());
        let (mut graph, initial) = StreamId::allocate()
            .executes(|| {
                capture_tensor_graph_with_initial_output(&example, |input| Ok(input * 3.0 + 1.0))
            })
            .unwrap();
        assert_eq!(initial, vec![4.0; 16]);
        for value in [2.0, 7.0] {
            let input = StreamId::allocate().executes(|| Tensor::full([1, 2, 8], value, &device));
            let result = StreamId::allocate()
                .executes(|| graph.decode_to_cpu_f32(input))
                .unwrap();
            assert_eq!(result, vec![value * 3.0 + 1.0; 16]);
        }
        assert_eq!(initial, vec![4.0; 16]);
    }

    #[test]
    fn preparation_errors_and_panics_release_the_capture_window() {
        let device =
            crate::backend_config::strict_fp32_device(&burn::backend::wgpu::WgpuDevice::default())
                .unwrap();
        let example = Tensor::<3>::ones([1, 1, 4], &device);
        let client = example
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .unwrap()
            .client;
        for (failed_call, panic) in [(1, false), (1, true), (2, true)] {
            let calls = std::cell::Cell::new(0usize);
            let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                capture_tensor_graph(&example, |input| {
                    calls.set(calls.get() + 1);
                    if calls.get() == failed_call {
                        assert!(!panic, "injected capture panic");
                        return Err(IrodoriError::Execution("injected priming error".into()));
                    }
                    Ok(input + 1.0)
                })
            }));
            if panic {
                assert!(attempt.is_err());
            } else {
                assert!(attempt.unwrap().is_err());
            }
            assert!(!client.is_capturing());
            let mut graph = capture_tensor_graph(&example, |input| Ok(input + 2.0)).unwrap();
            assert_eq!(
                graph
                    .enqueue(example.clone())
                    .unwrap()
                    .to_cpu_f32()
                    .unwrap(),
                vec![3.0; 4]
            );
        }
    }

    #[test]
    fn refresh_and_readback_follow_capture_across_streams_and_threads() {
        let device =
            crate::backend_config::strict_fp32_device(&burn::backend::wgpu::WgpuDevice::default())
                .unwrap();
        let example = Tensor::<3>::zeros([1, 2, 8], &device);
        let mut graph = StreamId::allocate()
            .executes(|| capture_tensor_graph(&example, |input| Ok(input * 3.0 + 1.0)))
            .unwrap();
        let input = StreamId::allocate().executes(|| Tensor::full([1, 2, 8], 2.0, &device));
        let caller_stream = StreamId::current();
        let output = StreamId::allocate()
            .executes(|| graph.enqueue(input))
            .unwrap();
        let values = StreamId::allocate()
            .executes(|| output.to_cpu_f32())
            .unwrap();
        assert_eq!(values, vec![7.0; 16]);
        assert_eq!(StreamId::current(), caller_stream);

        std::thread::spawn(move || {
            let input = Tensor::full([1, 2, 8], 7.0, &device);
            let output = graph.enqueue(input).unwrap();
            let values = std::thread::scope(|scope| {
                scope
                    .spawn(move || output.to_cpu_f32())
                    .join()
                    .unwrap()
                    .unwrap()
            });
            assert_eq!(values, vec![22.0; 16]);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn rejected_replay_does_not_corrupt_the_next_request() {
        let device =
            crate::backend_config::strict_fp32_device(&burn::backend::wgpu::WgpuDevice::default())
                .unwrap();
        let tail = Tensor::<3>::zeros([1, 2, 3], &device);
        assert!(capture_tensor_graph(&tail, |input| Ok(input * 3.0)).is_err());
        let example = Tensor::<3>::zeros([1, 2, 8], &device);
        let mut graph = capture_tensor_graph(&example, |input| Ok(input * 3.0)).unwrap();
        assert!(graph.enqueue(Tensor::zeros([1, 4, 4], &device)).is_err());
        assert!(
            graph
                .enqueue(example.clone().cast(burn::tensor::DType::F16))
                .is_err()
        );
        let output = graph.enqueue(Tensor::ones([1, 2, 8], &device)).unwrap();
        assert_eq!(output.dims(), [1, 2, 8]);
        assert_eq!(output.to_cpu_f32().unwrap(), vec![3.0; 16]);
    }

    #[test]
    fn failed_capture_ends_recording_before_propagating_the_model_error() {
        let device =
            crate::backend_config::strict_fp32_device(&burn::backend::wgpu::WgpuDevice::default())
                .unwrap();
        let example = Tensor::<3>::ones([1, 1, 4], &device);
        let calls = std::cell::Cell::new(0usize);
        let result = capture_tensor_graph(&example, |input| {
            calls.set(calls.get() + 1);
            if calls.get() == 2 {
                return Err(IrodoriError::Execution("injected capture failure".into()));
            }
            Ok(input + 1.0)
        });
        assert!(result.is_err());
        let mut graph = capture_tensor_graph(&example, |input| Ok(input + 2.0)).unwrap();
        assert_eq!(
            graph.enqueue(example).unwrap().to_cpu_f32().unwrap(),
            vec![3.0; 4]
        );
    }
}
