//! Accuracy-checked, device/shape-specific decoder transposed convolution.
use super::{
    conv_transpose1d_cached_col2im::{CachedCol2ImCase, conv_transpose1d_cached_col2im_wgsl},
    conv_transpose1d_polyphase::{
        self as polyphase, ConvTranspose1dStride, PackedConvTranspose1dWeight, PolyphaseTile,
    },
    tuning_validation::{check_accuracy, read},
};
use burn::{backend::wgpu::CubeTensor, tensor::DType};
use cubecl::{
    CubeTuneId,
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};
use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
struct Key {
    schema: u16,
    device: String,
    dtype: String,
    shape: [usize; 3],
    stride: usize,
    strides: Vec<usize>,
}
impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "transpose-v{}-{}-{}-{:?}-s{}-{:?}",
            self.schema, self.device, self.dtype, self.shape, self.stride, self.strides
        )
    }
}
impl cubecl::tune::AutotuneKey for Key {}
#[derive(Clone)]
struct Input {
    x: CubeTensor,
    weight: CubeTensor,
    packed: CubeTensor,
    bias: CubeTensor,
    case: CachedCol2ImCase,
    checks: Option<Arc<Vec<Result<(), String>>>>,
}
const CANDIDATES: &[(&str, Option<PolyphaseTile>)] = &[
    ("cached-col2im", None),
    ("wgsl-t32-o16-k16", Some(PolyphaseTile::PortableCin16)),
    ("wgsl-t64-o16-k32", Some(PolyphaseTile::Case0Cin32)),
    ("wgsl-t16-o16-k16", Some(PolyphaseTile::SmallCin16)),
    ("wgsl-t32-o16-k32", Some(PolyphaseTile::BalancedCin32)),
    ("wgsl-t32-o16-k64", Some(PolyphaseTile::BalancedCin64)),
    ("wgsl-t16-o32-k16", Some(PolyphaseTile::WideChannels)),
];
fn key(input: &Input) -> Key {
    Key {
        schema: 1,
        device: input.x.client.properties().identity.fingerprint.clone(),
        dtype: input.x.dtype.name().to_owned(),
        shape: input.x.meta.shape().dims::<3>(),
        stride: input.case.dimensions().2,
        strides: input.x.meta.strides().to_vec(),
    }
}
fn launch(tile: Option<PolyphaseTile>, input: &Input) -> Result<CubeTensor, String> {
    match tile {
        None => conv_transpose1d_cached_col2im_wgsl(
            input.x.clone(),
            input.weight.clone(),
            input.bias.clone(),
            input.case,
        )
        .map_err(|error| error.to_string()),
        Some(tile) => {
            let stride = ConvTranspose1dStride::try_from(input.case.dimensions().2)?;
            let packed = PackedConvTranspose1dWeight::from_tensor(input.packed.clone(), stride);
            polyphase::with_tile(input.x.clone(), &packed, input.bias.clone(), tile)
        }
    }
}
fn tuning_input(_: &Key, input: &Input) -> Input {
    let expected = launch(None, input)
        .and_then(read::<3>)
        .expect("admitted decoder reference");
    let half = input.x.dtype == DType::F16;
    check_accuracy(&expected, &expected, half).expect("decoder reference must be finite");
    let checks = CANDIDATES
        .iter()
        .map(|&(_, tile)| {
            launch(tile, input)
                .and_then(read::<3>)
                .and_then(|actual| check_accuracy(&expected, &actual, half))
        })
        .collect();
    Input {
        checks: Some(Arc::new(checks)),
        ..input.clone()
    }
}
pub(crate) fn transpose(
    x: CubeTensor,
    weight: CubeTensor,
    packed: CubeTensor,
    bias: CubeTensor,
    case: CachedCol2ImCase,
) -> CubeTensor {
    static TUNER: LocalTuner<Key, CubeTuneId> = local_tuner!("validated-conv-transpose-v1");
    let input = Input {
        x,
        weight,
        packed,
        bias,
        case,
        checks: None,
    };
    let id = CubeTuneId::new(&input.x.client, &input.x.device);
    let operations = TUNER.init(&id, || {
        CANDIDATES.iter().enumerate().fold(
            TunableSet::new(key, tuning_input).with_short_circuit(false),
            |set, (index, &(name, tile))| {
                set.with(Tunable::new(name, move |input: Input| {
                    if let Some(checks) = &input.checks {
                        checks[index].clone()?;
                    }
                    launch(tile, &input)
                }))
            },
        )
    });
    let client = input.x.client.clone();
    TUNER.execute(&id, &client, operations, input)
}
