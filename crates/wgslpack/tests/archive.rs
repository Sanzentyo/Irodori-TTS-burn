#![cfg(feature = "alloc")]

use wgslpack::{Archive, Dtype, ErrorKind, Header, Index, Layout, TensorSpec};

fn archive() -> Vec<u8> {
    let specs = [
        TensorSpec {
            name: "z.weight",
            dtype: Dtype::F32,
            shape: &[2, 3],
        },
        TensorSpec {
            name: "a.scalar",
            dtype: Dtype::F16,
            shape: &[],
        },
    ];
    let layout = Layout::new(&specs, br#"{"model":"example"}"#).unwrap();
    let mut bytes = vec![0; layout.total_len() as usize];
    bytes[..layout.prefix().len()].copy_from_slice(layout.prefix());
    for tensor in layout.index().tensors() {
        let range = tensor.data_range();
        for (i, byte) in bytes[range.start as usize..range.end as usize]
            .iter_mut()
            .enumerate()
        {
            *byte = i as u8;
        }
    }
    bytes
}

#[test]
fn borrowed_views_and_wgsl_storage_cover_the_same_bytes() {
    let bytes = archive();
    let view = Archive::parse(&bytes).unwrap();
    let index = view.index();
    assert_eq!(index.metadata(), br#"{"model":"example"}"#);
    assert_eq!(
        index.tensors().map(|x| x.name()).collect::<Vec<_>>(),
        ["a.scalar", "z.weight"]
    );
    let weight = index.get("z.weight").unwrap();
    assert_eq!(weight.shape().collect::<Vec<_>>(), [2, 3]);
    assert_eq!(weight.strides().collect::<Vec<_>>(), [3, 1]);
    assert_eq!(weight.dtype(), Dtype::F32);
    assert_eq!(view.data(1).unwrap(), &(0..24).collect::<Vec<u8>>());
    assert_eq!(
        view.data(1).unwrap().as_ptr(),
        bytes[weight.data_range().start as usize..].as_ptr()
    );
    assert_eq!(view.data(0).unwrap().len(), 2);
    assert_eq!(view.storage(0).unwrap(), &[0, 1, 0, 0]);
    assert!(index.get("absent").is_none());
    assert!(view.data(2).is_none());
    assert_eq!(weight.bytes().len(), 96);
}

#[test]
fn io_adapter_can_parse_only_the_prefix() {
    let bytes = archive();
    let header = Header::parse(&bytes[..64]).unwrap();
    let index = Index::parse(&bytes[..header.prefix_len()], bytes.len() as u64).unwrap();
    assert_eq!(index.tensors().count(), 2);
    assert_eq!(index.get("z.weight").unwrap().data_range().end, 792);
    assert_eq!(
        Index::parse(&bytes, bytes.len() as u64 + 1)
            .unwrap_err()
            .kind,
        ErrorKind::Bounds
    );
}

#[test]
fn every_truncated_prefix_or_payload_is_rejected() {
    let bytes = archive();
    for end in 0..bytes.len() {
        assert!(Archive::parse(&bytes[..end]).is_err(), "length {end}");
    }
}

#[test]
fn rejects_overlap_bad_shapes_names_versions_and_overflow() {
    let original = archive();
    // Header length, reserved words, type, rank, name reference, payload range,
    // tensor byte count, shape and contiguous stride must all be validated.
    for (at, value) in [
        (8, 2u32),
        (12, 64),
        (16, u32::MAX),
        (20, u32::MAX),
        (28, 1),
        (48, 1),
        (64, 99),
        (68, 9),
        (72, u32::MAX),
        (76, 0),
        (80, 0),
        (88, 3),
        (160 + 16, 512),
        (160 + 32, u32::MAX),
        (160 + 64, 1),
        (160 + 68, 3),
        (160 + 40, 1),
    ] {
        let mut bytes = original.clone();
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        assert!(Archive::parse(&bytes).is_err(), "offset {at}");
    }
    let mut bytes = original.clone();
    bytes[256] = 0xff; // Names begin after the two descriptors.
    assert_eq!(Archive::parse(&bytes).unwrap_err().kind, ErrorKind::Name);
    bytes = original;
    bytes[0] ^= 1;
    assert_eq!(Archive::parse(&bytes).unwrap_err().kind, ErrorKind::Magic);
}

#[test]
fn layout_is_canonical_without_payload_allocation() {
    let a = TensorSpec {
        name: "a",
        dtype: Dtype::F32,
        shape: &[1024, 1024],
    };
    let b = TensorSpec {
        name: "b",
        dtype: Dtype::Bf16,
        shape: &[3],
    };
    let lhs = Layout::new(&[a, b], b"opaque").unwrap();
    let rhs = Layout::new(&[b, a], b"opaque").unwrap();
    assert_eq!(lhs.prefix(), rhs.prefix());
    assert_eq!(lhs.prefix().len(), 512);
    assert!(lhs.total_len() > 4 * 1024 * 1024);
    assert!(Layout::new(&[a, a], b"").is_err());
    assert!(Layout::new(&[TensorSpec { name: "", ..a }], b"").is_err());
    assert!(Layout::new(&[TensorSpec { shape: &[0], ..a }], b"").is_err());
    assert!(
        Layout::new(
            &[TensorSpec {
                shape: &[u32::MAX, 2],
                ..a
            }],
            b""
        )
        .is_err()
    );
}

#[test]
fn malformed_inputs_do_not_panic() {
    let original = archive();
    let mut state = 1u64;
    for _ in 0..10000 {
        let mut bytes = original.clone();
        for _ in 0..4 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let pos = (state >> 32) as usize % bytes.len();
            bytes[pos] ^= state as u8;
        }
        let _ = Archive::parse(&bytes);
    }
}

#[test]
fn empty_archive_and_all_dtypes_roundtrip() {
    let empty = Layout::new(&[], b"metadata").unwrap();
    assert_eq!(
        Archive::parse(empty.prefix())
            .unwrap()
            .index()
            .tensors()
            .count(),
        0
    );
    for dtype in [Dtype::F32, Dtype::F16, Dtype::Bf16, Dtype::I32, Dtype::U32] {
        let layout = Layout::new(
            &[TensorSpec {
                name: "x",
                dtype,
                shape: &[3],
            }],
            b"",
        )
        .unwrap();
        assert_eq!(layout.index().get("x").unwrap().dtype(), dtype);
        let range = layout.index().get("x").unwrap().data_range();
        assert_eq!(range.end - range.start, (3 * dtype.size()) as u64);
    }
}
