//! Reader-only fixtures are intentionally independent of the optional writer.
use wgslpack::{Archive, Dtype, Header, Index};

fn scalar_archive() -> [u8; 512] {
    let mut bytes = [0u8; 512];
    bytes[..8].copy_from_slice(b"WGSLPK01");
    for (offset, value) in [(8, 1u32), (12, 96), (16, 1), (20, 1), (64, 1), (76, 1)] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (offset, value) in [(32, 256u64), (40, 512), (80, 256), (88, 4)] {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    bytes[160] = b'x';
    bytes[256..260].copy_from_slice(&42.0f32.to_le_bytes());
    bytes
}

#[test]
fn reader_borrows_a_scalar_without_the_writer_feature() {
    let bytes = scalar_archive();
    let header = Header::parse(&bytes[..64]).unwrap();
    let index = Index::parse(&bytes[..header.prefix_len()], bytes.len() as u64).unwrap();
    let descriptor = index.get("x").unwrap();
    assert_eq!(descriptor.dtype(), Dtype::F32);
    assert_eq!(descriptor.rank(), 0);
    assert_eq!(descriptor.data_range(), 256..260);
    let archive = Archive::parse(&bytes).unwrap();
    let payload = archive.data(0).unwrap();
    assert_eq!(payload.as_ptr(), bytes[256..].as_ptr());
    assert_eq!(payload, &42.0f32.to_le_bytes());
    assert_eq!(archive.storage(0), Some(payload));
}

#[test]
fn reader_rejects_truncation_and_invalid_offsets_without_the_writer_feature() {
    let mut bytes = scalar_archive();
    for end in 0..bytes.len() {
        assert!(Archive::parse(&bytes[..end]).is_err(), "length {end}");
    }
    bytes[80..88].copy_from_slice(&512u64.to_le_bytes());
    assert!(Archive::parse(&bytes).is_err());
}
