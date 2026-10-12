use crate::{
    ALIGNMENT, DESCRIPTOR_SIZE, Dtype, Error, ErrorKind, HEADER_SIZE, Index, MAGIC, MAX_RANK,
    aligned, host,
};
use alloc::{vec, vec::Vec};

#[derive(Clone, Copy, Debug)]
pub struct TensorSpec<'a> {
    pub name: &'a str,
    pub dtype: Dtype,
    pub shape: &'a [u32],
}

/// Canonical prefix and payload ranges. The caller supplies payload bytes and I/O.
#[derive(Debug)]
pub struct Layout {
    prefix: Vec<u8>,
    total_len: u64,
}
fn write_u32(out: &mut [u8], at: usize, value: u32) {
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}
fn write_u64(out: &mut [u8], at: usize, value: u64) {
    out[at..at + 8].copy_from_slice(&value.to_le_bytes());
}
fn narrow(value: usize) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::new(ErrorKind::Overflow, 0))
}

impl Layout {
    pub fn new(specs: &[TensorSpec<'_>], metadata: &[u8]) -> Result<Self, Error> {
        let mut specs = specs.to_vec();
        specs.sort_unstable_by_key(|s| s.name);
        let count = narrow(specs.len())?;
        let names_len = specs.iter().try_fold(0usize, |n, spec| {
            n.checked_add(spec.name.len())
                .ok_or(Error::new(ErrorKind::Overflow, 0))
        })?;
        let names_len_u32 = narrow(names_len)?;
        let metadata_len = narrow(metadata.len())?;
        let names_start = HEADER_SIZE as u64 + count as u64 * DESCRIPTOR_SIZE as u64;
        let metadata_start = names_start + names_len as u64;
        let prefix_len = host(aligned(metadata_start + metadata.len() as u64)?, 32)?;
        let mut prefix = vec![0; prefix_len];
        prefix[..8].copy_from_slice(MAGIC);
        write_u32(&mut prefix, 8, 1);
        write_u32(&mut prefix, 12, DESCRIPTOR_SIZE as u32);
        write_u32(&mut prefix, 16, count);
        write_u32(&mut prefix, 20, names_len_u32);
        write_u32(&mut prefix, 24, metadata_len);
        write_u64(&mut prefix, 32, prefix_len as u64);
        let mut payload = prefix_len as u64;
        let mut name_pos = 0;
        let mut previous = "";
        for (i, spec) in specs.iter().enumerate() {
            let at = HEADER_SIZE + i * DESCRIPTOR_SIZE;
            if spec.name.is_empty() || spec.name == previous || spec.name.as_bytes().contains(&0) {
                return Err(Error::new(ErrorKind::Name, at + 8));
            }
            previous = spec.name;
            if spec.shape.len() > MAX_RANK {
                return Err(Error::new(ErrorKind::Shape, at + 4));
            }
            write_u32(&mut prefix, at, spec.dtype as u32);
            write_u32(&mut prefix, at + 4, spec.shape.len() as u32);
            write_u32(&mut prefix, at + 8, name_pos as u32);
            write_u32(&mut prefix, at + 12, spec.name.len() as u32);
            let mut elements = 1u32;
            for (axis, &dimension) in spec.shape.iter().enumerate().rev() {
                if dimension == 0 {
                    return Err(Error::new(ErrorKind::Shape, at + 32 + axis * 4));
                }
                write_u32(&mut prefix, at + 32 + axis * 4, dimension);
                write_u32(&mut prefix, at + 64 + axis * 4, elements);
                elements = elements
                    .checked_mul(dimension)
                    .ok_or(Error::new(ErrorKind::Overflow, at + 32 + axis * 4))?;
            }
            let len = elements as u64 * spec.dtype.size() as u64;
            write_u64(&mut prefix, at + 16, payload);
            write_u64(&mut prefix, at + 24, len);
            payload = aligned(
                payload
                    .checked_add(len)
                    .ok_or(Error::new(ErrorKind::Overflow, at + 16))?,
            )?;
            let start = names_start as usize + name_pos;
            prefix[start..start + spec.name.len()].copy_from_slice(spec.name.as_bytes());
            name_pos += spec.name.len();
        }
        write_u64(&mut prefix, 40, payload);
        prefix[metadata_start as usize..metadata_start as usize + metadata.len()]
            .copy_from_slice(metadata);
        debug_assert_eq!(prefix.len() % ALIGNMENT, 0);
        Ok(Self {
            prefix,
            total_len: payload,
        })
    }
    pub fn prefix(&self) -> &[u8] {
        &self.prefix
    }
    pub const fn total_len(&self) -> u64 {
        self.total_len
    }
    pub fn index(&self) -> Index<'_> {
        Index::parse(&self.prefix, self.total_len)
            .expect("layout construction validates all index invariants")
    }
}
