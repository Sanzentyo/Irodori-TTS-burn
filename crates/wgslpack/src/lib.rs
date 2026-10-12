//! A Sans-I/O tensor format: validated borrowed views and an optional layout writer.
//!
//! Parsing never copies tensor payloads or performs I/O. Applications own storage,
//! uploads, hashes and resource lifetimes. Descriptors have the exact scalar-array
//! layout in [`WGSL`]; tensor payloads are little-endian and contiguous.
#![no_std]
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
mod layout;
#[cfg(feature = "alloc")]
pub use layout::{Layout, TensorSpec};

use core::{fmt, ops::Range};

pub const HEADER_SIZE: usize = 64;
pub const DESCRIPTOR_SIZE: usize = 96;
pub const ALIGNMENT: usize = 256;
pub const MAX_RANK: usize = 8;
pub const WGSL: &str = include_str!("descriptor.wgsl");
const MAGIC: &[u8; 8] = b"WGSLPK01";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Dtype {
    F32 = 1,
    F16 = 2,
    Bf16 = 3,
    I32 = 4,
    U32 = 5,
}

impl Dtype {
    pub const fn size(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            _ => 4,
        }
    }
    fn decode(value: u32, offset: usize) -> Result<Self, Error> {
        match value {
            1 => Ok(Self::F32),
            2 => Ok(Self::F16),
            3 => Ok(Self::Bf16),
            4 => Ok(Self::I32),
            5 => Ok(Self::U32),
            _ => Err(Error::new(ErrorKind::Dtype, offset)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    Truncated,
    Magic,
    Version,
    Reserved,
    Overflow,
    Bounds,
    Dtype,
    Shape,
    Name,
    Layout,
}

/// Structural errors never require allocation, including with `alloc` disabled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Error {
    pub kind: ErrorKind,
    pub offset: usize,
}

impl Error {
    const fn new(kind: ErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "wgslpack {:?} at byte {}", self.kind, self.offset)
    }
}
impl core::error::Error for Error {}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated field range"),
    )
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated field range"),
    )
}
fn aligned(value: u64) -> Result<u64, Error> {
    value
        .checked_add(255)
        .map(|x| x & !255)
        .ok_or(Error::new(ErrorKind::Overflow, 0))
}
fn host(value: u64, offset: usize) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| Error::new(ErrorKind::Overflow, offset))
}

/// Read just the fixed header to learn how much index data an I/O adapter needs.
#[derive(Clone, Copy, Debug)]
pub struct Header {
    count: usize,
    names_len: usize,
    metadata_len: usize,
    prefix_len: usize,
    total_len: u64,
}

impl Header {
    pub fn parse(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() < HEADER_SIZE {
            return Err(Error::new(ErrorKind::Truncated, bytes.len()));
        }
        if &bytes[..8] != MAGIC {
            return Err(Error::new(ErrorKind::Magic, 0));
        }
        if u32_at(bytes, 8) != 1 || u32_at(bytes, 12) != DESCRIPTOR_SIZE as u32 {
            return Err(Error::new(ErrorKind::Version, 8));
        }
        if u32_at(bytes, 28) != 0 || bytes[48..64].iter().any(|&x| x != 0) {
            return Err(Error::new(ErrorKind::Reserved, 28));
        }
        let count = u32_at(bytes, 16) as usize;
        let names_len = u32_at(bytes, 20) as usize;
        let metadata_len = u32_at(bytes, 24) as usize;
        let prefix_len = host(u64_at(bytes, 32), 32)?;
        let total_len = u64_at(bytes, 40);
        let unaligned = HEADER_SIZE as u64
            + count as u64 * DESCRIPTOR_SIZE as u64
            + names_len as u64
            + metadata_len as u64;
        if aligned(unaligned)? != prefix_len as u64 || total_len < prefix_len as u64 {
            return Err(Error::new(ErrorKind::Bounds, 32));
        }
        Ok(Self {
            count,
            names_len,
            metadata_len,
            prefix_len,
            total_len,
        })
    }
    pub const fn prefix_len(self) -> usize {
        self.prefix_len
    }
    pub const fn total_len(self) -> u64 {
        self.total_len
    }
    pub const fn tensor_count(self) -> usize {
        self.count
    }
}

/// Validated index; names, metadata and descriptors borrow the supplied bytes.
/// Payload bytes may live in a separate file, mapping or remote range source.
#[derive(Clone, Copy, Debug)]
pub struct Index<'a> {
    bytes: &'a [u8],
    header: Header,
}

impl<'a> Index<'a> {
    pub fn parse(bytes: &'a [u8], file_len: u64) -> Result<Self, Error> {
        let header = Header::parse(bytes)?;
        if header.total_len != file_len {
            return Err(Error::new(ErrorKind::Bounds, 40));
        }
        let bytes = bytes
            .get(..header.prefix_len)
            .ok_or(Error::new(ErrorKind::Truncated, bytes.len()))?;
        let index = Self { bytes, header };
        let mut payload_end = header.prefix_len as u64;
        let mut name_end = 0usize;
        let mut previous = "";
        for number in 0..header.count {
            let offset = HEADER_SIZE + number * DESCRIPTOR_SIZE;
            let raw = &bytes[offset..offset + DESCRIPTOR_SIZE];
            let dtype = Dtype::decode(u32_at(raw, 0), offset)?;
            let rank = u32_at(raw, 4) as usize;
            if rank > MAX_RANK {
                return Err(Error::new(ErrorKind::Shape, offset + 4));
            }
            let name_start = u32_at(raw, 8) as usize;
            let name_len = u32_at(raw, 12) as usize;
            if name_start != name_end {
                return Err(Error::new(ErrorKind::Layout, offset + 8));
            }
            name_end = name_start
                .checked_add(name_len)
                .filter(|&x| x <= header.names_len)
                .ok_or(Error::new(ErrorKind::Bounds, offset + 8))?;
            let name = core::str::from_utf8(&index.names()[name_start..name_end])
                .map_err(|_| Error::new(ErrorKind::Name, offset + 8))?;
            if name.is_empty() || name <= previous || name.as_bytes().contains(&0) {
                return Err(Error::new(ErrorKind::Name, offset + 8));
            }
            previous = name;
            let mut elements = 1u64;
            for axis in (0..MAX_RANK).rev() {
                let dimension = u32_at(raw, 32 + axis * 4);
                let stride = u32_at(raw, 64 + axis * 4);
                if axis >= rank {
                    if dimension != 0 || stride != 0 {
                        return Err(Error::new(ErrorKind::Shape, offset + 32 + axis * 4));
                    }
                } else {
                    if dimension == 0 || stride as u64 != elements {
                        return Err(Error::new(ErrorKind::Shape, offset + 32 + axis * 4));
                    }
                    elements = elements
                        .checked_mul(dimension as u64)
                        .filter(|&x| x <= u32::MAX as u64)
                        .ok_or(Error::new(ErrorKind::Overflow, offset + 32 + axis * 4))?;
                }
            }
            let data_start = u64_at(raw, 16);
            let data_len = u64_at(raw, 24);
            if data_start != payload_end || data_len != elements * dtype.size() as u64 {
                return Err(Error::new(ErrorKind::Layout, offset + 16));
            }
            payload_end = aligned(
                data_start
                    .checked_add(data_len)
                    .ok_or(Error::new(ErrorKind::Overflow, offset + 16))?,
            )?;
            if payload_end > file_len {
                return Err(Error::new(ErrorKind::Bounds, offset + 16));
            }
        }
        if name_end != header.names_len || payload_end != file_len {
            return Err(Error::new(ErrorKind::Layout, 40));
        }
        let metadata_end =
            HEADER_SIZE + header.count * DESCRIPTOR_SIZE + header.names_len + header.metadata_len;
        if bytes[metadata_end..].iter().any(|&x| x != 0) {
            return Err(Error::new(ErrorKind::Reserved, metadata_end));
        }
        Ok(index)
    }
    pub const fn header(self) -> Header {
        self.header
    }
    fn names(self) -> &'a [u8] {
        let start = HEADER_SIZE + self.header.count * DESCRIPTOR_SIZE;
        &self.bytes[start..start + self.header.names_len]
    }
    pub fn metadata(self) -> &'a [u8] {
        let start = HEADER_SIZE + self.header.count * DESCRIPTOR_SIZE + self.header.names_len;
        &self.bytes[start..start + self.header.metadata_len]
    }
    pub fn descriptor_bytes(self) -> &'a [u8] {
        &self.bytes[HEADER_SIZE..HEADER_SIZE + self.header.count * DESCRIPTOR_SIZE]
    }
    pub fn tensor(self, number: usize) -> Option<Descriptor<'a>> {
        if number >= self.header.count {
            return None;
        }
        let offset = HEADER_SIZE + number * DESCRIPTOR_SIZE;
        let raw = &self.bytes[offset..offset + DESCRIPTOR_SIZE];
        let start = u32_at(raw, 8) as usize;
        let end = start + u32_at(raw, 12) as usize;
        Some(Descriptor {
            raw,
            name: core::str::from_utf8(&self.names()[start..end]).expect("validated UTF-8"),
        })
    }
    pub fn tensors(self) -> impl ExactSizeIterator<Item = Descriptor<'a>> {
        (0..self.header.count).map(move |i| self.tensor(i).expect("index below count"))
    }
    pub fn get(self, name: &str) -> Option<Descriptor<'a>> {
        let mut start = 0;
        let mut end = self.header.count;
        while start < end {
            let mid = start + (end - start) / 2;
            let entry = self.tensor(mid).expect("binary search bounded by count");
            match entry.name().cmp(name) {
                core::cmp::Ordering::Equal => return Some(entry),
                core::cmp::Ordering::Less => start = mid + 1,
                core::cmp::Ordering::Greater => end = mid,
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Descriptor<'a> {
    raw: &'a [u8],
    name: &'a str,
}
impl<'a> Descriptor<'a> {
    pub const fn name(self) -> &'a str {
        self.name
    }
    pub const fn bytes(self) -> &'a [u8] {
        self.raw
    }
    pub fn dtype(self) -> Dtype {
        Dtype::decode(u32_at(self.raw, 0), 0).expect("validated dtype")
    }
    pub fn rank(self) -> usize {
        u32_at(self.raw, 4) as usize
    }
    pub fn shape(self) -> impl ExactSizeIterator<Item = u32> + 'a {
        (0..self.rank()).map(move |i| u32_at(self.raw, 32 + i * 4))
    }
    pub fn strides(self) -> impl ExactSizeIterator<Item = u32> + 'a {
        (0..self.rank()).map(move |i| u32_at(self.raw, 64 + i * 4))
    }
    pub fn data_range(self) -> Range<u64> {
        let start = u64_at(self.raw, 16);
        start..start + u64_at(self.raw, 24)
    }
    /// Four-byte storage extent, suitable for an `array<u32>` WGSL binding.
    pub fn storage_range(self) -> Range<u64> {
        let range = self.data_range();
        range.start..((range.end + 3) & !3)
    }
}

/// Immutable, zero-copy view over a complete in-memory archive.
#[derive(Clone, Copy, Debug)]
pub struct Archive<'a> {
    index: Index<'a>,
    bytes: &'a [u8],
}
impl<'a> Archive<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        Ok(Self {
            index: Index::parse(bytes, bytes.len() as u64)?,
            bytes,
        })
    }
    pub const fn index(self) -> Index<'a> {
        self.index
    }
    pub fn data(self, number: usize) -> Option<&'a [u8]> {
        let range = self.index.tensor(number)?.data_range();
        Some(&self.bytes[range.start as usize..range.end as usize])
    }
    pub fn storage(self, number: usize) -> Option<&'a [u8]> {
        let range = self.index.tensor(number)?.storage_range();
        Some(&self.bytes[range.start as usize..range.end as usize])
    }
}
