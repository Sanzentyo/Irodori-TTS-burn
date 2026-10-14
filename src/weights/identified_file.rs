//! Compute the original file identity while serving validated tensor ranges.
//! Hashing follows file order even when the consumer requests tensors out of
//! order. Already consumed bytes are never retained in a second CPU archive.

use sha2::{Digest, Sha256};
use std::io::{self, Read, Seek, SeekFrom};

pub(super) struct IdentifiedFile<R> {
    reader: R,
    length: u64,
    hashed: u64,
    digest: Sha256,
}

impl<R: Read + Seek> IdentifiedFile<R> {
    pub(super) fn new(reader: R, length: u64) -> Self {
        Self {
            reader,
            length,
            hashed: 0,
            digest: Sha256::new(),
        }
    }

    fn advance_to(&mut self, end: u64) -> io::Result<()> {
        if end > self.hashed {
            self.reader.seek(SeekFrom::Start(self.hashed))?;
            let mut scratch = [0; 8192];
            while self.hashed < end {
                let length = (end - self.hashed).min(scratch.len() as u64) as usize;
                self.reader.read_exact(&mut scratch[..length])?;
                self.digest.update(&scratch[..length]);
                self.hashed += length as u64;
            }
        }
        Ok(())
    }

    pub(super) fn read_range(&mut self, start: u64, bytes: &mut [u8]) -> io::Result<()> {
        let end = start
            .checked_add(bytes.len() as u64)
            .filter(|&end| end <= self.length)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "tensor range exceeds checkpoint",
                )
            })?;
        self.advance_to(start)?;
        self.reader.seek(SeekFrom::Start(start))?;
        self.reader.read_exact(bytes)?;
        if end > self.hashed {
            self.digest.update(&bytes[(self.hashed - start) as usize..]);
            self.hashed = end;
        }
        Ok(())
    }

    pub(super) fn finish(&mut self) -> io::Result<String> {
        self.advance_to(self.length)?;
        Ok(format!("{:x}", self.digest.clone().finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    struct Counted {
        bytes: Cursor<Vec<u8>>,
        read: usize,
    }
    impl Read for Counted {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            let count = self.bytes.read(output)?;
            self.read += count;
            Ok(count)
        }
    }
    impl Seek for Counted {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.bytes.seek(position)
        }
    }

    #[test]
    fn ordered_payloads_hash_header_gaps_and_trailer_in_one_pass() {
        let bytes: Vec<u8> = (0..30_003).map(|i| (i % 251) as u8).collect();
        let expected = format!("{:x}", Sha256::digest(&bytes));
        let mut reader = IdentifiedFile::new(
            Counted {
                bytes: Cursor::new(bytes.clone()),
                read: 0,
            },
            bytes.len() as u64,
        );
        for range in [17..999, 1024..12_000, 12_032..30_001] {
            let mut payload = vec![0; range.len()];
            reader.read_range(range.start as u64, &mut payload).unwrap();
            assert_eq!(payload, bytes[range]);
        }
        assert_eq!(reader.finish().unwrap(), expected);
        assert_eq!(reader.reader.read, bytes.len());
        assert_eq!(reader.finish().unwrap(), expected);
        assert_eq!(reader.reader.read, bytes.len());
    }

    #[test]
    fn repeated_and_overlapping_ranges_preserve_whole_file_identity() {
        let bytes: Vec<u8> = (0..128).collect();
        let expected = format!("{:x}", Sha256::digest(&bytes));
        let mut reader = IdentifiedFile::new(Cursor::new(bytes.clone()), bytes.len() as u64);
        for range in [32..64, 16..48, 32..64, 48..96, 0..0, 128..128] {
            let mut payload = vec![0; range.len()];
            reader.read_range(range.start as u64, &mut payload).unwrap();
            assert_eq!(payload, bytes[range]);
        }
        assert_eq!(reader.finish().unwrap(), expected);
    }

    #[test]
    fn rejects_out_of_bounds_overflow_and_truncation() {
        let mut reader = IdentifiedFile::new(Cursor::new(vec![1; 16]), 16);
        assert!(reader.read_range(15, &mut [0; 2]).is_err());
        assert!(reader.read_range(u64::MAX, &mut [0; 2]).is_err());
        let mut reader = IdentifiedFile::new(Cursor::new(vec![1; 15]), 16);
        assert_eq!(
            reader.finish().unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }
}
