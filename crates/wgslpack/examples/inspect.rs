//! Host I/O example and prefix parser benchmark; the library itself is Sans-I/O.
use std::{fs::File, hint::black_box, io::Read, time::Instant};
use wgslpack::{Header, Index};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("usage: inspect ARCHIVE.wgslpack")?;
    let mut file = File::open(path)?;
    let mut fixed = [0u8; wgslpack::HEADER_SIZE];
    file.read_exact(&mut fixed)?;
    let header = Header::parse(&fixed)?;
    // Application-owned I/O policy, separate from structural validation.
    if header.prefix_len() > 64 * 1024 * 1024 {
        return Err("index exceeds example limit".into());
    }
    let mut prefix = vec![0u8; header.prefix_len()];
    prefix[..fixed.len()].copy_from_slice(&fixed);
    file.read_exact(&mut prefix[fixed.len()..])?;
    let file_len = file.metadata()?.len();
    let index = Index::parse(&prefix, file_len)?;
    let start = Instant::now();
    for _ in 0..10_000 {
        black_box(Index::parse(black_box(&prefix), file_len)?);
    }
    println!(
        "tensors={} prefix_bytes={} file_bytes={} parse_us={:.3}",
        index.header().tensor_count(),
        prefix.len(),
        file_len,
        start.elapsed().as_secs_f64() * 1_000_000.0 / 10_000.0
    );
    Ok(())
}
