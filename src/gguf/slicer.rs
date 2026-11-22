use std::collections::BTreeMap;
use super::GgufHeader;

/// Exact byte range inside a `.gguf` file for a single transformer block `blk.i`.
#[derive(Debug, Clone)]
pub struct LayerByteRange {
    pub layer_index: u16,
    pub tensor_count: usize,
    pub start_byte: u64,
    pub end_byte: u64,
    pub total_bytes: u64,
}

/// Assignment plan for a single Worker Node based on its available VRAM budget.
#[derive(Debug, Clone)]
pub struct ShardSlicePlan {
    pub start_layer: u16,
    pub end_layer: u16,
    pub weights_bytes: u64,
    pub file_byte_range: (u64, u64),
}

/// Groups a parsed `GgufHeader` into per-layer byte ranges (`blk.0` .. `blk.(N-1)`).
pub fn group_layers(header: &GgufHeader) -> BTreeMap<u16, LayerByteRange> {
    let mut map: BTreeMap<u16, LayerByteRange> = BTreeMap::new();
    for t in &header.tensors {
        if let Some(l_idx) = t.layer_index {
            let end = t.absolute_file_offset + t.byte_size;
            let entry = map.entry(l_idx).or_insert(LayerByteRange {
                layer_index: l_idx,
                tensor_count: 0,
                start_byte: t.absolute_file_offset,
                end_byte: end,
                total_bytes: 0,
            });
            entry.tensor_count += 1;
            entry.start_byte = entry.start_byte.min(t.absolute_file_offset);
            entry.end_byte = entry.end_byte.max(end);
            entry.total_bytes += t.byte_size;
        }
    }
    map
}

/// Model-Agnostic VRAM Slicer:
/// Calculates how many contiguous layers starting at `start_layer` fit into `vram_budget_mb`
/// (after reserving `kv_reserve_mb` for the KV-cache).
pub fn plan_shard_for_vram_mb(
    header: &GgufHeader,
    start_layer: u16,
    vram_budget_mb: u64,
    kv_reserve_mb: u64,
) -> Option<ShardSlicePlan> {
    let layers = group_layers(header);
    let usable_bytes = vram_budget_mb
        .saturating_sub(kv_reserve_mb)
        .saturating_mul(1024 * 1024);

    let mut acc_bytes = 0u64;
    let mut end_layer = start_layer;
    let mut start_offset = u64::MAX;
    let mut end_offset = 0u64;

    for l in start_layer..header.total_layers {
        let slice = layers.get(&l)?;
        if acc_bytes + slice.total_bytes > usable_bytes && end_layer > start_layer {
            break;
        }
        acc_bytes += slice.total_bytes;
        start_offset = start_offset.min(slice.start_byte);
        end_offset = end_offset.max(slice.end_byte);
        end_layer = l + 1;
    }

    if end_layer > start_layer {
        Some(ShardSlicePlan {
            start_layer,
            end_layer,
            weights_bytes: acc_bytes,
            file_byte_range: (start_offset, end_offset),
        })
    } else {
        None
    }
}

/// Minimum tensor byte size queried via `RPC_CMD_SET_TENSOR_HASH` (64 KiB).
/// Lowered from upstream 10 MiB / 1 MiB so attention Q/K/V/O projections in compact models
/// (e.g. 0.5B - 3B) also hit the local `ggml-rpc-server -c` tensor cache instead of uploading from the Gateway.
pub const RPC_CACHE_HASH_THRESHOLD: u64 = 65_536;

/// Resolves the exact directory used by `ggml-rpc-server -c` (`get_cache_directory()` in `llama.cpp`).
pub fn default_ggml_rpc_cache_dir() -> std::path::PathBuf {
    if let Ok(custom) = std::env::var("LLAMA_CACHE") {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            return std::path::PathBuf::from(trimmed).join("rpc");
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            return std::path::PathBuf::from(local_app_data)
                .join("llama.cpp")
                .join("rpc");
        }
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        let trimmed = xdg.trim();
        if !trimmed.is_empty() {
            return std::path::PathBuf::from(trimmed)
                .join("llama.cpp")
                .join("rpc");
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home)
                .join("Library")
                .join("Caches")
                .join("llama.cpp")
                .join("rpc");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return std::path::PathBuf::from(home)
            .join(".cache")
            .join("llama.cpp")
            .join("rpc");
    }
    std::path::PathBuf::from("/root/.cache/llama.cpp/rpc")
}

/// Computes the exact 64-bit FNV-1a hash used by `llama.cpp` `ggml-rpc` (`RPC_CMD_SET_TENSOR_HASH`).
pub fn ggml_rpc_fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Deterministic 64-byte header tag written at `t.absolute_file_offset` inside a CPU-Master Sparse GGUF file
/// when the remaining `byte_size - 64` bytes are left as an unallocated sparse hole (`0x00`).
pub fn ggml_rpc_sparse_tensor_tag(tensor_name: &str, byte_size: u64) -> [u8; 64] {
    let mut tag = [0u8; 64];
    let prefix = b"CUTMESH_SPARSE_V1:";
    tag[..prefix.len()].copy_from_slice(prefix);
    let name_bytes = tensor_name.as_bytes();
    let copy_len = name_bytes.len().min(64 - prefix.len() - 8);
    tag[prefix.len()..prefix.len() + copy_len].copy_from_slice(&name_bytes[..copy_len]);
    tag[56..64].copy_from_slice(&byte_size.to_le_bytes());
    tag
}

/// Fast $O(\log N)$ binary exponentiation of the 64-bit FNV-1a prime `0x100000001b3` modulo $2^{64}$.
fn fnv1a_prime_pow(mut exp: u64) -> u64 {
    let mut base: u64 = 0x100000001b3;
    let mut result: u64 = 1;
    while exp > 0 {
        if (exp & 1) == 1 {
            result = result.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        exp >>= 1;
    }
    result
}

/// Computes in $O(\log(\text{byte\_size}))$ (~30 ns, 0 allocations) the exact 64-bit FNV-1a hash that
/// `llama-server` computes when reading a sparse tensor of length `byte_size` whose first 64 bytes are
/// [`ggml_rpc_sparse_tensor_tag`] and whose remaining `byte_size - 64` bytes are sparse zeroes (`0x00`).
pub fn ggml_rpc_sparse_tensor_hash(tensor_name: &str, byte_size: u64) -> u64 {
    let tag = ggml_rpc_sparse_tensor_tag(tensor_name, byte_size);
    if byte_size <= 64 {
        return ggml_rpc_fnv1a_hash(&tag[..(byte_size as usize)]);
    }
    let h64 = ggml_rpc_fnv1a_hash(&tag);
    h64.wrapping_mul(fnv1a_prime_pow(byte_size - 64))
}

/// Creates a zero-disk-overhead hardlink `/data/cutmesh/cache/rpc/<sparse_hash>` -> `<real_hash>`
/// so `ggml-rpc-server -c` natively resolves both full-file hashes and CPU-Master Sparse GGUF hashes.
pub fn link_sparse_rpc_cache_entry(
    cache_dir: &std::path::Path,
    real_path: &std::path::Path,
    tensor_name: &str,
    byte_size: u64,
) {
    let sparse_hash = ggml_rpc_sparse_tensor_hash(tensor_name, byte_size);
    let sparse_path = cache_dir.join(format!("{:016x}", sparse_hash));
    if sparse_path == real_path {
        return;
    }
    let ok = std::fs::metadata(&sparse_path)
        .map(|m| m.len() == byte_size)
        .unwrap_or(false);
    if !ok {
        let _ = std::fs::remove_file(&sparse_path);
        let _ = std::fs::hard_link(real_path, &sparse_path);
    }
}

/// Populates the local `ggml-rpc-server -c` tensor cache directory (`~/.cache/llama.cpp/rpc`)
/// directly from a local `.gguf` file or shard so `RPC_CMD_SET_TENSOR_HASH` achieves 100% cache hit
/// without streaming multi-gigabyte weights over WAN relays.
pub fn populate_ggml_rpc_cache(
    gguf_path: &std::path::Path,
    cache_dir: &std::path::Path,
    start_layer: Option<u16>,
    end_layer: Option<u16>,
) -> Result<(usize, u64), String> {
    use std::io::{Read, Seek, SeekFrom};

    let hdr_file = std::fs::File::open(gguf_path).map_err(|e| e.to_string())?;
    let mut hdr_reader = std::io::BufReader::new(hdr_file);
    let hdr = GgufHeader::parse_reader(&mut hdr_reader)?;
    std::fs::create_dir_all(cache_dir).map_err(|e| e.to_string())?;

    let mut file = std::fs::File::open(gguf_path).map_err(|e| e.to_string())?;
    let mut cached_count = 0usize;
    let mut cached_bytes = 0u64;
    let mut buf = Vec::new();

    for t in &hdr.tensors {
        if t.byte_size <= RPC_CACHE_HASH_THRESHOLD {
            continue;
        }
        if let (Some(s), Some(e)) = (start_layer, end_layer) {
            match t.layer_index {
                Some(l_idx) => {
                    if l_idx < s || l_idx >= e {
                        continue;
                    }
                }
                None => {
                    if s > 0 && e < 48 {
                        continue;
                    }
                }
            }
        }

        buf.resize(t.byte_size as usize, 0u8);
        file.seek(SeekFrom::Start(t.absolute_file_offset))
            .map_err(|e| e.to_string())?;
        file.read_exact(&mut buf).map_err(|e| e.to_string())?;

        let hash = ggml_rpc_fnv1a_hash(&buf);
        let hash_hex = format!("{:016x}", hash);
        let out_path = cache_dir.join(&hash_hex);

        let already_exists = std::fs::metadata(&out_path)
            .map(|m| m.len() == t.byte_size)
            .unwrap_or(false);
        if !already_exists {
            std::fs::write(&out_path, &buf).map_err(|e| e.to_string())?;
        }
        link_sparse_rpc_cache_entry(cache_dir, &out_path, &t.name, t.byte_size);
        cached_count += 1;
        cached_bytes += t.byte_size;
    }

    Ok((cached_count, cached_bytes))
}
