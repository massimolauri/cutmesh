pub mod slicer;

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// Official GGML quantization types (`ggml_type` in `llama.cpp`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            19 => Self::IQ1S,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MXFP4,
            40 => Self::NVFP4,
            41 => Self::Q1_0,
            42 => Self::Q2_0,
            other => Self::Unknown(other),
        }
    }

    pub fn block_layout(&self) -> (u64, u64) {
        match self {
            Self::F32 | Self::I32 => (1, 4),
            Self::F16 | Self::BF16 | Self::I16 => (1, 2),
            Self::I8 => (1, 1),
            Self::I64 | Self::F64 => (1, 8),
            Self::Q1_0 => (128, 18),
            Self::Q2_0 => (64, 18),
            Self::Q4_0 | Self::IQ4NL => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::MXFP4 => (32, 17),
            Self::NVFP4 => (64, 36),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2K => (256, 84),
            Self::Q3K | Self::IQ3S => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::IQ2XXS => (256, 66),
            Self::IQ2XS => (256, 74),
            Self::IQ3XXS => (256, 98),
            Self::IQ1S => (256, 50),
            Self::IQ1M => (256, 56),
            Self::IQ2S => (256, 82),
            Self::IQ4XS => (256, 136),
            Self::TQ1_0 => (256, 54),
            Self::TQ2_0 => (256, 66),
            Self::Unknown(_) => (1, 2),
        }
    }

    pub fn compute_byte_size(&self, dimensions: &[u64]) -> u64 {
        let total_elements: u64 = dimensions.iter().copied().product();
        let (blk_elems, blk_bytes) = self.block_layout();
        total_elements.div_ceil(blk_elems) * blk_bytes
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensorDescriptor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    pub relative_offset: u64,
    pub absolute_file_offset: u64,
    pub byte_size: u64,
    pub layer_index: Option<u16>,
    /// For multi-part split GGUF files (`*-00001-of-0000N.gguf`), records the exact shard URL or path
    /// (`*-0000k-of-0000N.gguf`) where this tensor's bytes reside at `absolute_file_offset`.
    pub shard_source: Option<String>,
}

/// Expands a primary GGUF URL or file path ending in `-00001-of-0000N.gguf` into the ordered list
/// of all `N` shard URLs/paths (`-00001-of-0000N.gguf` .. `-0000N-of-0000N.gguf`).
/// For single-file GGUFs, returns `vec![primary_source.to_string()]`.
pub fn expand_split_gguf_sources(primary_source: &str) -> Vec<String> {
    if let Some(idx) = primary_source.rfind("-00001-of-") {
        let prefix = &primary_source[..idx];
        let after = &primary_source[idx + "-00001-of-".len()..];
        if let Some(dot) = after.find(".gguf") {
            let num_str = &after[..dot];
            let suffix = &after[dot..];
            if num_str.len() == 5 {
                if let Ok(total_shards) = num_str.parse::<u32>() {
                    if total_shards > 1 && total_shards <= 128 {
                        return (1..=total_shards)
                            .map(|s_idx| format!("{}-{:05}-of-{:05}{}", prefix, s_idx, total_shards, suffix))
                            .collect();
                    }
                }
            }
        }
    }
    vec![primary_source.to_string()]
}

#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub architecture: String,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub data_section_offset: u64,
    pub total_layers: u16,
    pub embedding_length: u64,
    pub head_count: u64,
    pub head_count_kv: u64,
    pub key_length: u64,
    pub vocab_size: u64,
    pub expert_count: u64,
    pub expert_used_count: u64,
    pub tensors: Vec<GgufTensorDescriptor>,
}

impl GgufHeader {
    pub fn parse_file(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Cannot open GGUF {:?}: {}", path, e))?;
        let mut reader = BufReader::new(file);
        let mut hdr = Self::parse_reader(&mut reader)?;

        // Automatically aggregate tensor descriptors across multi-part split GGUF shards (`*-00001-of-0000N.gguf`)
        let primary_path_str = path.to_string_lossy().to_string();
        for t in &mut hdr.tensors {
            t.shard_source = Some(primary_path_str.clone());
        }
        if let Some(fname) = path.file_name().and_then(|s| s.to_str()) {
            if let Some(idx) = fname.find("-00001-of-") {
                let prefix = &fname[..idx];
                let after = &fname[idx + "-00001-of-".len()..];
                if let Some(dot) = after.find(".gguf") {
                    if let Ok(total_shards) = after[..dot].parse::<u32>() {
                        if let Some(parent) = path.parent() {
                            let mut max_layer: Option<u16> = if hdr.total_layers > 0 {
                                Some(hdr.total_layers - 1)
                            } else {
                                None
                            };
                            for s_idx in 2..=total_shards {
                                let sib_name = format!("{}-{:05}-of-{:05}.gguf", prefix, s_idx, total_shards);
                                let sib_path = parent.join(sib_name);
                                if let Ok(sib_file) = File::open(&sib_path) {
                                    let mut sib_reader = BufReader::new(sib_file);
                                    if let Ok(mut sib_hdr) = Self::parse_reader(&mut sib_reader) {
                                        let sib_str = sib_path.to_string_lossy().to_string();
                                        for t in &mut sib_hdr.tensors {
                                            t.shard_source = Some(sib_str.clone());
                                        }
                                        if sib_hdr.total_layers > 0 {
                                            let l = sib_hdr.total_layers - 1;
                                            max_layer = Some(max_layer.map_or(l, |m| m.max(l)));
                                        }
                                        if hdr.embedding_length == 0 && sib_hdr.embedding_length > 0 {
                                            hdr.embedding_length = sib_hdr.embedding_length;
                                        }
                                        if hdr.vocab_size == 0 && sib_hdr.vocab_size > 0 {
                                            hdr.vocab_size = sib_hdr.vocab_size;
                                        }
                                        hdr.tensor_count += sib_hdr.tensor_count;
                                        hdr.tensors.extend(sib_hdr.tensors);
                                    }
                                }
                            }
                            if let Some(m) = max_layer {
                                hdr.total_layers = m + 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(hdr)
    }

    pub fn parse_reader<R: Read + Seek>(r: &mut R) -> Result<Self, String> {
        let magic = read_u32(r)?;
        if magic != GGUF_MAGIC {
            return Err(format!("Invalid GGUF magic: 0x{:08X}", magic));
        }
        let version = read_u32(r)?;
        if !(2..=3).contains(&version) {
            return Err(format!("Unsupported GGUF version: {}", version));
        }
        let tensor_count = read_u64(r)?;
        let kv_count = read_u64(r)?;

        let mut architecture = String::from("unknown");
        let mut alignment = GGUF_DEFAULT_ALIGNMENT;
        let mut embedding_length = 0u64;
        let mut head_count = 0u64;
        let mut head_count_kv = 0u64;
        let mut key_length = 0u64;
        let mut expert_count = 0u64;
        let mut expert_used_count = 0u64;
        let mut tokenizer_vocab_count = 0u64;

        for _ in 0..kv_count {
            let key = read_string(r)?;
            let val_type = read_u32(r)?;
            if key == "general.architecture" && val_type == 8 {
                architecture = read_string(r)?;
            } else if key == "general.alignment" && is_int_or_array_type(val_type) {
                alignment = read_uint_or_array_max(r, val_type)?.max(1);
            } else if key.ends_with(".embedding_length") && is_int_or_array_type(val_type) {
                embedding_length = read_uint_or_array_max(r, val_type)?;
            } else if key.ends_with(".attention.head_count") && is_int_or_array_type(val_type) {
                head_count = read_uint_or_array_max(r, val_type)?;
            } else if key.ends_with(".attention.head_count_kv") && is_int_or_array_type(val_type) {
                head_count_kv = read_uint_or_array_max(r, val_type)?;
            } else if key.ends_with(".attention.key_length") && is_int_or_array_type(val_type) {
                key_length = read_uint_or_array_max(r, val_type)?;
            } else if key.ends_with(".expert_count") && is_int_or_array_type(val_type) {
                expert_count = read_uint_or_array_max(r, val_type)?;
            } else if key.ends_with(".expert_used_count") && is_int_or_array_type(val_type) {
                expert_used_count = read_uint_or_array_max(r, val_type)?;
            } else if key.ends_with(".vocab_size") && is_int_or_array_type(val_type) {
                let v = read_uint_or_array_max(r, val_type)?;
                if v > 0 {
                    tokenizer_vocab_count = v;
                }
            } else if key == "tokenizer.ggml.tokens" && val_type == 9 {
                let elem_type = read_u32(r)?;
                let count = read_u64(r)?;
                tokenizer_vocab_count = count;
                for _ in 0..count {
                    skip_value(r, elem_type)?;
                }
            } else {
                skip_value(r, val_type)?;
            }
        }

        const MAX_GGUF_TENSORS: u64 = 1_000_000;
        const MAX_GGUF_DIMS: u32 = 16;
        if tensor_count > MAX_GGUF_TENSORS {
            return Err(format!(
                "GGUF tensor count {} exceeds safety limit ({})",
                tensor_count, MAX_GGUF_TENSORS
            ));
        }
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut max_layer: Option<u16> = None;

        for _ in 0..tensor_count {
            let name = read_string(r)?;
            let raw_dims = read_u32(r)?;
            if raw_dims > MAX_GGUF_DIMS {
                return Err(format!(
                    "Tensor '{}' n_dims {} exceeds safety limit ({})",
                    name, raw_dims, MAX_GGUF_DIMS
                ));
            }
            let n_dims = raw_dims as usize;
            let mut dimensions = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dimensions.push(read_u64(r)?);
            }
            let ggml_type = GgmlType::from_u32(read_u32(r)?);
            let relative_offset = read_u64(r)?;
            let byte_size = ggml_type.compute_byte_size(&dimensions);
            let layer_index = parse_blk_index(&name);
            if let Some(l) = layer_index {
                max_layer = Some(max_layer.map_or(l, |m| m.max(l)));
            }

            tensors.push(GgufTensorDescriptor {
                name,
                dimensions,
                ggml_type,
                relative_offset,
                absolute_file_offset: 0,
                byte_size,
                layer_index,
                shard_source: None,
            });
        }

        let pos = r.stream_position().map_err(|e| e.to_string())?;
        let pad = (alignment - (pos % alignment)) % alignment;
        let data_section_offset = pos + pad;

        for t in &mut tensors {
            t.absolute_file_offset = data_section_offset + t.relative_offset;
        }

        // Derive exact embedding_length and vocab_size from token_embd.weight dimensions [d, vocab] or tokenizer metadata
        let mut vocab_size = tokenizer_vocab_count;
        if let Some(embd) = tensors
            .iter()
            .find(|t| t.name == "token_embd.weight" || t.name == "wte.weight")
        {
            if embd.dimensions.len() >= 2 {
                if embedding_length == 0 {
                    embedding_length = embd.dimensions[0];
                }
                vocab_size = embd.dimensions[1];
            }
        }
        if head_count > 0 && key_length == 0 && embedding_length > 0 {
            key_length = embedding_length / head_count;
        }
        if head_count_kv == 0 {
            head_count_kv = head_count;
        }

        Ok(Self {
            version,
            architecture,
            tensor_count,
            kv_count,
            data_section_offset,
            total_layers: max_layer.map_or(0, |m| m + 1),
            embedding_length,
            head_count,
            head_count_kv,
            key_length,
            vocab_size,
            expert_count,
            expert_used_count,
            tensors,
        })
    }
}

fn is_int_or_array_type(val_type: u32) -> bool {
    matches!(val_type, 4 | 5 | 9 | 10 | 11)
}

fn read_uint_or_array_max<R: Read + Seek>(r: &mut R, val_type: u32) -> Result<u64, String> {
    match val_type {
        4 => Ok(read_u32(r)? as u64),
        5 => Ok((read_u32(r)? as i32).max(0) as u64),
        10 | 11 => read_u64(r),
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            let mut max_val = 0u64;
            for _ in 0..count {
                if matches!(elem_type, 4 | 5 | 10 | 11) {
                    let v = read_uint_or_array_max(r, elem_type)?;
                    if v > max_val && v < (1u64 << 60) {
                        max_val = v;
                    }
                } else {
                    skip_value(r, elem_type)?;
                }
            }
            Ok(max_val)
        }
        other => {
            skip_value(r, other)?;
            Ok(0)
        }
    }
}

fn parse_blk_index(name: &str) -> Option<u16> {
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    rest[..dot].parse::<u16>().ok()
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> Result<String, String> {
    let len = read_u64(r)? as usize;
    if len > 10_000_000 {
        return Err("GGUF string exceeds 10MB".into());
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    String::from_utf8(buf).map_err(|e| e.to_string())
}

fn skip_value<R: Read + Seek>(r: &mut R, val_type: u32) -> Result<(), String> {
    match val_type {
        0 | 1 | 7 => {
            r.seek(SeekFrom::Current(1)).map_err(|e| e.to_string())?;
        }
        2 | 3 => {
            r.seek(SeekFrom::Current(2)).map_err(|e| e.to_string())?;
        }
        4..=6 => {
            r.seek(SeekFrom::Current(4)).map_err(|e| e.to_string())?;
        }
        8 => {
            let len = read_u64(r)?;
            r.seek(SeekFrom::Current(len as i64))
                .map_err(|e| e.to_string())?;
        }
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_value(r, elem_type)?;
            }
        }
        10..=12 => {
            r.seek(SeekFrom::Current(8)).map_err(|e| e.to_string())?;
        }
        other => return Err(format!("Unknown GGUF metadata type {}", other)),
    }
    Ok(())
}
