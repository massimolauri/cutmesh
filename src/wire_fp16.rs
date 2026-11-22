use std::collections::{HashMap, VecDeque};

/// Wire magic for Cutmesh v2 multiplexed `ggml-rpc` stream with lossless-bounded FP16 activation compression.
pub const RPC_STREAM_MAGIC_V2: &[u8; 8] = b"SYNRPC02";

pub const WIRE_TAG_RAW: u8 = 0;
pub const WIRE_TAG_SET_TENSOR_F16: u8 = 1;
pub const WIRE_TAG_GET_TENSOR_RSP_F16: u8 = 2;

const GGML_TYPE_F32: u32 = 0;
const MIN_COMPRESS_BYTES: usize = 4096;
const DEFAULT_RPC_TENSOR_SIZE: usize = 296;
const MAX_RAW_WIRE_CHUNK: usize = 4 * 1024 * 1024;
const MAX_REASONABLE_FRAME_BYTES: u64 = 512 * 1024 * 1024;

// ggml-rpc command IDs (llama.cpp 785322d)
const RPC_CMD_ALLOC_BUFFER: u8 = 0;
const RPC_CMD_GET_ALIGNMENT: u8 = 1;
const RPC_CMD_GET_MAX_SIZE: u8 = 2;
const RPC_CMD_BUFFER_GET_BASE: u8 = 3;
const RPC_CMD_SET_TENSOR: u8 = 6;
const RPC_CMD_SET_TENSOR_HASH: u8 = 7;
const RPC_CMD_GET_TENSOR: u8 = 8;
const RPC_CMD_COPY_TENSOR: u8 = 9;
const RPC_CMD_GET_DEVICE_MEMORY: u8 = 11;
const RPC_CMD_INIT_TENSOR: u8 = 12;
const RPC_CMD_GET_ALLOC_SIZE: u8 = 13;
const RPC_CMD_HELLO: u8 = 14;
const RPC_CMD_DEVICE_COUNT: u8 = 15;
const RPC_CMD_GET_TENSOR_2D: u8 = 19;
const RPC_CMD_COMM_INIT: u8 = 20;

/// Returns whether a `ggml-rpc` client command produces a serialized server response.
pub fn cmd_has_response(cmd: u8) -> bool {
    matches!(
        cmd,
        RPC_CMD_ALLOC_BUFFER
            | RPC_CMD_GET_ALIGNMENT
            | RPC_CMD_GET_MAX_SIZE
            | RPC_CMD_BUFFER_GET_BASE
            | RPC_CMD_SET_TENSOR_HASH
            | RPC_CMD_GET_TENSOR
            | RPC_CMD_COPY_TENSOR
            | RPC_CMD_GET_DEVICE_MEMORY
            | RPC_CMD_GET_ALLOC_SIZE
            | RPC_CMD_HELLO
            | RPC_CMD_DEVICE_COUNT
            | RPC_CMD_GET_TENSOR_2D
            | RPC_CMD_COMM_INIT
    )
}

/// Runtime toggle: `CUTMESH_WIRE_FP16=0` disables wire FP16 compression and falls back to `SYNRPC01`.
pub fn wire_fp16_enabled() -> bool {
    std::env::var("CUTMESH_WIRE_FP16").as_deref() != Ok("0")
}

/// Converts IEEE 754 `f32` to IEEE 754 `binary16` bit pattern with round-to-nearest-even.
#[inline]
pub fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        if mant == 0 {
            return sign | 0x7c00; // Inf
        }
        let nan_payload = ((mant >> 13) as u16).max(1);
        return sign | 0x7c00 | nan_payload; // NaN
    }

    let unbiased = exp - 127 + 15;
    if unbiased >= 31 {
        return sign | 0x7c00; // Overflow -> Inf
    }
    if unbiased <= 0 {
        if unbiased < -10 {
            return sign; // Underflow to signed zero
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - unbiased) as u32;
        let half_m = (m >> shift) as u16;
        let rem_mask = (1u32 << shift) - 1;
        let halfway = 1u32 << (shift - 1);
        let rem = m & rem_mask;
        let round_up = rem > halfway || (rem == halfway && (half_m & 1) != 0);
        return sign | (half_m + u16::from(round_up));
    }

    let half_exp = (unbiased as u16) << 10;
    let half_mant = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    let round_up = rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) != 0);
    sign | ((half_exp | half_mant) + u16::from(round_up))
}

/// Converts IEEE 754 `binary16` bit pattern back to exact `f32`.
#[inline]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127u32 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        if mant == 0 {
            sign | 0x7f80_0000
        } else {
            sign | 0x7fc0_0000 | (mant << 13)
        }
    } else {
        let e = exp + 127 - 15;
        sign | (e << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Encodes contiguous little-endian `f32` bytes into `binary16` (half size).
/// Returns `None` if smaller than `MIN_COMPRESS_BYTES`, unaligned, or if any finite value overflows `|x| > 65000.0`.
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn encode_f32_bytes_to_f16(src: &[u8]) -> Option<Vec<u8>> {
    if src.len() < MIN_COMPRESS_BYTES || (src.len() & 3) != 0 {
        return None;
    }
    let count = src.len() / 4;
    let mut out = Vec::with_capacity(count * 2);
    for chunk in src.chunks_exact(4) {
        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if f.is_finite() && f.abs() > 65_000.0 {
            return None;
        }
        out.extend_from_slice(&f32_to_f16_bits(f).to_le_bytes());
    }
    Some(out)
}

/// Decodes contiguous little-endian `binary16` bytes back into `f32` bytes (2x size).
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn decode_f16_bytes_to_f32(src: &[u8]) -> Option<Vec<u8>> {
    if (src.len() & 1) != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(src.len() * 2);
    for chunk in src.chunks_exact(2) {
        let h = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.extend_from_slice(&f16_bits_to_f32(h).to_le_bytes());
    }
    Some(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedRsp {
    Raw,
    GetTensorF32(u64),
}

#[derive(Debug, Default, Clone)]
pub struct WireCompressionEvent {
    pub raw_bytes: usize,
    pub compressed_bytes: usize,
}

fn push_wire_frame(out: &mut Vec<u8>, tag: u8, payload: &[u8]) {
    out.push(tag);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
}

fn push_raw_chunked(out: &mut Vec<u8>, bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    for chunk in bytes.chunks(MAX_RAW_WIRE_CHUNK) {
        push_wire_frame(out, WIRE_TAG_RAW, chunk);
    }
}

fn read_u64_le(bytes: &[u8], off: usize) -> Option<u64> {
    let s = bytes.get(off..off + 8)?;
    Some(u64::from_le_bytes(s.try_into().ok()?))
}

fn read_u32_le(bytes: &[u8], off: usize) -> Option<u32> {
    let s = bytes.get(off..off + 4)?;
    Some(u32::from_le_bytes(s.try_into().ok()?))
}

fn calibrate_tensor_size(cmd: u8, req_size: usize, current: usize) -> usize {
    match cmd {
        RPC_CMD_INIT_TENSOR if (128..=512).contains(&req_size) && (req_size & 7) == 0 => req_size,
        RPC_CMD_SET_TENSOR_HASH | RPC_CMD_GET_TENSOR if req_size > 16 => {
            let cand = req_size - 16;
            if (128..=512).contains(&cand) && (cand & 7) == 0 {
                cand
            } else {
                current
            }
        }
        _ => current,
    }
}

/// Frames client-side `ggml-rpc` TCP requests into `SYNRPC02` wire frames:
/// compresses eligible `SET_TENSOR` `F32` activation payloads to `binary16` and translates
/// sparse tensor hashes in place for `SET_TENSOR_HASH`.
pub struct ClientGgmlFramer {
    buf: Vec<u8>,
    rpc_tensor_size: usize,
    fp16_enabled: bool,
    pub logged_first_compression: bool,
}

impl ClientGgmlFramer {
    pub fn new(fp16_enabled: bool) -> Self {
        Self {
            buf: Vec::with_capacity(128 * 1024),
            rpc_tensor_size: DEFAULT_RPC_TENSOR_SIZE,
            fp16_enabled,
            logged_first_compression: false,
        }
    }

    pub fn push_tcp_bytes(
        &mut self,
        data: &[u8],
        hash_map: Option<&HashMap<u64, u64>>,
    ) -> (Vec<u8>, Option<WireCompressionEvent>) {
        self.buf.extend_from_slice(data);
        let mut out = Vec::with_capacity(data.len() + 16);
        let mut first_event = None;
        let mut consumed = 0usize;

        while consumed + 9 <= self.buf.len() {
            let cmd = self.buf[consumed];
            let Some(req_size_u64) = read_u64_le(&self.buf, consumed + 1) else {
                break;
            };
            if req_size_u64 > MAX_REASONABLE_FRAME_BYTES {
                // Unrecognized raw stream fallback
                push_raw_chunked(&mut out, &self.buf[consumed..]);
                consumed = self.buf.len();
                break;
            }
            let req_size = req_size_u64 as usize;
            let Some(frame_end) = (consumed + 9).checked_add(req_size) else {
                break;
            };
            if frame_end > self.buf.len() {
                break;
            }

            self.rpc_tensor_size = calibrate_tensor_size(cmd, req_size, self.rpc_tensor_size);

            if cmd == RPC_CMD_SET_TENSOR_HASH && frame_end >= consumed + 17 {
                if let Some(map) = hash_map {
                    if let Some(candidate) = read_u64_le(&self.buf, frame_end - 8) {
                        if let Some(&real_hash) = map.get(&candidate) {
                            self.buf[frame_end - 8..frame_end].copy_from_slice(&real_hash.to_le_bytes());
                        }
                    }
                }
                push_raw_chunked(&mut out, &self.buf[consumed..frame_end]);
            } else if self.fp16_enabled
                && cmd == RPC_CMD_SET_TENSOR
                && req_size > self.rpc_tensor_size + 9
            {
                let payload = &self.buf[consumed + 9..frame_end];
                let tensor_type = read_u32_le(payload, 8);
                let cache_flag = payload.get(self.rpc_tensor_size).copied();
                let data_slice = &payload[self.rpc_tensor_size + 9..];
                if tensor_type == Some(GGML_TYPE_F32) && cache_flag == Some(0) {
                    if let Some(f16_bytes) = encode_f32_bytes_to_f16(data_slice) {
                        let hdr_len = 9 + self.rpc_tensor_size + 9;
                        let wire_payload_len = 2 + hdr_len + f16_bytes.len();
                        out.push(WIRE_TAG_SET_TENSOR_F16);
                        out.extend_from_slice(&(wire_payload_len as u32).to_le_bytes());
                        out.extend_from_slice(&(hdr_len as u16).to_le_bytes());
                        out.extend_from_slice(&self.buf[consumed..consumed + hdr_len]);
                        out.extend_from_slice(&f16_bytes);
                        if !self.logged_first_compression {
                            self.logged_first_compression = true;
                            first_event = Some(WireCompressionEvent {
                                raw_bytes: data_slice.len(),
                                compressed_bytes: f16_bytes.len(),
                            });
                        }
                        consumed = frame_end;
                        continue;
                    }
                }
                push_raw_chunked(&mut out, &self.buf[consumed..frame_end]);
            } else {
                push_raw_chunked(&mut out, &self.buf[consumed..frame_end]);
            }

            consumed = frame_end;
        }

        if consumed > 0 {
            self.buf.drain(..consumed);
        }
        (out, first_event)
    }
}

/// Tracks decoded client `ggml-rpc` requests on the Worker to record expected server response kinds.
pub struct DecodedReqTracker {
    buf: Vec<u8>,
    rpc_tensor_size: usize,
}

impl DecodedReqTracker {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(64 * 1024),
            rpc_tensor_size: DEFAULT_RPC_TENSOR_SIZE,
        }
    }

    pub fn observe_tcp_bytes(&mut self, data: &[u8], expected_queue: &mut VecDeque<ExpectedRsp>) {
        self.buf.extend_from_slice(data);
        let mut consumed = 0usize;

        while consumed + 9 <= self.buf.len() {
            let cmd = self.buf[consumed];
            let Some(req_size_u64) = read_u64_le(&self.buf, consumed + 1) else {
                break;
            };
            if req_size_u64 > MAX_REASONABLE_FRAME_BYTES {
                self.buf.clear();
                return;
            }
            let req_size = req_size_u64 as usize;
            let Some(frame_end) = (consumed + 9).checked_add(req_size) else {
                break;
            };
            if frame_end > self.buf.len() {
                break;
            }

            self.rpc_tensor_size = calibrate_tensor_size(cmd, req_size, self.rpc_tensor_size);

            if cmd_has_response(cmd) {
                let mut kind = ExpectedRsp::Raw;
                if cmd == RPC_CMD_GET_TENSOR && req_size >= self.rpc_tensor_size + 16 {
                    let payload = &self.buf[consumed + 9..frame_end];
                    let tensor_type = read_u32_le(payload, 8);
                    let get_size = read_u64_le(payload, req_size - 8);
                    if tensor_type == Some(GGML_TYPE_F32) {
                        if let Some(sz) = get_size {
                            kind = ExpectedRsp::GetTensorF32(sz);
                        }
                    }
                }
                expected_queue.push_back(kind);
            }

            consumed = frame_end;
        }

        if consumed > 0 {
            self.buf.drain(..consumed);
        }
    }
}

impl Default for DecodedReqTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Frames server-side `ggml-rpc` responses (`[rsp_size: u64 LE][payload]`) into `SYNRPC02` wire frames,
/// compressing `GET_TENSOR` `F32` responses to `binary16`.
pub struct WorkerRspFramer {
    buf: Vec<u8>,
    fp16_enabled: bool,
    pub logged_first_compression: bool,
}

impl WorkerRspFramer {
    pub fn new(fp16_enabled: bool) -> Self {
        Self {
            buf: Vec::with_capacity(128 * 1024),
            fp16_enabled,
            logged_first_compression: false,
        }
    }

    pub fn push_tcp_bytes(
        &mut self,
        data: &[u8],
        expected_queue: &mut VecDeque<ExpectedRsp>,
    ) -> (Vec<u8>, Option<WireCompressionEvent>) {
        self.buf.extend_from_slice(data);
        let mut out = Vec::with_capacity(data.len() + 16);
        let mut first_event = None;
        let mut consumed = 0usize;

        while consumed + 8 <= self.buf.len() {
            let Some(rsp_size_u64) = read_u64_le(&self.buf, consumed) else {
                break;
            };
            if rsp_size_u64 > MAX_REASONABLE_FRAME_BYTES {
                push_raw_chunked(&mut out, &self.buf[consumed..]);
                consumed = self.buf.len();
                break;
            }
            let rsp_size = rsp_size_u64 as usize;
            let Some(frame_end) = (consumed + 8).checked_add(rsp_size) else {
                break;
            };
            if frame_end > self.buf.len() {
                break;
            }

            let expected = expected_queue.pop_front().unwrap_or(ExpectedRsp::Raw);
            let payload = &self.buf[consumed + 8..frame_end];
            if self.fp16_enabled && matches!(expected, ExpectedRsp::GetTensorF32(sz) if sz == rsp_size_u64) {
                if let Some(f16_bytes) = encode_f32_bytes_to_f16(payload) {
                    let wire_payload_len = 8 + f16_bytes.len();
                    out.push(WIRE_TAG_GET_TENSOR_RSP_F16);
                    out.extend_from_slice(&(wire_payload_len as u32).to_le_bytes());
                    out.extend_from_slice(&rsp_size_u64.to_le_bytes());
                    out.extend_from_slice(&f16_bytes);
                    if !self.logged_first_compression {
                        self.logged_first_compression = true;
                        first_event = Some(WireCompressionEvent {
                            raw_bytes: payload.len(),
                            compressed_bytes: f16_bytes.len(),
                        });
                    }
                    consumed = frame_end;
                    continue;
                }
            }

            push_raw_chunked(&mut out, &self.buf[consumed..frame_end]);
            consumed = frame_end;
        }

        if consumed > 0 {
            self.buf.drain(..consumed);
        }
        (out, first_event)
    }
}

/// Decodes incoming `SYNRPC02` wire frames (`[tag: u8][wire_len: u32 LE][payload]`) back into raw `ggml-rpc` TCP bytes.
pub struct WireStreamDecoder {
    buf: Vec<u8>,
}

impl WireStreamDecoder {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(128 * 1024),
        }
    }

    pub fn push_wire_bytes(&mut self, data: &[u8]) -> Result<(Vec<u8>, Option<WireCompressionEvent>), String> {
        self.buf.extend_from_slice(data);
        let mut out = Vec::with_capacity(data.len());
        let mut first_event = None;
        let mut consumed = 0usize;

        while consumed + 5 <= self.buf.len() {
            let tag = self.buf[consumed];
            let Some(wire_len_u32) = read_u32_le(&self.buf, consumed + 1) else {
                break;
            };
            let wire_len = wire_len_u32 as usize;
            let Some(frame_end) = (consumed + 5).checked_add(wire_len) else {
                return Err("SYNRPC02 wire_len overflow".to_string());
            };
            if frame_end > self.buf.len() {
                break;
            }

            let payload = &self.buf[consumed + 5..frame_end];
            match tag {
                WIRE_TAG_RAW => {
                    out.extend_from_slice(payload);
                }
                WIRE_TAG_SET_TENSOR_F16 => {
                    if payload.len() < 2 {
                        return Err("Truncated SET_TENSOR_F16 payload".to_string());
                    }
                    let hdr_len = u16::from_le_bytes([payload[0], payload[1]]) as usize;
                    if hdr_len < 18 || payload.len() < 2 + hdr_len {
                        return Err(format!("Invalid SET_TENSOR_F16 hdr_len={}", hdr_len));
                    }
                    let hdr = &payload[2..2 + hdr_len];
                    let f16_data = &payload[2 + hdr_len..];
                    let f32_data = decode_f16_bytes_to_f32(f16_data)
                        .ok_or_else(|| "Odd byte count in SET_TENSOR_F16".to_string())?;
                    let expected_req_size = read_u64_le(hdr, 1).unwrap_or(0) as usize;
                    let prefix_without_9 = hdr_len - 9;
                    if prefix_without_9 + f32_data.len() != expected_req_size {
                        return Err(format!(
                            "SET_TENSOR_F16 size mismatch: decoded {} != expected {}",
                            prefix_without_9 + f32_data.len(),
                            expected_req_size
                        ));
                    }
                    out.extend_from_slice(hdr);
                    out.extend_from_slice(&f32_data);
                    if first_event.is_none() {
                        first_event = Some(WireCompressionEvent {
                            raw_bytes: f32_data.len(),
                            compressed_bytes: f16_data.len(),
                        });
                    }
                }
                WIRE_TAG_GET_TENSOR_RSP_F16 => {
                    let orig_rsp_size = read_u64_le(payload, 0)
                        .ok_or_else(|| "Truncated GET_TENSOR_RSP_F16 header".to_string())?;
                    let f16_data = &payload[8..];
                    let f32_data = decode_f16_bytes_to_f32(f16_data)
                        .ok_or_else(|| "Odd byte count in GET_TENSOR_RSP_F16".to_string())?;
                    if f32_data.len() as u64 != orig_rsp_size {
                        return Err(format!(
                            "GET_TENSOR_RSP_F16 size mismatch: decoded {} != expected {}",
                            f32_data.len(),
                            orig_rsp_size
                        ));
                    }
                    out.extend_from_slice(&orig_rsp_size.to_le_bytes());
                    out.extend_from_slice(&f32_data);
                    if first_event.is_none() {
                        first_event = Some(WireCompressionEvent {
                            raw_bytes: f32_data.len(),
                            compressed_bytes: f16_data.len(),
                        });
                    }
                }
                other => {
                    return Err(format!("Unknown SYNRPC02 wire frame tag: {}", other));
                }
            }
            consumed = frame_end;
        }

        if consumed > 0 {
            self.buf.drain(..consumed);
        }
        Ok((out, first_event))
    }
}

impl Default for WireStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ieee754_f16_roundtrip_edge_cases_and_precision() {
        for v in [0.0f32, -0.0f32, 1.0, -1.0, 0.0005, -2.75, 64500.0, f32::INFINITY, f32::NEG_INFINITY] {
            let h = f32_to_f16_bits(v);
            let r = f16_bits_to_f32(h);
            assert_eq!(v.to_bits() >> 31, r.to_bits() >> 31, "sign mismatch for {}", v);
            if v.is_infinite() {
                assert!(r.is_infinite());
            } else if v == 0.0 {
                assert_eq!(r, 0.0);
            } else {
                let rel = (r - v).abs() / v.abs().max(1e-6);
                assert!(rel < 1e-3, "precision loss too large for {}: got {}", v, r);
            }
        }
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
    }

    #[test]
    fn overflow_falls_back_to_lossless_raw() {
        let mut raw = vec![0u8; 4096];
        raw[0..4].copy_from_slice(&70_000.0f32.to_le_bytes());
        assert!(encode_f32_bytes_to_f16(&raw).is_none());

        let small = vec![0u8; 2048];
        assert!(encode_f32_bytes_to_f16(&small).is_none());
    }

    fn build_rpc_tensor(t_type: u32) -> Vec<u8> {
        let mut t = vec![0u8; 296];
        t[8..12].copy_from_slice(&t_type.to_le_bytes());
        t
    }

    fn make_ggml_req(cmd: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(9 + payload.len());
        out.push(cmd);
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn end_to_end_client_worker_framing_with_chunk_splits_and_hash_translation() {
        let mut hm = HashMap::new();
        let sparse_h = 0x1122_3344_5566_7788u64;
        let real_h = 0xAABB_CCDD_EEFF_0011u64;
        hm.insert(sparse_h, real_h);

        let mut client_framer = ClientGgmlFramer::new(true);
        let mut tracker = DecodedReqTracker::new();
        let mut worker_in_dec = WireStreamDecoder::new();
        let mut worker_rsp = WorkerRspFramer::new(true);
        let mut client_in_dec = WireStreamDecoder::new();
        let mut expected_q = VecDeque::new();

        // 1. INIT_TENSOR (296B)
        let init_req = make_ggml_req(RPC_CMD_INIT_TENSOR, &build_rpc_tensor(GGML_TYPE_F32));

        // 2. SET_TENSOR_HASH (sparse hash -> translated)
        let mut hash_payload = build_rpc_tensor(12); // Q4_K
        hash_payload.extend_from_slice(&0u64.to_le_bytes());
        hash_payload.extend_from_slice(&sparse_h.to_le_bytes());
        let hash_req = make_ggml_req(RPC_CMD_SET_TENSOR_HASH, &hash_payload);

        // 3. SET_TENSOR F32 activation (20,480 B = 5120 floats -> should compress 2x to 10,240 B!)
        let mut f32_act = Vec::with_capacity(5120 * 4);
        for i in 0..5120usize {
            let v = ((i as f32) - 2560.0) * 0.01;
            f32_act.extend_from_slice(&v.to_le_bytes());
        }
        let mut set_payload = build_rpc_tensor(GGML_TYPE_F32);
        set_payload.push(0); // cache_flag = 0
        set_payload.extend_from_slice(&64u64.to_le_bytes()); // offset
        set_payload.extend_from_slice(&f32_act);
        let set_req = make_ggml_req(RPC_CMD_SET_TENSOR, &set_payload);

        // 4. GET_TENSOR F32 activation (20,480 B)
        let mut get_payload = build_rpc_tensor(GGML_TYPE_F32);
        get_payload.extend_from_slice(&0u64.to_le_bytes());
        get_payload.extend_from_slice(&((5120 * 4) as u64).to_le_bytes());
        let get_req = make_ggml_req(RPC_CMD_GET_TENSOR, &get_payload);

        let mut combined_req = Vec::new();
        combined_req.extend_from_slice(&init_req);
        combined_req.extend_from_slice(&hash_req);
        combined_req.extend_from_slice(&set_req);
        combined_req.extend_from_slice(&get_req);

        // Feed in small fragmented chunks to exercise frame buffering
        let mut wire_tx = Vec::new();
        let mut saw_tx_compress = false;
        for chunk in combined_req.chunks(73) {
            let (w, ev) = client_framer.push_tcp_bytes(chunk, Some(&hm));
            wire_tx.extend_from_slice(&w);
            if let Some(ev) = ev {
                assert_eq!(ev.raw_bytes, 20_480);
                assert_eq!(ev.compressed_bytes, 10_240);
                saw_tx_compress = true;
            }
        }
        assert!(saw_tx_compress);
        assert!(wire_tx.len() < combined_req.len() - 9_000, "wire must save ~10 KB on SET_TENSOR");

        let mut worker_tcp_rx = Vec::new();
        for chunk in wire_tx.chunks(41) {
            let (tcp, _) = worker_in_dec.push_wire_bytes(chunk).unwrap();
            tracker.observe_tcp_bytes(&tcp, &mut expected_q);
            worker_tcp_rx.extend_from_slice(&tcp);
        }

        // Verify SET_TENSOR_HASH was translated on wire
        let hash_pos = init_req.len() + hash_req.len() - 8;
        assert_eq!(
            u64::from_le_bytes(worker_tcp_rx[hash_pos..hash_pos + 8].try_into().unwrap()),
            real_h
        );
        // Expected queue should contain ONLY SET_TENSOR_HASH (Raw) and GET_TENSOR (GetTensorF32)
        assert_eq!(expected_q.len(), 2);
        assert_eq!(expected_q[0], ExpectedRsp::Raw);
        assert_eq!(expected_q[1], ExpectedRsp::GetTensorF32(20_480));

        // Simulate Worker sending SET_TENSOR_HASH rsp (1 byte) + GET_TENSOR F32 rsp (20,480 bytes)
        let mut server_tcp_tx = Vec::new();
        server_tcp_tx.extend_from_slice(&1u64.to_le_bytes());
        server_tcp_tx.push(1u8);
        server_tcp_tx.extend_from_slice(&20_480u64.to_le_bytes());
        server_tcp_tx.extend_from_slice(&f32_act);

        let mut wire_rsp = Vec::new();
        let mut saw_rx_compress = false;
        for chunk in server_tcp_tx.chunks(111) {
            let (w, ev) = worker_rsp.push_tcp_bytes(chunk, &mut expected_q);
            wire_rsp.extend_from_slice(&w);
            if let Some(ev) = ev {
                assert_eq!(ev.raw_bytes, 20_480);
                assert_eq!(ev.compressed_bytes, 10_240);
                saw_rx_compress = true;
            }
        }
        assert!(saw_rx_compress);
        assert!(wire_rsp.len() < server_tcp_tx.len() - 9_000, "wire rsp must save ~10 KB");

        let mut client_tcp_rx = Vec::new();
        for chunk in wire_rsp.chunks(59) {
            let (tcp, _) = client_in_dec.push_wire_bytes(chunk).unwrap();
            client_tcp_rx.extend_from_slice(&tcp);
        }
        assert_eq!(client_tcp_rx.len(), server_tcp_tx.len());
        assert_eq!(u64::from_le_bytes(client_tcp_rx[9..17].try_into().unwrap()), 20_480);
    }
}
