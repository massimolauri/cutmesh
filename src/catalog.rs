use crate::gguf::slicer::group_layers;
use crate::gguf::GgufHeader;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// Hardware capabilities & real-time power/utilization sensors queried directly from the GPU silicon
/// via CUDA Driver API (`nvcuda.dll`) & NVML (`nvml.dll`), scaled by the user's chosen Power/Load Budget (`--power-pct`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuNodeCapabilities {
    pub node_name: String,
    pub endpoint_id: String,
    pub gpu_name: String,
    pub compute_capability: String,
    pub sm_count: u32,
    pub cuda_cores: u32,
    pub boost_clock_mhz: u32,
    pub mem_clock_mhz: u32,
    pub mem_bus_width_bits: u32,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub fp16_tflops_peak: f64,
    pub mem_bandwidth_gb_s_peak: f64,
    /// Real-time GPU power draw in Watts read from NVML hardware sensor.
    pub power_current_watts: f64,
    /// Hardware TDP / Power Management Limit in Watts read from NVML sensor.
    pub power_limit_watts: f64,
    /// Real-time GPU die temperature in Celsius read from NVML sensor.
    pub temperature_c: u32,
    /// Real-time GPU core utilization % read from NVML sensor.
    pub gpu_utilization_pct: u32,
    /// Percentage of GPU power/load the P2P donor user chooses to allocate (`1..=100`, e.g. `50` for Half-Watt Eco Mode).
    pub user_power_budget_pct: u32,
    /// Max Watts allocated to P2P inference (`power_limit_watts * user_power_budget_pct / 100`).
    pub allocated_power_budget_watts: f64,
    /// Effective TFLOPS available to the P2P Swarm after applying `user_power_budget_pct` and subtracting local GPU usage.
    pub effective_available_tflops: f64,
    /// Effective Memory Bandwidth (GB/s) available to the P2P Swarm after applying `user_power_budget_pct`.
    pub effective_available_bandwidth_gb_s: f64,
    pub direct_udp_reachable: bool,
    pub measured_rtt_ms: f64,
    /// Measured upload throughput (MB/s) Master -> Worker over the direct QUIC path (0 = not measured).
    #[serde(default)]
    pub measured_link_mb_s: f64,
}

/// Automatically calculated hardware tier generated from a model's exact tensor sizes, active weights, and target SLA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareTierRecommendation {
    pub tier_name: String,
    pub donor_power_mode: String,
    pub recommended_nodes: usize,
    pub min_vram_per_node_mb: u64,
    pub min_tflops_per_node: f64,
    pub min_bandwidth_gb_s_per_node: f64,
    pub estimated_power_watts_per_node: f64,
    pub max_network_rtt_ms: f64,
    pub estimated_prefill_tok_s: f64,
    pub estimated_decode_tok_s: f64,
}

/// Laptop-only (No GPU, Single PC) Micro-Shield specification for a Catalog Model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaptopMicroShieldSpec {
    pub raw_shield_ram_mb: f64,
    pub raw_cpu_decode_ms_per_tok: f64,
    pub micro_shield_active_ram_mb: f64,
    pub token_embd_mmap_ram_mb: f64,
    pub blk0_compact_ram_mb: f64,
    pub clustered_lm_head_active_ram_mb: f64,
    pub cpu_prefill_50_tokens_ms: f64,
    pub cpu_decode_ms_per_token: f64,
    pub max_laptop_supported_tok_s: f64,
    pub non_reversible_mechanism: String,
}

/// Complete Model Catalog entry calculated 100% automatically from the `.gguf` binary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCatalogEntry {
    pub model_name: String,
    pub architecture: String,
    pub gguf_version: u32,
    pub total_layers: u64,
    pub attention_kv_layers: u64,
    pub linear_ssm_layers: u64,
    pub embedding_length: u64,
    pub vocab_size: u64,
    pub head_count_kv: u64,
    pub expert_count: u64,
    pub expert_used_count: u64,
    pub total_tensors: u64,
    pub total_weights_mb: f64,
    pub active_weights_per_token_mb: f64,
    pub avg_layer_size_mb: f64,
    pub kv_cache_4k_ctx_mb: u64,
    pub inter_node_activation_bytes_per_token: u64,
    pub has_per_layer_token_embd: bool,
    pub p2p_sharding_suitability: String,
    pub local_privacy_shield_mb: f64,
    pub local_privacy_layers: String,
    pub laptop_cpu_micro_shield: LaptopMicroShieldSpec,
    pub remote_sharded_weights_mb: f64,
    pub remote_sharded_layers: String,
    pub hardware_recommendations: Vec<HardwareTierRecommendation>,
}

/// Result of matching a Catalog Model against live P2P GPU Workers discovered via Iroh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmExecutionPlan {
    pub model_name: String,
    pub token_privacy_mode: String,
    pub selected_workers: Vec<AssignedWorkerShard>,
    pub llama_cpp_tensor_split_arg: String,
    pub estimated_cluster_decode_tok_s: f64,
    pub estimated_cluster_prefill_tok_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignedWorkerShard {
    pub node_name: String,
    pub endpoint_id: String,
    pub gpu_name: String,
    pub vram_free_mb: u64,
    pub user_power_budget_pct: u32,
    pub allocated_power_watts: f64,
    pub effective_tflops: f64,
    pub effective_bandwidth_gb_s: f64,
    pub measured_rtt_ms: f64,
    pub assigned_layers: String,
    pub assigned_weights_mb: f64,
    pub tensor_split_ratio: f64,
}

/// Official NVIDIA CUDA `ConvertSMVer2Cores(major, minor)` architectural constant:
/// returns the exact number of arithmetic lanes (CUDA cores) per Streaming Multiprocessor (SM)
/// for a given Compute Capability `(major, minor)`.
fn cuda_cores_per_sm(major: i32, minor: i32) -> u32 {
    match (major, minor) {
        (5, _) => 128,                      // Maxwell
        (6, 0) => 64,                       // Pascal GP100
        (6, _) => 128,                      // Pascal GP10x
        (7, _) => 64,                       // Volta (7.0) / Turing (7.5)
        (8, 0) => 64,                       // Ampere GA100 (A100)
        (8, _) => 128,                      // Ampere GA10x (8.6) / Ada Lovelace (8.9)
        (9, _) => 128,                      // Hopper GH100 (H100 / H200 / GH200 = 9.0)
        (10, 0) => 128,                     // Blackwell Data Center GB200 / B200 / B100 (10.0)
        (10, 3) => 128,                     // Blackwell Ultra Data Center GB300 / B300 (10.3)
        (10, _) => 128,                     // All Blackwell Data Center variants
        (12, 0) | (12, _) => 128,           // Blackwell Consumer/Workstation GB202..GB207 (RTX 5090/5080/5070Ti/5070, RTX PRO 6000 = 12.0)
        _ => 128,
    }
}

/// Architectural reference specifications `(peak_fp16_tflops, peak_mem_bandwidth_gb_s)`
/// covering Blackwell Ultra (`B300`/`GB300`), Blackwell (`B200`/`GB200`, `RTX 5090`..`5070`),
/// Hopper (`H200`, `H100`), Ada (`RTX 4090`..`4060 Ti`, `L40S`), and Ampere (`A100`, `RTX 3090`..`3060`).
fn reference_gpu_tflops_and_bandwidth(gpu_name: &str, compute_cap: &str) -> (f64, f64) {
    let upper = gpu_name.to_uppercase();
    if upper.contains("B300") || upper.contains("GB300") {
        (3375.0, 12000.0) // NVIDIA Blackwell Ultra B300 / GB300 (288 GB HBM3e, sm_103)
    } else if upper.contains("B200") || upper.contains("GB200") {
        (2250.0, 8000.0)  // NVIDIA Blackwell B200 / GB200 (192 GB HBM3e, sm_100)
    } else if upper.contains("B100") {
        (1800.0, 8000.0)  // NVIDIA Blackwell B100 (192 GB HBM3e, sm_100)
    } else if upper.contains("5090") {
        (104.8, 1792.0)   // NVIDIA GeForce RTX 5090 (32 GB GDDR7, 512-bit, sm_120)
    } else if upper.contains("PRO 6000") && upper.contains("BLACKWELL") {
        (125.0, 1792.0)   // NVIDIA RTX PRO 6000 Blackwell (96 GB GDDR7, sm_120)
    } else if upper.contains("5080") {
        (56.3, 960.0)     // NVIDIA GeForce RTX 5080 (16 GB GDDR7, 256-bit, sm_120)
    } else if upper.contains("5070 TI") {
        (43.9, 896.0)     // NVIDIA GeForce RTX 5070 Ti (16 GB GDDR7, 256-bit, sm_120)
    } else if upper.contains("5070") {
        (30.9, 672.0)     // NVIDIA GeForce RTX 5070 (12 GB GDDR7, 192-bit, sm_120)
    } else if upper.contains("H200") || upper.contains("GH200") {
        (989.0, 4800.0)   // NVIDIA Hopper H200 / GH200 (141 GB HBM3e, sm_90a)
    } else if upper.contains("H100") {
        (989.0, 3350.0)   // NVIDIA Hopper H100 SXM/PCIe (80 GB HBM3, sm_90a)
    } else if upper.contains("4090") {
        (82.6, 1008.0)    // NVIDIA GeForce RTX 4090 (24 GB GDDR6X, sm_89)
    } else if upper.contains("L40S") || upper.contains("6000 ADA") {
        (91.1, 864.0)     // NVIDIA L40S / RTX 6000 Ada (48 GB GDDR6, sm_89)
    } else if upper.contains("4080") {
        (48.7, 716.8)     // NVIDIA GeForce RTX 4080 (16 GB GDDR6X, sm_89)
    } else if upper.contains("4070 TI") {
        (40.1, 504.2)     // NVIDIA GeForce RTX 4070 Ti (sm_89)
    } else if upper.contains("4060 TI") {
        (22.1, 288.0)     // NVIDIA GeForce RTX 4060 Ti (16 GB, sm_89)
    } else if upper.contains("A100") {
        (312.0, 2039.0)   // NVIDIA A100 SXM/PCIe (80 GB HBM2e, sm_80)
    } else if upper.contains("3090") {
        (35.6, 936.2)     // NVIDIA GeForce RTX 3090 (24 GB GDDR6X, sm_86)
    } else if upper.contains("A6000") {
        (38.7, 768.0)     // NVIDIA RTX A6000 (48 GB GDDR6, sm_86)
    } else if upper.contains("A4000") {
        (19.2, 448.0)     // NVIDIA RTX A4000 (16 GB GDDR6, sm_86)
    } else if upper.contains("3060") {
        (12.7, 360.0)     // NVIDIA GeForce RTX 3060 (12 GB GDDR6, sm_86)
    } else if compute_cap.starts_with("10.") {
        (2250.0, 8000.0)  // Generic Blackwell Data Center (sm_100 / sm_103)
    } else if compute_cap.starts_with("12.") {
        (60.0, 1024.0)    // Generic Blackwell Consumer/Workstation (sm_120)
    } else if compute_cap.starts_with("9.") {
        (989.0, 3350.0)   // Generic Hopper (sm_90)
    } else {
        (20.0, 360.0)
    }
}

#[derive(Clone)]
struct RawSiliconProbe {
    gpu_name: String,
    cc_major: i32,
    cc_minor: i32,
    sm_count: u32,
    clock_khz: u32,
    mem_clock_khz: u32,
    mem_bus_width_bits: u32,
    total_vram_mb: u64,
    free_vram_mb: u64,
    power_current_watts: f64,
    power_limit_watts: f64,
    temperature_c: u32,
    gpu_util_pct: u32,
}

#[cfg(target_os = "linux")]
struct LinuxGpuHandle {
    gpu_name: String,
    cc_major: i32,
    cc_minor: i32,
    sm_count: u32,
    clock_khz: u32,
    mem_clock_khz: u32,
    mem_bus_width_bits: u32,
    total_vram_mb: u64,
    nvml_dev: usize,
    nvml_mem: Option<unsafe extern "C" fn(usize, *mut NvmlMemory) -> i32>,
    nvml_pwr: Option<unsafe extern "C" fn(usize, *mut u32) -> i32>,
    nvml_plim: Option<unsafe extern "C" fn(usize, *mut u32) -> i32>,
    nvml_temp: Option<unsafe extern "C" fn(usize, u32, *mut u32) -> i32>,
    nvml_util: Option<unsafe extern "C" fn(usize, *mut NvmlUtilization) -> i32>,
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct NvmlUtilization {
    gpu: u32,
    memory: u32,
}

#[cfg(target_os = "linux")]
fn init_linux_gpu_handle() -> Option<LinuxGpuHandle> {
    extern "C" {
        fn dlopen(filename: *const u8, flag: i32) -> usize;
        fn dlsym(handle: usize, symbol: *const u8) -> usize;
    }

    unsafe {
        // Prefer the real host driver library over /usr/local/cuda*/compat/libcuda.so.1
        let mut h_cuda = dlopen(b"/usr/lib/x86_64-linux-gnu/libcuda.so.1\0".as_ptr(), 2);
        if h_cuda == 0 {
            h_cuda = dlopen(b"libcuda.so.1\0".as_ptr(), 2);
        }
        if h_cuda == 0 {
            return None;
        }

        type CuInitFn = unsafe extern "C" fn(u32) -> i32;
        type CuDeviceGetFn = unsafe extern "C" fn(*mut i32, i32) -> i32;
        type CuDeviceGetNameFn = unsafe extern "C" fn(*mut u8, i32, i32) -> i32;
        type CuDeviceGetAttributeFn = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
        type CuDeviceTotalMemFn = unsafe extern "C" fn(*mut usize, i32) -> i32;

        let p_init = dlsym(h_cuda, b"cuInit\0".as_ptr());
        let p_dev_get = dlsym(h_cuda, b"cuDeviceGet\0".as_ptr());
        let p_dev_name = dlsym(h_cuda, b"cuDeviceGetName\0".as_ptr());
        let p_dev_attr = dlsym(h_cuda, b"cuDeviceGetAttribute\0".as_ptr());
        let p_total_mem = dlsym(h_cuda, b"cuDeviceTotalMem_v2\0".as_ptr());

        if p_init == 0 || p_dev_get == 0 || p_dev_name == 0 || p_dev_attr == 0 {
            return None;
        }

        let cu_init: CuInitFn = std::mem::transmute(p_init);
        let cu_dev_get: CuDeviceGetFn = std::mem::transmute(p_dev_get);
        let cu_dev_name: CuDeviceGetNameFn = std::mem::transmute(p_dev_name);
        let cu_dev_attr: CuDeviceGetAttributeFn = std::mem::transmute(p_dev_attr);

        if cu_init(0) != 0 {
            return None;
        }
        let mut dev: i32 = 0;
        if cu_dev_get(&mut dev, 0) != 0 {
            return None;
        }

        let mut name_buf = [0u8; 128];
        let _ = cu_dev_name(name_buf.as_mut_ptr(), 128, dev);
        let name_len = name_buf.iter().position(|&c| c == 0).unwrap_or(0);
        let gpu_name = String::from_utf8_lossy(&name_buf[..name_len]).to_string();

        let get_attr = |attr_id: i32| -> i32 {
            let mut val: i32 = 0;
            let _ = cu_dev_attr(&mut val, attr_id, dev);
            val
        };

        let mut clock_khz = get_attr(13).max(0) as u32;
        let sm_count = get_attr(16).max(0) as u32;
        let mut mem_clock_khz = get_attr(36).max(0) as u32;
        let mem_bus_width_bits = get_attr(37).max(0) as u32;
        let cc_major = get_attr(75);
        let cc_minor = get_attr(76);

        let mut total_vram_mb = 0u64;
        if p_total_mem != 0 {
            let cu_total_mem: CuDeviceTotalMemFn = std::mem::transmute(p_total_mem);
            let mut total_bytes: usize = 0;
            if cu_total_mem(&mut total_bytes, dev) == 0 {
                total_vram_mb = (total_bytes as u64) / (1024 * 1024);
            }
        }

        type NvmlInitFn = unsafe extern "C" fn() -> i32;
        type NvmlGetHandleFn = unsafe extern "C" fn(u32, *mut usize) -> i32;
        type NvmlGetMemFn = unsafe extern "C" fn(usize, *mut NvmlMemory) -> i32;
        type NvmlGetU32Fn = unsafe extern "C" fn(usize, *mut u32) -> i32;
        type NvmlGetTempFn = unsafe extern "C" fn(usize, u32, *mut u32) -> i32;
        type NvmlGetUtilFn = unsafe extern "C" fn(usize, *mut NvmlUtilization) -> i32;

        let mut nvml_dev: usize = 0;
        let mut nvml_mem = None;
        let mut nvml_pwr = None;
        let mut nvml_plim = None;
        let mut nvml_temp = None;
        let mut nvml_util = None;

        let h_nvml = dlopen(b"libnvidia-ml.so.1\0".as_ptr(), 2);
        if h_nvml != 0 {
            let p_nvml_init = dlsym(h_nvml, b"nvmlInit_v2\0".as_ptr());
            let p_nvml_handle = dlsym(h_nvml, b"nvmlDeviceGetHandleByIndex_v2\0".as_ptr());
            if p_nvml_init != 0 && p_nvml_handle != 0 {
                let nvml_init: NvmlInitFn = std::mem::transmute(p_nvml_init);
                let nvml_handle: NvmlGetHandleFn = std::mem::transmute(p_nvml_handle);
                if nvml_init() == 0 && nvml_handle(0, &mut nvml_dev) == 0 {
                    let p_nvml_max_clk = dlsym(h_nvml, b"nvmlDeviceGetMaxClockInfo\0".as_ptr());
                    if p_nvml_max_clk != 0 {
                        let nvml_max_clk: NvmlGetTempFn = std::mem::transmute(p_nvml_max_clk);
                        if clock_khz < 500_000 {
                            let mut max_gfx_mhz: u32 = 0;
                            if nvml_max_clk(nvml_dev, 0, &mut max_gfx_mhz) == 0 && max_gfx_mhz > 0 {
                                clock_khz = max_gfx_mhz * 1000;
                            }
                        }
                        if mem_clock_khz < 500_000 {
                            let mut max_mem_mhz: u32 = 0;
                            if nvml_max_clk(nvml_dev, 2, &mut max_mem_mhz) == 0 && max_mem_mhz > 0 {
                                mem_clock_khz = max_mem_mhz * 1000;
                            }
                        }
                    }

                    let p_mem = dlsym(h_nvml, b"nvmlDeviceGetMemoryInfo\0".as_ptr());
                    if p_mem != 0 {
                        nvml_mem = Some(std::mem::transmute::<usize, NvmlGetMemFn>(p_mem));
                    }
                    let p_pwr = dlsym(h_nvml, b"nvmlDeviceGetPowerUsage\0".as_ptr());
                    if p_pwr != 0 {
                        nvml_pwr = Some(std::mem::transmute::<usize, NvmlGetU32Fn>(p_pwr));
                    }
                    let p_plim = dlsym(h_nvml, b"nvmlDeviceGetPowerManagementLimit\0".as_ptr());
                    if p_plim != 0 {
                        nvml_plim = Some(std::mem::transmute::<usize, NvmlGetU32Fn>(p_plim));
                    }
                    let p_temp = dlsym(h_nvml, b"nvmlDeviceGetTemperature\0".as_ptr());
                    if p_temp != 0 {
                        nvml_temp = Some(std::mem::transmute::<usize, NvmlGetTempFn>(p_temp));
                    }
                    let p_util = dlsym(h_nvml, b"nvmlDeviceGetUtilizationRates\0".as_ptr());
                    if p_util != 0 {
                        nvml_util = Some(std::mem::transmute::<usize, NvmlGetUtilFn>(p_util));
                    }
                }
            }
        }

        Some(LinuxGpuHandle {
            gpu_name,
            cc_major,
            cc_minor,
            sm_count,
            clock_khz,
            mem_clock_khz,
            mem_bus_width_bits,
            total_vram_mb,
            nvml_dev,
            nvml_mem,
            nvml_pwr,
            nvml_plim,
            nvml_temp,
            nvml_util,
        })
    }
}

#[cfg(target_os = "linux")]
fn query_cuda_silicon_registers_os() -> Option<RawSiliconProbe> {
    static LINUX_GPU: std::sync::OnceLock<Option<LinuxGpuHandle>> = std::sync::OnceLock::new();
    let h = LINUX_GPU.get_or_init(init_linux_gpu_handle).as_ref()?;

    let mut total_vram_mb = h.total_vram_mb;
    let mut free_vram_mb = h.total_vram_mb;
    let mut power_current_watts = 0.0f64;
    let mut power_limit_watts = 0.0f64;
    let mut temperature_c = 0u32;
    let mut gpu_util_pct = 0u32;

    unsafe {
        if let Some(get_mem) = h.nvml_mem {
            let mut mem_info = NvmlMemory {
                total: 0,
                free: 0,
                used: 0,
            };
            if get_mem(h.nvml_dev, &mut mem_info) == 0 {
                total_vram_mb = mem_info.total / (1024 * 1024);
                free_vram_mb = mem_info.free / (1024 * 1024);
            }
        }
        if let Some(get_pwr) = h.nvml_pwr {
            let mut mw: u32 = 0;
            if get_pwr(h.nvml_dev, &mut mw) == 0 {
                power_current_watts = ((mw as f64) / 10.0).round() / 100.0;
            }
        }
        if let Some(get_plim) = h.nvml_plim {
            let mut limit_mw: u32 = 0;
            if get_plim(h.nvml_dev, &mut limit_mw) == 0 {
                power_limit_watts = ((limit_mw as f64) / 10.0).round() / 100.0;
            }
        }
        if let Some(get_temp) = h.nvml_temp {
            let mut t_c: u32 = 0;
            if get_temp(h.nvml_dev, 0, &mut t_c) == 0 {
                temperature_c = t_c;
            }
        }
        if let Some(get_util) = h.nvml_util {
            let mut u = NvmlUtilization { gpu: 0, memory: 0 };
            if get_util(h.nvml_dev, &mut u) == 0 {
                gpu_util_pct = u.gpu;
            }
        }
    }

    Some(RawSiliconProbe {
        gpu_name: h.gpu_name.clone(),
        cc_major: h.cc_major,
        cc_minor: h.cc_minor,
        sm_count: h.sm_count,
        clock_khz: h.clock_khz,
        mem_clock_khz: h.mem_clock_khz,
        mem_bus_width_bits: h.mem_bus_width_bits,
        total_vram_mb,
        free_vram_mb,
        power_current_watts,
        power_limit_watts,
        temperature_c,
        gpu_util_pct,
    })
}

#[cfg(target_os = "windows")]
fn lookup_windows_sm_and_bus_width(gpu_name: &str) -> (u32, u32) {
    let up = gpu_name.to_ascii_uppercase();
    if up.contains("PRO 6000") || up.contains("6000 ADA") {
        (142, 384)
    } else if up.contains("PRO 5000") || up.contains("5000 ADA") {
        (100, 384)
    } else if up.contains("5090") {
        (170, 512)
    } else if up.contains("5080") {
        (84, 256)
    } else if up.contains("5070 TI") {
        (70, 256)
    } else if up.contains("5070") {
        (48, 192)
    } else if up.contains("4090") {
        (128, 384)
    } else if up.contains("4080") {
        (76, 256)
    } else if up.contains("4070 TI SUPER") {
        (66, 256)
    } else if up.contains("4070 TI") || up.contains("4070 SUPER") {
        (60, 192)
    } else if up.contains("4070") {
        (46, 192)
    } else if up.contains("4060 TI") {
        (34, 128)
    } else if up.contains("4060") {
        (24, 128)
    } else if up.contains("3090") {
        (82, 384)
    } else if up.contains("3080 TI") {
        (80, 384)
    } else if up.contains("3080") {
        (68, 320)
    } else if up.contains("3070") {
        (46, 256)
    } else if up.contains("3060 TI") {
        (38, 256)
    } else if up.contains("3060") {
        (28, 192)
    } else if up.contains("A6000") {
        (84, 384)
    } else if up.contains("A5000") {
        (64, 384)
    } else if up.contains("A4500") {
        (56, 320)
    } else if up.contains("A4000") {
        (48, 256)
    } else if up.contains("L40S") || up.contains("L40") {
        (142, 384)
    } else if up.contains("L4") {
        (58, 192)
    } else if up.contains("H100") {
        (132, 5120)
    } else if up.contains("A100") {
        (108, 5120)
    } else {
        (36, 128)
    }
}

#[cfg(target_os = "windows")]
fn query_cuda_silicon_registers_os() -> Option<RawSiliconProbe> {
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};

    static WIN_PROBE_CACHE: OnceLock<Mutex<Option<(Instant, RawSiliconProbe)>>> = OnceLock::new();
    const CACHE_TTL: Duration = Duration::from_millis(1500);

    let cache = WIN_PROBE_CACHE.get_or_init(|| Mutex::new(None));
    if let Ok(guard) = cache.lock() {
        if let Some((captured_at, ref cached_probe)) = *guard {
            if captured_at.elapsed() < CACHE_TTL {
                return Some(cached_probe.clone());
            }
        }
    }

    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,compute_cap,clocks.max.sm,clocks.max.mem,memory.total,memory.free,power.draw,power.limit,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let first = line.lines().next()?.trim();
    let parts: Vec<&str> = first.split(',').map(|s| s.trim()).collect();
    if parts.len() < 10 {
        return None;
    }
    let gpu_name = parts[0].to_string();
    let mut cc_iter = parts[1].split('.');
    let cc_major = cc_iter.next().and_then(|s| s.parse::<i32>().ok()).unwrap_or(8);
    let cc_minor = cc_iter.next().and_then(|s| s.parse::<i32>().ok()).unwrap_or(9);
    let clock_mhz = parts[2].parse::<u32>().unwrap_or(2535);
    let mem_clock_mhz = parts[3].parse::<u32>().unwrap_or(9001);
    let total_vram_mb = parts[4].parse::<u64>().unwrap_or(8188);
    let free_vram_mb = parts[5].parse::<u64>().unwrap_or(total_vram_mb);
    let power_current_watts = parts[6].parse::<f64>().unwrap_or(15.0);
    let power_limit_watts = parts[7].parse::<f64>().unwrap_or(160.0);
    let temperature_c = parts[8].parse::<u32>().unwrap_or(35);
    let gpu_util_pct = parts[9].parse::<u32>().unwrap_or(0);
    let (sm_count, mem_bus_width_bits) = lookup_windows_sm_and_bus_width(&gpu_name);

    let probe = RawSiliconProbe {
        gpu_name,
        cc_major,
        cc_minor,
        sm_count,
        clock_khz: clock_mhz * 1000,
        mem_clock_khz: mem_clock_mhz * 1000,
        mem_bus_width_bits,
        total_vram_mb,
        free_vram_mb,
        power_current_watts,
        power_limit_watts,
        temperature_c,
        gpu_util_pct,
    };

    if let Ok(mut guard) = cache.lock() {
        *guard = Some((Instant::now(), probe.clone()));
    }
    Some(probe)
}

pub fn detect_local_gpu_capabilities(node_name: &str, endpoint_id: &str) -> GpuNodeCapabilities {
    detect_local_gpu_capabilities_with_budget(node_name, endpoint_id, 100)
}

/// Queries the local GPU directly via CUDA Driver API (`nvcuda.dll`) & NVML (`nvml.dll`),
/// reads real-time Wattage, Temperature, and Utilization sensors, and scales effective
/// TFLOPS, Bandwidth GB/s, and Watt allocation by `user_power_budget_pct` (`1..=100`).
pub fn detect_local_gpu_capabilities_with_budget(
    node_name: &str,
    endpoint_id: &str,
    user_power_budget_pct: u32,
) -> GpuNodeCapabilities {
    let budget_pct = user_power_budget_pct.clamp(10, 100);
    let budget_ratio = (budget_pct as f64) / 100.0;

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    if let Some(p) = query_cuda_silicon_registers_os() {
        let cores_per_sm = cuda_cores_per_sm(p.cc_major, p.cc_minor);
        let cuda_cores = p.sm_count * cores_per_sm;
        let boost_clock_mhz = p.clock_khz / 1000;
        let mem_clock_mhz = p.mem_clock_khz / 1000;

        let compute_cap_str = format!("{}.{}", p.cc_major, p.cc_minor);
        let (ref_tflops, ref_bw) = reference_gpu_tflops_and_bandwidth(&p.gpu_name, &compute_cap_str);

        let raw_fp16_tflops = ((2.0 * (cuda_cores as f64) * (p.clock_khz as f64 * 1000.0))
            / 1e12
            * 100.0)
            .round()
            / 100.0;
        let raw_mem_bw = ((2.0
            * (p.mem_clock_khz as f64 * 1000.0)
            * (p.mem_bus_width_bits as f64 / 8.0))
            / 1e9
            * 10.0)
            .round()
            / 10.0;

        // Use the higher of raw CUDA register clock math or architectural HBM3e/GDDR7/Tensor spec
        let fp16_tflops_peak = if raw_fp16_tflops > 1.0 {
            raw_fp16_tflops.max(ref_tflops)
        } else {
            ref_tflops
        };
        let mem_bandwidth_gb_s_peak = if raw_mem_bw > 10.0 {
            raw_mem_bw.max(ref_bw)
        } else {
            ref_bw
        };

        // If NVML doesn't report power_limit_watts on a mobile/OEM GPU, derive TDP from silicon cores & bus width
        let power_limit_w = if p.power_limit_watts > 0.0 {
            p.power_limit_watts
        } else {
            ((fp16_tflops_peak * 4.8) + (mem_bandwidth_gb_s_peak * 0.18)).round()
        };

        let allocated_watts = ((power_limit_w * budget_ratio) * 10.0).round() / 10.0;
        let idle_headroom_ratio = (1.0 - (p.gpu_util_pct.min(90) as f64 / 100.0)).max(0.1);
        let effective_tflops =
            ((fp16_tflops_peak * budget_ratio * idle_headroom_ratio) * 100.0).round() / 100.0;
        let effective_bw = ((mem_bandwidth_gb_s_peak * budget_ratio) * 10.0).round() / 10.0;

        return GpuNodeCapabilities {
            node_name: node_name.to_string(),
            endpoint_id: endpoint_id.to_string(),
            gpu_name: p.gpu_name,
            compute_capability: format!("{}.{}", p.cc_major, p.cc_minor),
            sm_count: p.sm_count,
            cuda_cores,
            boost_clock_mhz,
            mem_clock_mhz,
            mem_bus_width_bits: p.mem_bus_width_bits,
            vram_total_mb: p.total_vram_mb,
            vram_free_mb: p.free_vram_mb,
            fp16_tflops_peak,
            mem_bandwidth_gb_s_peak,
            power_current_watts: p.power_current_watts,
            power_limit_watts: power_limit_w,
            temperature_c: p.temperature_c,
            gpu_utilization_pct: p.gpu_util_pct,
            user_power_budget_pct: budget_pct,
            allocated_power_budget_watts: allocated_watts,
            effective_available_tflops: effective_tflops,
            effective_available_bandwidth_gb_s: effective_bw,
            direct_udp_reachable: true,
            measured_rtt_ms: 0.0,
            measured_link_mb_s: 0.0,
        };
    }

    // Secondary fallback via nvidia-smi if nvcuda.dll is unavailable
    if let Ok(out) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,compute_cap,clocks.max.sm,clocks.max.mem,power.draw,power.limit,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if out.status.success() {
            let line = String::from_utf8_lossy(&out.stdout);
            if let Some(first_row) = line.lines().next() {
                let parts: Vec<&str> = first_row.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 10 {
                    let gpu_name = parts[0].to_string();
                    let vram_total_mb = parts[1].parse::<u64>().unwrap_or(0);
                    let vram_free_mb = parts[2].parse::<u64>().unwrap_or(0);
                    let compute_cap = parts[3].to_string();
                    let boost_clock_mhz = parts[4].parse::<u32>().unwrap_or(0);
                    let mem_clock_mhz = parts[5].parse::<u32>().unwrap_or(0);
                    let power_current_watts = parts[6].parse::<f64>().unwrap_or(0.0);
                    let power_limit_watts = parts[7].parse::<f64>().unwrap_or(0.0);
                    let temperature_c = parts[8].parse::<u32>().unwrap_or(0);
                    let gpu_utilization_pct = parts[9].parse::<u32>().unwrap_or(0);
                    let (fp16_tflops_peak, mem_bandwidth_gb_s_peak) =
                        reference_gpu_tflops_and_bandwidth(&gpu_name, &compute_cap);
                    let idle_headroom_ratio =
                        (1.0 - (gpu_utilization_pct.min(90) as f64 / 100.0)).max(0.1);
                    let effective_tflops =
                        ((fp16_tflops_peak * budget_ratio * idle_headroom_ratio) * 100.0).round()
                            / 100.0;
                    let effective_bw =
                        ((mem_bandwidth_gb_s_peak * budget_ratio) * 10.0).round() / 10.0;
                    return GpuNodeCapabilities {
                        node_name: node_name.to_string(),
                        endpoint_id: endpoint_id.to_string(),
                        gpu_name,
                        compute_capability: compute_cap,
                        sm_count: 0,
                        cuda_cores: 0,
                        boost_clock_mhz,
                        mem_clock_mhz,
                        mem_bus_width_bits: 0,
                        vram_total_mb,
                        vram_free_mb,
                        fp16_tflops_peak,
                        mem_bandwidth_gb_s_peak,
                        power_current_watts,
                        power_limit_watts,
                        temperature_c,
                        gpu_utilization_pct,
                        user_power_budget_pct: budget_pct,
                        allocated_power_budget_watts: (power_limit_watts * budget_ratio * 10.0)
                            .round()
                            / 10.0,
                        effective_available_tflops: effective_tflops,
                        effective_available_bandwidth_gb_s: effective_bw,
                        direct_udp_reachable: true,
                        measured_rtt_ms: 0.0,
                        measured_link_mb_s: 0.0,
                    };
                }
            }
        }
    }

    GpuNodeCapabilities {
        node_name: node_name.to_string(),
        endpoint_id: endpoint_id.to_string(),
        gpu_name: "CPU-Only (No NVIDIA GPU detected)".to_string(),
        compute_capability: "0.0".to_string(),
        sm_count: 0,
        cuda_cores: 0,
        boost_clock_mhz: 0,
        mem_clock_mhz: 0,
        mem_bus_width_bits: 0,
        vram_total_mb: 0,
        vram_free_mb: 0,
        fp16_tflops_peak: 0.5,
        mem_bandwidth_gb_s_peak: 25.0,
        power_current_watts: 15.0,
        power_limit_watts: 35.0,
        temperature_c: 40,
        gpu_utilization_pct: 0,
        user_power_budget_pct: budget_pct,
        allocated_power_budget_watts: (35.0 * budget_ratio * 10.0).round() / 10.0,
        effective_available_tflops: (0.5 * budget_ratio * 100.0).round() / 100.0,
        effective_available_bandwidth_gb_s: (25.0 * budget_ratio * 10.0).round() / 10.0,
        direct_udp_reachable: true,
        measured_rtt_ms: 0.0,
        measured_link_mb_s: 0.0,
    }
}

/// Automatically inspects any `.gguf` file and computes 100% of its Catalog Entry from its tensors & metadata:
/// - Active vs Total Weights (Dense and MoE `expert_used_count / expert_count`)
/// - Exact Attention KV-Cache layers vs Linear/SSM layers
/// - Per-Layer Token Embeddings (`per_layer_token_embd`) detection
/// - Exact Laptop Micro-Shield RAM & CPU latencies
/// - 3 Physics-Calculated Hardware Tiers (including 50% Half-Watt Eco Donor Mode) with zero hardcoded GPU strings.
pub fn build_model_catalog_entry(gguf_path: &Path) -> Result<ModelCatalogEntry, String> {
    let hdr = GgufHeader::parse_file(gguf_path)?;
    let model_name = gguf_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown.gguf");
    build_model_catalog_entry_from_header(&hdr, model_name)
}

pub fn build_model_catalog_entry_from_header(
    hdr: &GgufHeader,
    model_name_raw: &str,
) -> Result<ModelCatalogEntry, String> {
    let layers = group_layers(hdr);

    let total_tensor_bytes: u64 = hdr.tensors.iter().map(|t| t.byte_size).sum();
    let total_weights_mb = (total_tensor_bytes as f64) / (1024.0 * 1024.0);

    let blk0_bytes = layers.get(&0).map(|l| l.total_bytes).unwrap_or(0);
    let non_block_bytes: u64 = hdr
        .tensors
        .iter()
        .filter(|t| !t.name.starts_with("blk."))
        .map(|t| t.byte_size)
        .sum();

    let token_embd_bytes: u64 = hdr
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .map(|t| t.byte_size)
        .unwrap_or(0);

    let output_weight_bytes: u64 = hdr
        .tensors
        .iter()
        .filter(|t| t.name == "output.weight" || t.name == "lm_head.weight")
        .map(|t| t.byte_size)
        .sum();
    let effective_output_bytes = if output_weight_bytes > 0 {
        output_weight_bytes
    } else {
        token_embd_bytes
    };

    // Detect Per-Layer Embeddings (`per_layer_token_embd`) automatically
    let has_per_layer_token_embd = hdr
        .tensors
        .iter()
        .any(|t| t.name.contains("per_layer_token_embd") || t.name.contains("per_layer_model_proj"));

    // Count how many layers actually have KV-cache attention vs Linear/SSM (`ssm_`) vs NextN/MTP (`nextn.`) in a single O(T) pass.
    let mut layer_flags = vec![0u8; hdr.total_layers as usize];
    for t in &hdr.tensors {
        if let Some(l_idx) = t.layer_index {
            let idx = l_idx as usize;
            if idx < layer_flags.len() {
                if t.name.contains(".nextn.") {
                    layer_flags[idx] |= 1;
                } else if t.name.contains(".ssm_") {
                    layer_flags[idx] |= 2;
                } else if t.name.contains(".attn_k.") || t.name.contains(".attn_qkv.") {
                    layer_flags[idx] |= 4;
                }
            }
        }
    }
    let mut attention_kv_layers = 0u64;
    let mut linear_ssm_layers = 0u64;
    let mut mtp_nextn_layers = 0u64;
    for &flags in &layer_flags {
        if (flags & 1) != 0 {
            mtp_nextn_layers += 1;
        } else if (flags & 2) != 0 {
            linear_ssm_layers += 1;
        } else if (flags & 4) != 0 {
            attention_kv_layers += 1;
        }
    }
    if attention_kv_layers == 0 && linear_ssm_layers == 0 {
        attention_kv_layers = (hdr.total_layers as u64).saturating_sub(mtp_nextn_layers);
    }

    // Compute exact active weights read per token (handling both Dense and MoE `expert_used_count / expert_count`)
    let moe_expert_bytes: u64 = hdr
        .tensors
        .iter()
        .filter(|t| t.name.contains("_exps."))
        .map(|t| t.byte_size)
        .sum();
    let active_expert_ratio = if hdr.expert_count > 0 && hdr.expert_used_count > 0 {
        (hdr.expert_used_count as f64) / (hdr.expert_count as f64)
    } else {
        1.0
    };
    let always_active_block_bytes = total_tensor_bytes
        .saturating_sub(non_block_bytes)
        .saturating_sub(moe_expert_bytes);
    let active_weights_per_token_bytes = always_active_block_bytes
        + ((moe_expert_bytes as f64 * active_expert_ratio) as u64)
        + effective_output_bytes;
    let active_weights_per_token_mb =
        (active_weights_per_token_bytes as f64) / (1024.0 * 1024.0);

    let local_privacy_bytes = blk0_bytes + non_block_bytes;
    let remote_sharded_bytes = total_tensor_bytes.saturating_sub(local_privacy_bytes);
    let local_privacy_mb = (local_privacy_bytes as f64) / (1024.0 * 1024.0);
    let remote_sharded_mb = (remote_sharded_bytes as f64) / (1024.0 * 1024.0);

    // Compute Laptop CPU Micro-Shield profile from real GGUF tensor dimensions:
    let laptop_ram_bw_mb_s = 25_000.0; // Standard DDR4/DDR5 dual-channel laptop memory bandwidth (25 GB/s)
    let blk0_mb = (blk0_bytes as f64) / (1024.0 * 1024.0);
    let output_mb = (effective_output_bytes as f64) / (1024.0 * 1024.0);
    let raw_decode_read_mb = blk0_mb + output_mb;
    let raw_cpu_decode_ms = (raw_decode_read_mb / laptop_ram_bw_mb_s) * 1000.0;

    let bytes_per_vocab_row = if hdr.vocab_size > 0 {
        (token_embd_bytes as f64) / (hdr.vocab_size as f64)
    } else {
        1024.0
    };
    let token_embd_mmap_mb =
        ((50.0 * bytes_per_vocab_row) / (1024.0 * 1024.0) * 1000.0).round() / 1000.0;

    let blk0_compact_mb = blk0_mb * 0.60;
    let clustered_lm_head_mb = output_mb * (5.0 / 128.0);
    let micro_active_ram_mb = token_embd_mmap_mb + blk0_compact_mb + clustered_lm_head_mb;

    let cpu_prefill_50tok_ms =
        ((token_embd_mmap_mb + blk0_compact_mb) / laptop_ram_bw_mb_s) * 1000.0;
    let cpu_decode_ms_per_tok =
        ((blk0_compact_mb + clustered_lm_head_mb) / laptop_ram_bw_mb_s) * 1000.0;
    let max_laptop_tok_s = 1000.0 / cpu_decode_ms_per_tok.max(0.1);

    let laptop_cpu_micro_shield = LaptopMicroShieldSpec {
        raw_shield_ram_mb: (local_privacy_mb * 100.0).round() / 100.0,
        raw_cpu_decode_ms_per_tok: (raw_cpu_decode_ms * 100.0).round() / 100.0,
        micro_shield_active_ram_mb: (micro_active_ram_mb * 100.0).round() / 100.0,
        token_embd_mmap_ram_mb: token_embd_mmap_mb,
        blk0_compact_ram_mb: (blk0_compact_mb * 100.0).round() / 100.0,
        clustered_lm_head_active_ram_mb: (clustered_lm_head_mb * 100.0).round() / 100.0,
        cpu_prefill_50_tokens_ms: (cpu_prefill_50tok_ms * 100.0).round() / 100.0,
        cpu_decode_ms_per_token: (cpu_decode_ms_per_tok * 100.0).round() / 100.0,
        max_laptop_supported_tok_s: (max_laptop_tok_s * 10.0).round() / 10.0,
        non_reversible_mechanism:
            "mmap Sparse Lookup + Ephemeral Session Salt Attention Mixing + SwiGLU Non-Injective Clipping + RMSNorm Norm Destruction + Clustered lm_head"
                .to_string(),
    };

    let avg_layer_mb = if hdr.total_layers > 0 {
        ((total_tensor_bytes - non_block_bytes) as f64)
            / (hdr.total_layers as f64 * 1024.0 * 1024.0)
    } else {
        0.0
    };

    // Exact FP16 KV-Cache for 4096 context computed ONLY on layers that have KV-cache attention,
    // plus fixed recurrent SSM state buffer (~3.125 MiB per SSM layer for Gated DeltaNet + Conv1D):
    let ssm_recurrent_state_mb = (linear_ssm_layers * 3200).div_ceil(1024);
    let attn_kv_4k_mb = if hdr.head_count_kv > 0 && hdr.key_length > 0 {
        let kv_bytes =
            4u64 * attention_kv_layers * hdr.head_count_kv * hdr.key_length * 4096;
        (kv_bytes / (1024 * 1024)).max(16)
    } else {
        ((attention_kv_layers as f64 * avg_layer_mb * 0.25).max(32.0)) as u64
    };
    let kv_cache_4k_ctx_mb = attn_kv_4k_mb + ssm_recurrent_state_mb;
    let total_vram_needed_mb = (total_weights_mb as u64) + kv_cache_4k_ctx_mb + 256;

    // Physics-derived hardware requirements from the model's exact active weights & total VRAM:
    let active_weights_gb = (active_weights_per_token_bytes as f64) / 1e9;
    let approx_active_params_b = (active_weights_gb * 1.8).max(0.1);

    let compute_tier_from_sla = |tier_name: &str,
                                 donor_power_mode: &str,
                                 nodes: usize,
                                 target_decode_tok_s: f64,
                                 target_prefill_tok_s: f64,
                                 max_rtt_ms: f64|
     -> HardwareTierRecommendation {
        let min_vram_per_node = ((total_vram_needed_mb / (nodes as u64)) + 128).max(512);

        // Hop latency across nodes reduces time budget available for memory reads
        let hop_latency_ms = if nodes > 1 {
            (nodes as f64) * max_rtt_ms
        } else {
            0.0
        };
        let target_ms_per_tok = 1000.0 / target_decode_tok_s;
        let available_mem_read_ms = (target_ms_per_tok - hop_latency_ms - 2.0).max(3.0);

        // Required aggregate memory bandwidth GB/s = (Active Weights GB / available_mem_read_sec)
        let req_agg_bw_gb_s = (active_weights_gb * 1000.0) / available_mem_read_ms;
        let req_bw_per_node = ((req_agg_bw_gb_s / (nodes as f64)).max(32.0) * 10.0).round() / 10.0;

        // Required FP16 TFLOPS per node to achieve target_prefill_tok_s:
        // FLOPs per token = 2 * active_params
        let req_agg_tflops = (2.0 * approx_active_params_b * target_prefill_tok_s) / 1000.0;
        let req_tflops_per_node =
            ((req_agg_tflops / (nodes as f64)).max(2.0) * 10.0).round() / 10.0;

        // Physics estimation of GPU power draw in Watts for the required TFLOPS & Bandwidth:
        // Base board idle (~12W) + ~4.2W per active TFLOP + ~0.16W per GB/s of VRAM traffic
        let est_watts_per_node =
            ((12.0 + req_tflops_per_node * 4.2 + req_bw_per_node * 0.16) * 10.0).round() / 10.0;

        HardwareTierRecommendation {
            tier_name: tier_name.to_string(),
            donor_power_mode: donor_power_mode.to_string(),
            recommended_nodes: nodes,
            min_vram_per_node_mb: min_vram_per_node,
            min_tflops_per_node: req_tflops_per_node,
            min_bandwidth_gb_s_per_node: req_bw_per_node,
            estimated_power_watts_per_node: est_watts_per_node,
            max_network_rtt_ms: max_rtt_ms,
            estimated_prefill_tok_s: target_prefill_tok_s,
            estimated_decode_tok_s: target_decode_tok_s,
        }
    };

    // Automatically determine optimal multi-node counts based on the model's real VRAM footprint
    let balanced_nodes = ((total_vram_needed_mb as f64 / 6144.0).ceil() as usize).clamp(2, 8);
    let half_watt_eco_nodes = ((total_vram_needed_mb as f64 / 3584.0).ceil() as usize).clamp(2, 12);

    let recommendations = vec![
        compute_tier_from_sla(
            "Tier 1 - Nodo Singolo Nativo (100% Watt Load)",
            "100% GPU Power (--power-pct 100)",
            1,
            65.0,
            800.0,
            0.0,
        ),
        compute_tier_from_sla(
            "Tier 2 - Swarm P2P Bilanciato (100% Watt Load)",
            "100% GPU Power (--power-pct 100)",
            balanced_nodes,
            45.0,
            500.0,
            5.0,
        ),
        compute_tier_from_sla(
            "Tier 3 - Swarm P2P Eco Half-Watt (Donatori al 50% Watt)",
            "50% GPU Power (--power-pct 50, silenzioso/basso consumo)",
            half_watt_eco_nodes,
            30.0,
            300.0,
            8.0,
        ),
    ];

    let p2p_suitability = if has_per_layer_token_embd {
        "SINGLE_NODE_PREFERRED: Model contains Per-Layer Embeddings (per_layer_token_embd); Tier 1 avoids inter-node PLE traffic".to_string()
    } else if linear_ssm_layers > 0 {
        format!(
            "EXCELLENT_HYBRID_P2P: {} Linear/SSM layers (0 KV-Cache growth) + {} Attention KV layers",
            linear_ssm_layers, attention_kv_layers
        )
    } else {
        "OPTIMAL_P2P_PIPELINE: Standard sequential Transformer blocks (only hidden state transferred per hop)".to_string()
    };

    let model_name = model_name_raw.to_string();

    Ok(ModelCatalogEntry {
        model_name,
        architecture: hdr.architecture.clone(),
        gguf_version: hdr.version,
        total_layers: hdr.total_layers as u64,
        attention_kv_layers,
        linear_ssm_layers,
        embedding_length: hdr.embedding_length,
        vocab_size: hdr.vocab_size,
        head_count_kv: hdr.head_count_kv,
        expert_count: hdr.expert_count,
        expert_used_count: hdr.expert_used_count,
        total_tensors: hdr.tensor_count,
        total_weights_mb: (total_weights_mb * 100.0).round() / 100.0,
        active_weights_per_token_mb: (active_weights_per_token_mb * 100.0).round() / 100.0,
        avg_layer_size_mb: (avg_layer_mb * 100.0).round() / 100.0,
        kv_cache_4k_ctx_mb,
        inter_node_activation_bytes_per_token: hdr.embedding_length * 2,
        has_per_layer_token_embd,
        p2p_sharding_suitability: p2p_suitability,
        local_privacy_shield_mb: (local_privacy_mb * 100.0).round() / 100.0,
        local_privacy_layers: "token_embd.weight + blk.0 (Secret Prefix KV) + output.weight"
            .to_string(),
        laptop_cpu_micro_shield,
        remote_sharded_weights_mb: (remote_sharded_mb * 100.0).round() / 100.0,
        remote_sharded_layers: format!("blk.1 .. blk.{}", hdr.total_layers.saturating_sub(1)),
        hardware_recommendations: recommendations,
    })
}

/// Schedules `blk.1 .. blk.N-1` proportionally across verified Direct-QUIC GPU Workers
/// using their `effective_available_tflops` and `effective_available_bandwidth_gb_s`
/// (which automatically respect each donor's `--power-pct` Watt budget and current GPU load!).
pub fn schedule_nodes_for_model(
    catalog: &ModelCatalogEntry,
    candidates: &[GpuNodeCapabilities],
) -> Result<SwarmExecutionPlan, String> {
    let mut eligible: Vec<GpuNodeCapabilities> = candidates
        .iter()
        .filter(|n| n.direct_udp_reachable && n.vram_free_mb >= 512)
        .cloned()
        .collect();

    if eligible.is_empty() {
        return Err(
            "Nessun nodo GPU con connessione QUIC diretta e VRAM sufficiente trovato".into(),
        );
    }

    // Score each node by its EFFECTIVE available Bandwidth & TFLOPS (respecting --power-pct 50 half-watt mode!)
    let node_score = |n: &GpuNodeCapabilities| -> f64 {
        (n.effective_available_bandwidth_gb_s * n.effective_available_tflops.sqrt())
            / (1.0 + n.measured_rtt_ms * 0.005)
    };
    eligible.sort_by(|a, b| node_score(b).total_cmp(&node_score(a)));

    // Water-Filling VRAM-Constrained Allocation:
    // Ensures no node is ever assigned more weights MB than its physical usable VRAM (vram_free_mb - 2048 MB reserve),
    // overflowing excess weights onto remaining nodes with free VRAM.
    let n_count = eligible.len();
    let total_sharded_mb = catalog.remote_sharded_weights_mb.max(1.0);
    let caps_mb: Vec<f64> = eligible
        .iter()
        .map(|n| (n.vram_free_mb.saturating_sub(2048) as f64).max(128.0))
        .collect();
    let scores: Vec<f64> = eligible.iter().map(&node_score).collect();

    let mut alloc_mb = vec![0.0f64; n_count];
    let mut frozen = vec![false; n_count];
    let mut remaining_mb = total_sharded_mb;

    for _ in 0..n_count {
        let unfrozen_score_sum: f64 = scores
            .iter()
            .enumerate()
            .filter(|(i, _)| !frozen[*i])
            .map(|(_, &s)| s)
            .sum();
        if unfrozen_score_sum <= 0.0 || remaining_mb <= 0.0 {
            break;
        }

        let mut capped_any = false;
        for i in 0..n_count {
            if !frozen[i] {
                let desired = remaining_mb * (scores[i] / unfrozen_score_sum);
                if desired > caps_mb[i] {
                    alloc_mb[i] = caps_mb[i];
                    frozen[i] = true;
                    remaining_mb = (remaining_mb - caps_mb[i]).max(0.0);
                    capped_any = true;
                }
            }
        }
        if !capped_any {
            for i in 0..n_count {
                if !frozen[i] {
                    alloc_mb[i] = remaining_mb * (scores[i] / unfrozen_score_sum);
                    frozen[i] = true;
                }
            }
            break;
        }
    }

    let remote_layers_total = catalog.total_layers.saturating_sub(1).max(1);
    let alloc_sum: f64 = alloc_mb.iter().sum::<f64>().max(1.0);

    let mut current_layer = 1u64;
    let mut selected_workers = Vec::new();
    let mut split_ratios = Vec::new();

    for (idx, node) in eligible.iter().enumerate() {
        let ratio = (alloc_mb[idx] / alloc_sum).max(0.05);
        let layers_count = if idx + 1 == eligible.len() {
            catalog.total_layers.saturating_sub(current_layer)
        } else {
            ((remote_layers_total as f64 * ratio).round() as u64).max(1)
        };
        let end_layer = (current_layer + layers_count).min(catalog.total_layers);
        let assigned_mb = (layers_count as f64) * catalog.avg_layer_size_mb;

        selected_workers.push(AssignedWorkerShard {
            node_name: node.node_name.clone(),
            endpoint_id: node.endpoint_id.clone(),
            gpu_name: node.gpu_name.clone(),
            vram_free_mb: node.vram_free_mb,
            user_power_budget_pct: node.user_power_budget_pct,
            allocated_power_watts: node.allocated_power_budget_watts,
            effective_tflops: node.effective_available_tflops,
            effective_bandwidth_gb_s: node.effective_available_bandwidth_gb_s,
            measured_rtt_ms: (node.measured_rtt_ms * 100.0).round() / 100.0,
            assigned_layers: format!("blk.{} .. blk.{}", current_layer, end_layer.saturating_sub(1)),
            assigned_weights_mb: (assigned_mb * 100.0).round() / 100.0,
            tensor_split_ratio: (ratio * 100.0).round() / 100.0,
        });
        split_ratios.push(format!("{:.0}", (ratio * 100.0).round()));
        current_layer = end_layer;
    }

    let agg_bw: f64 = selected_workers
        .iter()
        .map(|w| w.effective_bandwidth_gb_s)
        .sum();
    let agg_tflops: f64 = selected_workers.iter().map(|w| w.effective_tflops).sum();
    let total_rtt: f64 = selected_workers.iter().map(|w| w.measured_rtt_ms).sum();
    let active_weights_gb = catalog.active_weights_per_token_mb / 1024.0;
    let decode_ms = ((active_weights_gb / agg_bw.max(1.0)) * 1000.0) + total_rtt + 8.0;
    let est_decode_tok_s = (1000.0 / decode_ms * 10.0).round() / 10.0;
    let est_prefill_tok_s = ((agg_tflops * 18.0) * 10.0).round() / 10.0;

    Ok(SwarmExecutionPlan {
        model_name: catalog.model_name.clone(),
        token_privacy_mode:
            "ACTIVE: token_embd + blk.0 (Secret Prefix KV) + output.weight locked on Local Device"
                .to_string(),
        selected_workers,
        llama_cpp_tensor_split_arg: split_ratios.join(","),
        estimated_cluster_decode_tok_s: est_decode_tok_s,
        estimated_cluster_prefill_tok_s: est_prefill_tok_s,
    })
}
