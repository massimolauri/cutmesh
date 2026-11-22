#!/usr/bin/env python3
"""
Clean architectural patches for llama.cpp's ggml-rpc (transport.cpp + ggml-rpc.cpp):

1. Userspace 256 KiB TX Coalescing Buffer (`socket_t::impl::tx_buf`) in `transport.cpp`
   + batch-flush in `ggml-rpc.cpp` (`send_rpc_cmd` / `rpc_dispatcher::work`):
   - Eliminates the 3 separate `send()` syscalls per RPC command (1B cmd + 8B size + NB payload)
     and coalesces all `SET_TENSOR` + `GRAPH_RECOMPUTE` + `GET_TENSOR` commands of a split
     into a single TCP/QUIC write, and coalesces the server's 8B size + NB tensor response
     into a single TCP/QUIC write.
   - `recv_data()` automatically flushes `tx_buf` before reading, guaranteeing zero deadlock risk.

2. Asynchronous `SET_TENSOR` / `MEMSET_TENSOR` / `INIT_TENSOR` Dispatch in `ggml-rpc.cpp`:
   - `serialize_set_tensor` already deep-copies tensor data into a heap-allocated `std::shared_ptr<uint8_t>`
     synchronously on the caller thread. Using `dispatcher->send_async` instead of blocking `send`
     for activation copies (`cache_flag == 0 && size <= 4 MiB`) prevents 7 sequential main-thread
     context-switch stalls per token.

3. Content-Addressed Graph Caching (`fnv_hash` of serialized graph topology) in `ggml-rpc.cpp`:
   - Upstream `llama.cpp` assigns a new sequential `cgraph->uid` whenever switching between
     Prefill (`n_tokens > 1`) and Decode (`n_tokens = 1`), causing `device_graph_uids` to miss
     and re-transmit 198 KB (`RPC1`) + 1.4 MB (`RPC0`) of identical graph topology on every turn.
   - Hashing the serialized graph payload (after the 12-byte `device + uid` header) maps identical
     topologies to a stable 64-bit `content_uid`, turning cross-turn Decode/Prefill graph switches
     into 12-byte `RPC_CMD_GRAPH_RECOMPUTE` hits.

4. CUDA-Exact `get_alloc_size` Short-Circuit + `HASH_THRESHOLD = 65536` in `ggml-rpc.cpp`:
   - On CUDA backends, `get_alloc_size` only exceeds `ggml_nbytes(tensor)` for non-512-aligned
     quantized tensors (computable locally via `ggml_row_size`) or `GGML_OP_FLASH_ATTN_EXT` with
     non-F16 KV caches. Computing this locally eliminates 8 synchronous WAN round-trips (~640 ms)
     on every new prompt token length.
"""

import pathlib
import sys


def patch_transport_cpp(path: pathlib.Path) -> None:
    src = path.read_text(encoding="utf-8")

    if "TX_COAL_CAP" in src:
        print(f"[patch_ggml_rpc] {path} already patched, skipping.")
        return

    # 1. Include <vector> and <algorithm> in transport.cpp
    target_inc = "#include <mutex>\n#include <optional>"
    replacement_inc = "#include <algorithm>\n#include <mutex>\n#include <optional>\n#include <vector>"
    if target_inc not in src:
        raise RuntimeError(f"Could not find include block in {path}")
    src = src.replace(target_inc, replacement_inc, 1)

    # 2. Add tx_buf and send_raw to struct socket_t::impl
    target_struct = """struct socket_t::impl {
    impl(sockfd_t fd) : use_rdma(false), fd(fd) {}
    ~impl();
    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    bool flush();"""

    replacement_struct = """struct socket_t::impl {
    static constexpr size_t TX_COAL_CAP = 256 * 1024;
    std::vector<uint8_t> tx_buf;
    impl(sockfd_t fd) : use_rdma(false), fd(fd) {
        tx_buf.reserve(TX_COAL_CAP);
    }
    ~impl();
    bool send_raw(const void * data, size_t size);
    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    bool flush();"""

    if target_struct not in src:
        raise RuntimeError(f"Could not find socket_t::impl struct header in {path}")
    src = src.replace(target_struct, replacement_struct, 1)

    # 3. Replace send_data, recv_data, and flush in transport.cpp
    target_send_recv = """bool socket_t::impl::send_data(const void * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma->send(data, size);
    }
#elif defined(GGML_RPC_RDMA)
    if (use_rdma) {
        return rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (const char *)data + bytes_sent, size_to_send, 0);
        if (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\\n",
                           bytes_sent, size_to_send);
            return false;
        }
        bytes_sent += (size_t)n;
    }
    return true;
}

bool socket_t::impl::recv_data(void * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma->recv(data, size);
    }
#elif defined(GGML_RPC_RDMA)
    if (use_rdma) {
        return rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;"""

    replacement_send_recv = """bool socket_t::impl::send_raw(const void * data, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (const char *)data + bytes_sent, size_to_send, 0);
        if (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\\n",
                           bytes_sent, size_to_send);
            return false;
        }
        bytes_sent += (size_t)n;
    }
    return true;
}

bool socket_t::impl::send_data(const void * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma->send(data, size);
    }
#elif defined(GGML_RPC_RDMA)
    if (use_rdma) {
        return rdma_send(data, size);
    }
#endif
    if (size == 0) {
        return true;
    }
    const uint8_t * ptr = static_cast<const uint8_t *>(data);
    if (tx_buf.size() + size <= TX_COAL_CAP) {
        tx_buf.insert(tx_buf.end(), ptr, ptr + size);
        return true;
    }
    if (!tx_buf.empty()) {
        if (!send_raw(tx_buf.data(), tx_buf.size())) {
            tx_buf.clear();
            return false;
        }
        tx_buf.clear();
    }
    if (size <= TX_COAL_CAP) {
        tx_buf.insert(tx_buf.end(), ptr, ptr + size);
        return true;
    }
    return send_raw(data, size);
}

bool socket_t::impl::recv_data(void * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma->recv(data, size);
    }
#elif defined(GGML_RPC_RDMA)
    if (use_rdma) {
        return rdma_recv(data, size);
    }
#endif
    if (!tx_buf.empty() && !flush()) {
        return false;
    }
    size_t bytes_recv = 0;"""

    if target_send_recv not in src:
        raise RuntimeError(f"Could not find send_data/recv_data in {path}")
    src = src.replace(target_send_recv, replacement_send_recv, 1)

    target_flush = """bool socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma->flush();
    }
#endif
    return true;
}"""

    replacement_flush = """bool socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma->flush();
    }
#endif
    if (!tx_buf.empty()) {
        bool ok = send_raw(tx_buf.data(), tx_buf.size());
        tx_buf.clear();
        return ok;
    }
    return true;
}"""

    if target_flush not in src:
        raise RuntimeError(f"Could not find socket_t::impl::flush in {path}")
    src = src.replace(target_flush, replacement_flush, 1)

    path.write_text(src, encoding="utf-8")
    print(f"[patch_ggml_rpc] Successfully patched {path}")


def patch_ggml_rpc_cpp(path: pathlib.Path) -> None:
    src = path.read_text(encoding="utf-8")

    if "seq_to_content_uid" in src:
        print(f"[patch_ggml_rpc] {path} already patched, skipping.")
        return

    # 0. Ensure HASH_THRESHOLD = 65536
    src = src.replace(
        "const size_t HASH_THRESHOLD = 10 * 1024 * 1024;",
        "const size_t HASH_THRESHOLD = 65536;",
    )
    if "const size_t HASH_THRESHOLD = 65536;" not in src:
        raise RuntimeError(f"HASH_THRESHOLD = 65536 not found in {path}")

    # 1. Coalesce send_rpc_cmd_payload into tx_buf (flushed by recv_data or dispatcher work loop)
    target_send_cmd = """static bool send_rpc_cmd_payload(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, const void * payload, size_t payload_size) {
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    const size_t request_size = input_size + payload_size;
    if (!sock->send_data(&request_size, sizeof(request_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    if (payload_size > 0 && !sock->send_data(payload, payload_size)) {
        return false;
    }
    return sock->flush();
}"""

    replacement_send_cmd = """static bool send_rpc_cmd_payload(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, const void * payload, size_t payload_size) {
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    const size_t request_size = input_size + payload_size;
    if (!sock->send_data(&request_size, sizeof(request_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    if (payload_size > 0 && !sock->send_data(payload, payload_size)) {
        return false;
    }
    return true;
}"""

    if target_send_cmd not in src:
        raise RuntimeError(f"Could not find send_rpc_cmd_payload in {path}")
    src = src.replace(target_send_cmd, replacement_send_cmd, 1)

    # 2. Add seq_to_content_uid to rpc_dispatcher and batch-flushing to rpc_dispatcher::work()
    target_disp_fields = """    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;"""

    replacement_disp_fields = """    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    std::unordered_map<uint32_t, std::unordered_map<uint64_t, uint64_t>> seq_to_content_uid;"""

    if target_disp_fields not in src:
        raise RuntimeError(f"Could not find graph_uids in {path}")
    src = src.replace(target_disp_fields, replacement_disp_fields, 1)

    target_work = """void rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    while (running) {
        rpc_msg_ptr msg_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        const bool spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        if (spin) {
            if (!queue.try_pop(&msg_ptr)) {
                rpc_cpu_relax();
                continue;
            }
        } else if (!queue.pop(&msg_ptr)) {
            break;
        }
        if (msg_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\\n", __func__, rpc_cmd_name(msg_ptr->cmd),
                     rpc_log_format_size(msg_ptr->input_size).c_str(), rpc_log_format_size(msg_ptr->output_size).c_str());
            const bool trace = rpc_log_debug_level() >= 3;
            const auto t0 = std::chrono::steady_clock::now();
            if (msg_ptr->output) {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size, msg_ptr->output, msg_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } else {
                bool status = send_rpc_cmd_payload(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size, msg_ptr->payload, msg_ptr->payload_size);
                RPC_STATUS_ASSERT(status);
            }
            if (trace) {
                const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\\n", __func__, rpc_cmd_name(msg_ptr->cmd), elapsed.count());
            }
        } else {
            LOG_DBG3("[%s] sync barrier\\n", __func__);
        }
        msg_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}"""

    replacement_work = """void rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    bool has_unflushed_tx = false;
    while (running) {
        rpc_msg_ptr msg_ptr;
        if (has_unflushed_tx) {
            if (!queue.try_pop(&msg_ptr)) {
                if (std::chrono::steady_clock::now() - last_cmd >= std::chrono::microseconds(250)) {
                    RPC_STATUS_ASSERT(sock->flush());
                    has_unflushed_tx = false;
                } else {
                    rpc_cpu_relax();
                }
                continue;
            }
        } else {
            // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
            const bool spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                              std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
            if (spin) {
                if (!queue.try_pop(&msg_ptr)) {
                    rpc_cpu_relax();
                    continue;
                }
            } else if (!queue.pop(&msg_ptr)) {
                break;
            }
        }
        if (msg_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\\n", __func__, rpc_cmd_name(msg_ptr->cmd),
                     rpc_log_format_size(msg_ptr->input_size).c_str(), rpc_log_format_size(msg_ptr->output_size).c_str());
            const bool trace = rpc_log_debug_level() >= 3;
            const auto t0 = std::chrono::steady_clock::now();
            if (msg_ptr->output) {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size, msg_ptr->output, msg_ptr->output_size);
                RPC_STATUS_ASSERT(status);
                has_unflushed_tx = false;
            } else {
                bool status = send_rpc_cmd_payload(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size, msg_ptr->payload, msg_ptr->payload_size);
                RPC_STATUS_ASSERT(status);
                has_unflushed_tx = true;
            }
            if (trace) {
                const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\\n", __func__, rpc_cmd_name(msg_ptr->cmd), elapsed.count());
            }
        } else {
            LOG_DBG3("[%s] sync barrier\\n", __func__);
        }
        msg_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
    if (sock && has_unflushed_tx) {
        sock->flush();
    }
}"""

    if target_work not in src:
        raise RuntimeError(f"Could not find rpc_dispatcher::work in {path}")
    src = src.replace(target_work, replacement_work, 1)

    # 3. Async enqueueing for init_tensor, memset_tensor, and activation set_tensor
    target_set_tensor = """        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, sizeof(*request), nullptr, 0, nullptr, 0);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, sizeof(*request), nullptr, 0, nullptr, 0);
}"""

    replacement_set_tensor = """        ctx->dispatcher->send_async(RPC_CMD_INIT_TENSOR, request, sizeof(*request));
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send_async(RPC_CMD_MEMSET_TENSOR, request, sizeof(*request));
}"""

    if target_set_tensor not in src:
        raise RuntimeError(f"Could not find init_tensor/memset_tensor in {path}")
    src = src.replace(target_set_tensor, replacement_set_tensor, 1)

    target_buf_set = """    size_t header_size;
    auto header = serialize_set_tensor(rpc_tensor, cache_flag, offset, nullptr, 0, header_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, header, header_size, data, size, nullptr, 0);
}"""

    replacement_buf_set = """    if (cache_flag == 0 && size <= 4 * 1024 * 1024) {
        size_t input_size;
        auto input = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, input_size);
        ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, input, input_size);
    } else {
        size_t header_size;
        auto header = serialize_set_tensor(rpc_tensor, cache_flag, offset, nullptr, 0, header_size);
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR, header, header_size, data, size, nullptr, 0);
    }
}"""

    if target_buf_set not in src:
        raise RuntimeError(f"Could not find ggml_backend_rpc_buffer_set_tensor tail in {path}")
    src = src.replace(target_buf_set, replacement_buf_set, 1)

    # 4. CUDA-exact get_alloc_size short-circuit in ggml_backend_rpc_buffer_type_get_alloc_size
    target_alloc_size = """static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);"""

    replacement_alloc_size = """static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // Exact local formula for CUDA 512-element quantized row padding (zero network round-trips)
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        return ggml_nbytes(tensor) + ggml_row_size(tensor->type, 512 - tensor->ne[0] % 512);
    }

    // On CUDA backends, only FLASH_ATTN_EXT with non-F16 K/V caches ever expands alloc_size beyond ggml_nbytes(tensor)
    bool rpc_get = false;
    static const bool full_query = std::getenv("GGML_RPC_FULL_ALLOC_SIZE_QUERY") != nullptr;
    if (full_query) {
        rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);
    } else if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
        const ggml_tensor * k = tensor->src[1];
        const ggml_tensor * v = tensor->src[2];
        rpc_get = (k && k->type != GGML_TYPE_F16) || (v && v->type != GGML_TYPE_F16);
    }"""

    if target_alloc_size not in src:
        raise RuntimeError(f"Could not find ggml_backend_rpc_buffer_type_get_alloc_size in {path}")
    src = src.replace(target_alloc_size, replacement_alloc_size, 1)

    # 5. Content-addressed graph caching in rpc_dispatcher::graph_compute
    target_graph_compute = """void rpc_dispatcher::graph_compute(uint32_t device, const ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    bool reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    if (reuse) {
        auto request = std::make_shared<rpc_msg_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, sizeof(*request));
    } else {
        if (cgraph->uid != 0) {
            if (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t input_size = 0;
        uint8_t * input = serialize_graph(device, cgraph, this, &input_size);
        LOG_DBG2("[%s] serialized graph: %s\\n", __func__, rpc_log_format_size(input_size).c_str());
        std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, input_ptr, input_size);
    }
}"""

    replacement_graph_compute = """void rpc_dispatcher::graph_compute(uint32_t device, const ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    auto & seq_map = seq_to_content_uid[device];

    if (cgraph->uid != 0) {
        auto it = seq_map.find(cgraph->uid);
        if (it != seq_map.end() && device_graph_uids.count(it->second) > 0) {
            auto request = std::make_shared<rpc_msg_graph_recompute_req>();
            request->device = device;
            request->uid    = it->second;
            send_async(RPC_CMD_GRAPH_RECOMPUTE, request, sizeof(*request));
            return;
        }
    }

    size_t input_size = 0;
    uint8_t * input = serialize_graph(device, cgraph, this, &input_size);
    const size_t hdr_bytes = sizeof(uint32_t) + sizeof(uint64_t);

    if (cgraph->uid != 0 && input_size > hdr_bytes) {
        uint64_t content_uid = fnv_hash(input + hdr_bytes, input_size - hdr_bytes);
        if (content_uid == 0) {
            content_uid = 1;
        }
        seq_map[cgraph->uid] = content_uid;

        if (device_graph_uids.count(content_uid) > 0) {
            delete[] input;
            LOG_DBG("[%s] device: %u, seq_uid: %" PRIu64 " -> content_uid: 0x%" PRIx64 " (cross-turn cache hit)\\n",
                    __func__, device, cgraph->uid, content_uid);
            auto request = std::make_shared<rpc_msg_graph_recompute_req>();
            request->device = device;
            request->uid    = content_uid;
            send_async(RPC_CMD_GRAPH_RECOMPUTE, request, sizeof(*request));
            return;
        }

        if (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
            LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\\n",
                    __func__, device, device_graph_uids.size());
            device_graph_uids.clear();
            seq_map.clear();
            seq_map[cgraph->uid] = content_uid;
        }
        device_graph_uids.insert(content_uid);
        memcpy(input + sizeof(uint32_t), &content_uid, sizeof(content_uid));
    }

    LOG_DBG2("[%s] serialized graph: %s\\n", __func__, rpc_log_format_size(input_size).c_str());
    std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
    send_async(RPC_CMD_GRAPH_COMPUTE, input_ptr, input_size);
}"""

    if target_graph_compute not in src:
        raise RuntimeError(f"Could not find rpc_dispatcher::graph_compute in {path}")
    src = src.replace(target_graph_compute, replacement_graph_compute, 1)

    path.write_text(src, encoding="utf-8")
    print(f"[patch_ggml_rpc] Successfully patched {path}")


def patch_llama_model_and_context(root: pathlib.Path) -> None:
    model_cpp = root / "src" / "llama-model.cpp"
    if model_cpp.is_file():
        src = model_cpp.read_text(encoding="utf-8")
        if "MTP/output layer" not in src:
            target = """        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};"""

            replacement = """        if (params.tensor_buft_overrides) {
            const std::string probe_name = (il < n_layer_all)
                ? ("blk." + std::to_string(il) + ".attn_norm.weight")
                : "output.weight";
            for (const auto * overrides = params.tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
                std::regex pattern(overrides->pattern);
                if (std::regex_search(probe_name, pattern) && overrides->buft) {
                    ggml_backend_dev_t ov_dev = ggml_backend_buft_get_device(overrides->buft);
                    if (ov_dev && pimpl->gpu_buft_list.find(ov_dev) != pimpl->gpu_buft_list.end()) {
                        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to overridden device %s, is_swa = %d\\n", il, ggml_backend_dev_name(ov_dev), is_swa);
                        return {ov_dev, &pimpl->gpu_buft_list.at(ov_dev)};
                    }
                }
            }
        }
        if (ml.load_mtp && il >= int(hparams.n_layer()) && !devices.empty()) {
            auto * dev0 = devices.at(0).dev;
            LLAMA_LOG_DEBUG("load_tensors: MTP/output layer %3d assigned to primary device %s\\n", il, ggml_backend_dev_name(dev0));
            return {dev0, &pimpl->gpu_buft_list.at(dev0)};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};"""

            if target not in src:
                raise RuntimeError(f"Could not find get_layer_buft_list target in {model_cpp}")
            src = src.replace(target, replacement, 1)
            if "#include <regex>" not in src:
                src = "#include <regex>\n" + src
            model_cpp.write_text(src, encoding="utf-8")
            print(f"[patch_ggml_rpc] Successfully patched {model_cpp}")

    ctx_cpp = root / "src" / "llama-context.cpp"
    if ctx_cpp.is_file():
        csrc = ctx_cpp.read_text(encoding="utf-8")
        if "!model.has_tensor_overrides()" in csrc:
            csrc = csrc.replace(
                "            !model.has_tensor_overrides() &&\n",
                "",
                1,
            ).replace(
                "            cparams.offload_kqv &&\n            !model.has_tensor_overrides();",
                "            cparams.offload_kqv;",
                1,
            )
            ctx_cpp.write_text(csrc, encoding="utf-8")
            print(f"[patch_ggml_rpc] Successfully patched {ctx_cpp}")


def main() -> None:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
    transport_cpp = root / "ggml" / "src" / "ggml-rpc" / "transport.cpp"
    ggml_rpc_cpp = root / "ggml" / "src" / "ggml-rpc" / "ggml-rpc.cpp"

    if not transport_cpp.is_file() or not ggml_rpc_cpp.is_file():
        raise FileNotFoundError(
            f"Expected {transport_cpp} and {ggml_rpc_cpp} to exist under {root}"
        )

    patch_transport_cpp(transport_cpp)
    patch_ggml_rpc_cpp(ggml_rpc_cpp)
    patch_llama_model_and_context(root)


if __name__ == "__main__":
    main()

