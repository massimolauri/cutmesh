package io.cutmesh.controlplane.fleet;

import com.fasterxml.jackson.annotation.JsonProperty;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.Instant;

public final class WorkerAndEnrollmentDtos {

    private WorkerAndEnrollmentDtos() {
    }

    public record CreateEnrollmentTokenRequest(
        @NotBlank String label,
        @NotNull PoolMode defaultPoolMode
    ) {
    }

    public record EnrollmentTokenIssuedResponse(
        String id,
        String label,
        String rawTokenSecret,
        String tokenPrefix,
        PoolMode defaultPoolMode,
        String dockerRunLinuxCommand,
        String dockerRunWsl2Command,
        String nativeRustCliCommand,
        Instant createdAt
    ) {
    }

    /**
     * Exact wire format sent by Rust `send_heartbeat_to_control_plane` in `src/discovery.rs`.
     */
    public record RustWorkerHeartbeatRequest(
        @NotBlank @JsonProperty("iroh_node_id") String irohNodeId,
        @NotBlank @JsonProperty("iroh_ticket") String irohTicket,
        @NotBlank @JsonProperty("gpu_name") String gpuName,
        @Min(1) @JsonProperty("free_vram_mb") long freeVramMb,
        @Min(1) @JsonProperty("total_vram_mb") long totalVramMb,
        @Min(1) @JsonProperty("max_allocatable_vram_mb") long maxAllocatableVramMb,
        @NotBlank @JsonProperty("pool_mode") String poolMode,
        @JsonProperty("rpc_healthy") boolean rpcHealthy,
        @JsonProperty("watchdog_restarts") long watchdogRestarts,
        @NotBlank @JsonProperty("version") String version,
        @JsonProperty("node_name") String nodeName,
        @JsonProperty("local_rpc_port") Integer localRpcPort,
        @JsonProperty("device") String device,
        @JsonProperty("effective_tflops") Double effectiveTflops,
        @JsonProperty("effective_bandwidth_gb_s") Double effectiveBandwidthGbS,
        @JsonProperty("allocated_power_watts") Double allocatedPowerWatts,
        @JsonProperty("power_budget_pct") Integer powerBudgetPct,
        @JsonProperty("temperature_c") Integer temperatureC,
        @JsonProperty("gpu_utilization_pct") Integer gpuUtilizationPct
    ) {
    }

    /**
     * Exact wire format expected back by Rust `ControlPlaneHeartbeatResponse` in `src/discovery.rs`.
     */
    public record RustWorkerHeartbeatResponse(
        @JsonProperty("accepted") boolean accepted,
        @JsonProperty("assignment_state") String assignmentState,
        @JsonProperty("assigned_rig_id") String assignedRigId,
        @JsonProperty("assigned_model_id") String assignedModelId,
        @JsonProperty("assigned_role") String assignedRole,
        @JsonProperty("assigned_model_url") String assignedModelUrl,
        @JsonProperty("assigned_layer_start") Integer assignedLayerStart,
        @JsonProperty("assigned_layer_end") Integer assignedLayerEnd,
        @JsonProperty("next_heartbeat_secs") long nextHeartbeatSecs
    ) {
    }
}
