package io.cutmesh.controlplane.fleet;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.Instant;
import java.util.UUID;

/**
 * Physical or Dockerized GPU Worker node registered via Iroh P2P heartbeat with full silicon telemetry.
 */
@Entity
@Table(
    name = "gpu_worker_nodes",
    indexes = {
        @Index(name = "idx_worker_iroh_node_id", columnList = "iroh_node_id", unique = true),
        @Index(name = "idx_worker_owner", columnList = "owner_user_id"),
        @Index(name = "idx_worker_state", columnList = "assignment_state")
    }
)
public class GpuWorkerNode {

    @Id
    @Column(name = "id", nullable = false, updatable = false, length = 36)
    private String id;

    @Column(name = "owner_user_id", nullable = false, length = 36)
    private String ownerUserId;

    @Column(name = "enrollment_token_id", length = 36)
    private String enrollmentTokenId;

    @Column(name = "node_name", length = 128)
    private String nodeName = "gpu-worker";

    @Column(name = "iroh_node_id", nullable = false, unique = true, length = 128)
    private String irohNodeId;

    @Column(name = "iroh_ticket", nullable = false, length = 1024)
    private String irohTicket;

    @Column(name = "local_rpc_port")
    private int localRpcPort = 50051;

    @Column(name = "device_name", length = 32)
    private String device = "CUDA0";

    @Column(name = "gpu_name", nullable = false, length = 160)
    private String gpuName;

    @Column(name = "free_vram_mb", nullable = false)
    private long freeVramMb;

    @Column(name = "total_vram_mb", nullable = false)
    private long totalVramMb;

    @Column(name = "max_allocatable_vram_mb", nullable = false)
    private long maxAllocatableVramMb;

    @Column(name = "effective_tflops")
    private double effectiveTflops = 0.0;

    @Column(name = "effective_bandwidth_gb_s")
    private double effectiveBandwidthGbS = 0.0;

    @Column(name = "allocated_power_watts")
    private double allocatedPowerWatts = 0.0;

    @Column(name = "power_budget_pct")
    private int powerBudgetPct = 100;

    @Column(name = "temperature_c")
    private int temperatureC = 45;

    @Column(name = "gpu_utilization_pct")
    private int gpuUtilizationPct = 0;

    @Enumerated(EnumType.STRING)
    @Column(name = "pool_mode", nullable = false, length = 24)
    private PoolMode poolMode = PoolMode.SHARED_POOL;

    @Enumerated(EnumType.STRING)
    @Column(name = "assignment_state", nullable = false, length = 32)
    private WorkerAssignmentState assignmentState = WorkerAssignmentState.IDLE_STANDBY_POOL;

    @Column(name = "assigned_rig_id", length = 36)
    private String assignedRigId;

    @Column(name = "assigned_model_id", length = 64)
    private String assignedModelId;

    @Column(name = "assigned_role", length = 48)
    private String assignedRole;

    @Column(name = "rpc_healthy", nullable = false)
    private boolean rpcHealthy = true;

    @Column(name = "watchdog_restarts", nullable = false)
    private long watchdogRestarts = 0L;

    @Column(name = "version", nullable = false, length = 32)
    private String version = "1.0.0";

    @Column(name = "registered_at", nullable = false, updatable = false)
    private Instant registeredAt;

    @Column(name = "last_heartbeat_at", nullable = false)
    private Instant lastHeartbeatAt;

    public GpuWorkerNode() {
    }

    @PrePersist
    public void prePersist() {
        if (this.id == null) {
            this.id = UUID.randomUUID().toString();
        }
        if (this.registeredAt == null) {
            this.registeredAt = Instant.now();
        }
        if (this.lastHeartbeatAt == null) {
            this.lastHeartbeatAt = Instant.now();
        }
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getOwnerUserId() {
        return ownerUserId;
    }

    public void setOwnerUserId(String ownerUserId) {
        this.ownerUserId = ownerUserId;
    }

    public String getEnrollmentTokenId() {
        return enrollmentTokenId;
    }

    public void setEnrollmentTokenId(String enrollmentTokenId) {
        this.enrollmentTokenId = enrollmentTokenId;
    }

    public String getNodeName() {
        return nodeName;
    }

    public void setNodeName(String nodeName) {
        this.nodeName = nodeName;
    }

    public String getIrohNodeId() {
        return irohNodeId;
    }

    public void setIrohNodeId(String irohNodeId) {
        this.irohNodeId = irohNodeId;
    }

    public String getIrohTicket() {
        return irohTicket;
    }

    public void setIrohTicket(String irohTicket) {
        this.irohTicket = irohTicket;
    }

    public int getLocalRpcPort() {
        return localRpcPort;
    }

    public void setLocalRpcPort(int localRpcPort) {
        this.localRpcPort = localRpcPort;
    }

    public String getDevice() {
        return device;
    }

    public void setDevice(String device) {
        this.device = device;
    }

    public String getGpuName() {
        return gpuName;
    }

    public void setGpuName(String gpuName) {
        this.gpuName = gpuName;
    }

    public long getFreeVramMb() {
        return freeVramMb;
    }

    public void setFreeVramMb(long freeVramMb) {
        this.freeVramMb = freeVramMb;
    }

    public long getTotalVramMb() {
        return totalVramMb;
    }

    public void setTotalVramMb(long totalVramMb) {
        this.totalVramMb = totalVramMb;
    }

    public long getMaxAllocatableVramMb() {
        return maxAllocatableVramMb;
    }

    public void setMaxAllocatableVramMb(long maxAllocatableVramMb) {
        this.maxAllocatableVramMb = maxAllocatableVramMb;
    }

    public double getEffectiveTflops() {
        return effectiveTflops;
    }

    public void setEffectiveTflops(double effectiveTflops) {
        this.effectiveTflops = effectiveTflops;
    }

    public double getEffectiveBandwidthGbS() {
        return effectiveBandwidthGbS;
    }

    public void setEffectiveBandwidthGbS(double effectiveBandwidthGbS) {
        this.effectiveBandwidthGbS = effectiveBandwidthGbS;
    }

    public double getAllocatedPowerWatts() {
        return allocatedPowerWatts;
    }

    public void setAllocatedPowerWatts(double allocatedPowerWatts) {
        this.allocatedPowerWatts = allocatedPowerWatts;
    }

    public int getPowerBudgetPct() {
        return powerBudgetPct;
    }

    public void setPowerBudgetPct(int powerBudgetPct) {
        this.powerBudgetPct = powerBudgetPct;
    }

    public int getTemperatureC() {
        return temperatureC;
    }

    public void setTemperatureC(int temperatureC) {
        this.temperatureC = temperatureC;
    }

    public int getGpuUtilizationPct() {
        return gpuUtilizationPct;
    }

    public void setGpuUtilizationPct(int gpuUtilizationPct) {
        this.gpuUtilizationPct = gpuUtilizationPct;
    }

    public PoolMode getPoolMode() {
        return poolMode;
    }

    public void setPoolMode(PoolMode poolMode) {
        this.poolMode = poolMode;
    }

    public WorkerAssignmentState getAssignmentState() {
        return assignmentState;
    }

    public void setAssignmentState(WorkerAssignmentState assignmentState) {
        this.assignmentState = assignmentState;
    }

    public String getAssignedRigId() {
        return assignedRigId;
    }

    public void setAssignedRigId(String assignedRigId) {
        this.assignedRigId = assignedRigId;
    }

    public String getAssignedModelId() {
        return assignedModelId;
    }

    public void setAssignedModelId(String assignedModelId) {
        this.assignedModelId = assignedModelId;
    }

    public String getAssignedRole() {
        return assignedRole;
    }

    public void setAssignedRole(String assignedRole) {
        this.assignedRole = assignedRole;
    }

    public boolean isRpcHealthy() {
        return rpcHealthy;
    }

    public void setRpcHealthy(boolean rpcHealthy) {
        this.rpcHealthy = rpcHealthy;
    }

    public long getWatchdogRestarts() {
        return watchdogRestarts;
    }

    public void setWatchdogRestarts(long watchdogRestarts) {
        this.watchdogRestarts = watchdogRestarts;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public Instant getRegisteredAt() {
        return registeredAt;
    }

    public Instant getLastHeartbeatAt() {
        return lastHeartbeatAt;
    }

    public void setLastHeartbeatAt(Instant lastHeartbeatAt) {
        this.lastHeartbeatAt = lastHeartbeatAt;
    }
}
