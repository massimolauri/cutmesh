package io.cutmesh.controlplane.orchestration;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.OneToMany;
import jakarta.persistence.OrderBy;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import jakarta.persistence.Transient;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Closed & Locked Swarm Rig serving a specific CatalogModel over proportional Water-Filling VRAM allocations.
 */
@Entity
@Table(
    name = "managed_rigs",
    indexes = {
        @Index(name = "idx_managed_rigs_model_state", columnList = "model_id,lifecycle_state")
    }
)
public class ManagedRig {

    @Id
    @Column(name = "id", nullable = false, updatable = false, length = 36)
    private String id;

    @Column(name = "rig_name", nullable = false, length = 128)
    private String rigName;

    @Column(name = "model_id", nullable = false, length = 64)
    private String modelId;

    @Column(name = "model_display_name", nullable = false, length = 128)
    private String modelDisplayName;

    @Enumerated(EnumType.STRING)
    @Column(name = "lifecycle_state", nullable = false, length = 32)
    private RigLifecycleState lifecycleState = RigLifecycleState.SERVING;

    @Enumerated(EnumType.STRING)
    @Column(name = "orchestration_mode", nullable = false, length = 24)
    private OrchestrationMode orchestrationMode = OrchestrationMode.AUTO_PILOT;

    @Enumerated(EnumType.STRING)
    @Column(name = "security_profile", nullable = false, length = 48)
    private SecurityProfile securityProfile = SecurityProfile.FULL_COVARIANT_SHIELD;

    @Column(name = "total_allocated_vram_mb", nullable = false)
    private long totalAllocatedVramMb;

    @Column(name = "required_model_vram_mb", nullable = false)
    private long requiredModelVramMb;

    @Column(name = "auto_maximized_context_size", nullable = false)
    private int autoMaximizedContextSize;

    @Column(name = "tensor_split_csv", nullable = false, length = 256)
    private String tensorSplitCsv;

    @Column(name = "anti_collusion_verified", nullable = false)
    private boolean antiCollusionVerified = true;

    @Column(name = "served_requests_count", nullable = false)
    private long servedRequestsCount = 0L;

    @Column(name = "inference_endpoint_url", length = 256)
    private String inferenceEndpointUrl = "http://127.0.0.1:8089";

    @Transient
    private int activeInFlightRequests = 0;

    @Transient
    private double avgLatencyMs = 150.0;

    @Transient
    private double lastPrefillTokPerSec = 0.0;

    @Transient
    private double lastDecodeTokPerSec = 0.0;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    @Column(name = "closed_at")
    private Instant closedAt;

    @OneToMany(mappedBy = "rig", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.EAGER)
    @OrderBy("pipelineStageIndex ASC")
    private List<RigMemberAllocation> allocations = new ArrayList<>();

    @Transient
    private boolean inferenceEngineReady = false;

    @Transient
    private boolean llamaProcessAlive = false;

    @Transient
    private String runtimePhase = "LAYERS_ALLOCATED";

    @Transient
    private int runtimeStep = 1;

    @Transient
    private String runtimeLastLog = "";

    @Transient
    private String operationalStatusMessage = "SOLO LAYER PIANIFICATI — Pesi non caricati in VRAM";

    public ManagedRig() {
    }

    @PrePersist
    public void prePersist() {
        if (this.id == null) {
            this.id = UUID.randomUUID().toString();
        }
        if (this.createdAt == null) {
            this.createdAt = Instant.now();
        }
    }

    public void addAllocation(RigMemberAllocation allocation) {
        allocation.setRig(this);
        this.allocations.add(allocation);
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getRigName() {
        return rigName;
    }

    public void setRigName(String rigName) {
        this.rigName = rigName;
    }

    public String getModelId() {
        return modelId;
    }

    public void setModelId(String modelId) {
        this.modelId = modelId;
    }

    public String getModelDisplayName() {
        return modelDisplayName;
    }

    public void setModelDisplayName(String modelDisplayName) {
        this.modelDisplayName = modelDisplayName;
    }

    public RigLifecycleState getLifecycleState() {
        return lifecycleState;
    }

    public void setLifecycleState(RigLifecycleState lifecycleState) {
        this.lifecycleState = lifecycleState;
    }

    public OrchestrationMode getOrchestrationMode() {
        return orchestrationMode;
    }

    public void setOrchestrationMode(OrchestrationMode orchestrationMode) {
        this.orchestrationMode = orchestrationMode;
    }

    public SecurityProfile getSecurityProfile() {
        return securityProfile;
    }

    public void setSecurityProfile(SecurityProfile securityProfile) {
        this.securityProfile = securityProfile;
    }

    public long getTotalAllocatedVramMb() {
        return totalAllocatedVramMb;
    }

    public void setTotalAllocatedVramMb(long totalAllocatedVramMb) {
        this.totalAllocatedVramMb = totalAllocatedVramMb;
    }

    public long getRequiredModelVramMb() {
        return requiredModelVramMb;
    }

    public void setRequiredModelVramMb(long requiredModelVramMb) {
        this.requiredModelVramMb = requiredModelVramMb;
    }

    public int getAutoMaximizedContextSize() {
        return autoMaximizedContextSize;
    }

    public void setAutoMaximizedContextSize(int autoMaximizedContextSize) {
        this.autoMaximizedContextSize = autoMaximizedContextSize;
    }

    public String getTensorSplitCsv() {
        return tensorSplitCsv;
    }

    public void setTensorSplitCsv(String tensorSplitCsv) {
        this.tensorSplitCsv = tensorSplitCsv;
    }

    public boolean isAntiCollusionVerified() {
        return antiCollusionVerified;
    }

    public void setAntiCollusionVerified(boolean antiCollusionVerified) {
        this.antiCollusionVerified = antiCollusionVerified;
    }

    public long getServedRequestsCount() {
        return servedRequestsCount;
    }

    public void setServedRequestsCount(long servedRequestsCount) {
        this.servedRequestsCount = servedRequestsCount;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getClosedAt() {
        return closedAt;
    }

    public void setClosedAt(Instant closedAt) {
        this.closedAt = closedAt;
    }

    public List<RigMemberAllocation> getAllocations() {
        return allocations;
    }

    public boolean isInferenceEngineReady() {
        return inferenceEngineReady;
    }

    public void setInferenceEngineReady(boolean inferenceEngineReady) {
        this.inferenceEngineReady = inferenceEngineReady;
    }

    public String getOperationalStatusMessage() {
        return operationalStatusMessage;
    }

    public void setOperationalStatusMessage(String operationalStatusMessage) {
        this.operationalStatusMessage = operationalStatusMessage;
    }

    public String getInferenceEndpointUrl() {
        return inferenceEndpointUrl != null && !inferenceEndpointUrl.isBlank()
            ? inferenceEndpointUrl
            : "http://127.0.0.1:8089";
    }

    public void setInferenceEndpointUrl(String inferenceEndpointUrl) {
        this.inferenceEndpointUrl = inferenceEndpointUrl;
    }

    public int getActiveInFlightRequests() {
        return activeInFlightRequests;
    }

    public void setActiveInFlightRequests(int activeInFlightRequests) {
        this.activeInFlightRequests = activeInFlightRequests;
    }

    public double getAvgLatencyMs() {
        return avgLatencyMs;
    }

    public void setAvgLatencyMs(double avgLatencyMs) {
        this.avgLatencyMs = avgLatencyMs;
    }

    public double getLastPrefillTokPerSec() {
        return lastPrefillTokPerSec;
    }

    public void setLastPrefillTokPerSec(double lastPrefillTokPerSec) {
        this.lastPrefillTokPerSec = lastPrefillTokPerSec;
    }

    public double getLastDecodeTokPerSec() {
        return lastDecodeTokPerSec;
    }

    public void setLastDecodeTokPerSec(double lastDecodeTokPerSec) {
        this.lastDecodeTokPerSec = lastDecodeTokPerSec;
    }

    public boolean isLlamaProcessAlive() {
        return llamaProcessAlive;
    }

    public void setLlamaProcessAlive(boolean llamaProcessAlive) {
        this.llamaProcessAlive = llamaProcessAlive;
    }

    public String getRuntimePhase() {
        return runtimePhase;
    }

    public void setRuntimePhase(String runtimePhase) {
        this.runtimePhase = runtimePhase;
    }

    public int getRuntimeStep() {
        return runtimeStep;
    }

    public void setRuntimeStep(int runtimeStep) {
        this.runtimeStep = runtimeStep;
    }

    public String getRuntimeLastLog() {
        return runtimeLastLog;
    }

    public void setRuntimeLastLog(String runtimeLastLog) {
        this.runtimeLastLog = runtimeLastLog;
    }
}
