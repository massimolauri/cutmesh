package io.cutmesh.controlplane.orchestration;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.Instant;

/**
 * LLM Model registered in the Cutmesh Catalog with Auto-Pilot SLA, Covariant GGUF state,
 * and portable DB-backed Sharding/Engine/Sampling Policy JSON (`policyJson`).
 */
@Entity
@Table(name = "catalog_models")
public class CatalogModel {

    @Id
    @Column(name = "model_id", nullable = false, updatable = false, length = 64)
    private String modelId;

    @Column(name = "display_name", nullable = false, length = 128)
    private String displayName;

    @Column(name = "huggingface_repo", nullable = false, length = 180)
    private String huggingfaceRepo;

    @Column(name = "base_gguf_path", nullable = false, length = 512)
    private String baseGgufPath;

    @Column(name = "covariant_gguf_path", nullable = false, length = 512)
    private String covariantGgufPath;

    @Column(name = "required_vram_mb", nullable = false)
    private long requiredVramMb;

    @Column(name = "total_layers", nullable = false)
    private int totalLayers;

    @Column(name = "kv_vram_per_1k_tokens_mb", nullable = false)
    private long kvVramPer1kTokensMb;

    @Column(name = "max_context_length", nullable = false, columnDefinition = "INTEGER DEFAULT 262144")
    private int maxContextLength = 262144;

    @Column(name = "policy_json", columnDefinition = "TEXT")
    private String policyJson;

    @Enumerated(EnumType.STRING)
    @Column(name = "security_profile", nullable = false, length = 48)
    private SecurityProfile securityProfile = SecurityProfile.FULL_COVARIANT_SHIELD;

    @Enumerated(EnumType.STRING)
    @Column(name = "orchestration_mode", nullable = false, length = 24)
    private OrchestrationMode orchestrationMode = OrchestrationMode.AUTO_PILOT;

    @Column(name = "min_active_rigs", nullable = false)
    private int minActiveRigs = 1;

    @Column(name = "max_active_rigs", nullable = false)
    private int maxActiveRigs = 4;

    @Column(name = "priority_order", nullable = false)
    private int priorityOrder = 100;

    @Column(name = "covariant_compiled", nullable = false)
    private boolean covariantCompiled = false;

    @Column(name = "last_compile_ms")
    private Double lastCompileMs;

    @Column(name = "last_compile_throughput_mb_s")
    private Double lastCompileThroughputMbPerSec;

    @Column(name = "transformed_q5_0_rows")
    private Long transformedQ50Rows;

    @Column(name = "calibrated_rmsnorm_tensors")
    private Integer calibratedRmsnormTensors;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    public CatalogModel() {
    }

    @PrePersist
    public void prePersist() {
        if (this.createdAt == null) {
            this.createdAt = Instant.now();
        }
        if (this.maxContextLength <= 0) {
            this.maxContextLength = 262144;
        }
    }

    public String getModelId() {
        return modelId;
    }

    public void setModelId(String modelId) {
        this.modelId = modelId;
    }

    public String getDisplayName() {
        return displayName;
    }

    public void setDisplayName(String displayName) {
        this.displayName = displayName;
    }

    public String getHuggingfaceRepo() {
        return huggingfaceRepo;
    }

    public void setHuggingfaceRepo(String huggingfaceRepo) {
        this.huggingfaceRepo = huggingfaceRepo;
    }

    public String getBaseGgufPath() {
        return baseGgufPath;
    }

    public void setBaseGgufPath(String baseGgufPath) {
        this.baseGgufPath = baseGgufPath;
    }

    public String getCovariantGgufPath() {
        return covariantGgufPath;
    }

    public void setCovariantGgufPath(String covariantGgufPath) {
        this.covariantGgufPath = covariantGgufPath;
    }

    public long getRequiredVramMb() {
        return requiredVramMb;
    }

    public void setRequiredVramMb(long requiredVramMb) {
        this.requiredVramMb = requiredVramMb;
    }

    public int getTotalLayers() {
        return totalLayers;
    }

    public void setTotalLayers(int totalLayers) {
        this.totalLayers = totalLayers;
    }

    public long getKvVramPer1kTokensMb() {
        return kvVramPer1kTokensMb;
    }

    public void setKvVramPer1kTokensMb(long kvVramPer1kTokensMb) {
        this.kvVramPer1kTokensMb = kvVramPer1kTokensMb;
    }

    public int getMaxContextLength() {
        return maxContextLength <= 0 ? 262144 : maxContextLength;
    }

    public void setMaxContextLength(int maxContextLength) {
        this.maxContextLength = maxContextLength <= 0 ? 262144 : maxContextLength;
    }

    public String getPolicyJson() {
        return policyJson;
    }

    public void setPolicyJson(String policyJson) {
        this.policyJson = policyJson;
    }

    public SecurityProfile getSecurityProfile() {
        return securityProfile;
    }

    public void setSecurityProfile(SecurityProfile securityProfile) {
        this.securityProfile = securityProfile;
    }

    public OrchestrationMode getOrchestrationMode() {
        return orchestrationMode;
    }

    public void setOrchestrationMode(OrchestrationMode orchestrationMode) {
        this.orchestrationMode = orchestrationMode;
    }

    public int getMinActiveRigs() {
        return minActiveRigs;
    }

    public void setMinActiveRigs(int minActiveRigs) {
        this.minActiveRigs = minActiveRigs;
    }

    public int getMaxActiveRigs() {
        return maxActiveRigs;
    }

    public void setMaxActiveRigs(int maxActiveRigs) {
        this.maxActiveRigs = maxActiveRigs;
    }

    public int getPriorityOrder() {
        return priorityOrder;
    }

    public void setPriorityOrder(int priorityOrder) {
        this.priorityOrder = priorityOrder;
    }

    public boolean isCovariantCompiled() {
        return covariantCompiled;
    }

    public void setCovariantCompiled(boolean covariantCompiled) {
        this.covariantCompiled = covariantCompiled;
    }

    public Double getLastCompileMs() {
        return lastCompileMs;
    }

    public void setLastCompileMs(Double lastCompileMs) {
        this.lastCompileMs = lastCompileMs;
    }

    public Double getLastCompileThroughputMbPerSec() {
        return lastCompileThroughputMbPerSec;
    }

    public void setLastCompileThroughputMbPerSec(Double lastCompileThroughputMbPerSec) {
        this.lastCompileThroughputMbPerSec = lastCompileThroughputMbPerSec;
    }

    public Long getTransformedQ50Rows() {
        return transformedQ50Rows;
    }

    public void setTransformedQ50Rows(Long transformedQ50Rows) {
        this.transformedQ50Rows = transformedQ50Rows;
    }

    public Integer getCalibratedRmsnormTensors() {
        return calibratedRmsnormTensors;
    }

    public void setCalibratedRmsnormTensors(Integer calibratedRmsnormTensors) {
        this.calibratedRmsnormTensors = calibratedRmsnormTensors;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }
}
