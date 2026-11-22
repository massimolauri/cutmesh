package io.cutmesh.controlplane.audit;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.Instant;
import java.util.UUID;

/**
 * Immutable Security, Mathematical & Operational Audit Event persisted by the Control Plane.
 * Tracks Zero-Weight-Disclosure Covariant Audits, Physical GGUF Compilations, Anti-Collusion
 * Rig Locks (PK_1 != PK_K), Canary Verification on inference requests, and Watchdog events.
 */
@Entity
@Table(
    name = "security_audit_events",
    indexes = {
        @Index(name = "idx_audit_category_created", columnList = "category,created_at"),
        @Index(name = "idx_audit_created", columnList = "created_at")
    }
)
public class SecurityAuditEvent {

    @Id
    @Column(name = "id", nullable = false, updatable = false, length = 36)
    private String id;

    @Column(name = "category", nullable = false, length = 64)
    private String category;

    @Column(name = "severity", nullable = false, length = 24)
    private String severity;

    @Column(name = "title_en", nullable = false, length = 200)
    private String titleEn;

    @Column(name = "title_it", nullable = false, length = 200)
    private String titleIt;

    @Column(name = "summary_en", nullable = false, length = 1024)
    private String summaryEn;

    @Column(name = "summary_it", nullable = false, length = 1024)
    private String summaryIt;

    @Column(name = "subject_id", length = 128)
    private String subjectId;

    @Column(name = "passed", nullable = false)
    private boolean passed = true;

    @Column(name = "f32_relative_err")
    private Double f32RelativeErr;

    @Column(name = "q8_to_q5_relative_err")
    private Double q8ToQ5RelativeErr;

    @Column(name = "canary_relative_err")
    private Double canaryRelativeErr;

    @Column(name = "rank1_corr_max")
    private Double rank1CorrMax;

    @Column(name = "anti_collusion_enforced")
    private Boolean antiCollusionEnforced;

    @Column(name = "metrics_json", length = 4000)
    private String metricsJson;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    public SecurityAuditEvent() {
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

    public String getId() {
        return id;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public String getSeverity() {
        return severity;
    }

    public void setSeverity(String severity) {
        this.severity = severity;
    }

    public String getTitleEn() {
        return titleEn;
    }

    public void setTitleEn(String titleEn) {
        this.titleEn = titleEn;
    }

    public String getTitleIt() {
        return titleIt;
    }

    public void setTitleIt(String titleIt) {
        this.titleIt = titleIt;
    }

    public String getSummaryEn() {
        return summaryEn;
    }

    public void setSummaryEn(String summaryEn) {
        this.summaryEn = summaryEn;
    }

    public String getSummaryIt() {
        return summaryIt;
    }

    public void setSummaryIt(String summaryIt) {
        this.summaryIt = summaryIt;
    }

    public String getSubjectId() {
        return subjectId;
    }

    public void setSubjectId(String subjectId) {
        this.subjectId = subjectId;
    }

    public boolean isPassed() {
        return passed;
    }

    public void setPassed(boolean passed) {
        this.passed = passed;
    }

    public Double getF32RelativeErr() {
        return f32RelativeErr;
    }

    public void setF32RelativeErr(Double f32RelativeErr) {
        this.f32RelativeErr = f32RelativeErr;
    }

    public Double getQ8ToQ5RelativeErr() {
        return q8ToQ5RelativeErr;
    }

    public void setQ8ToQ5RelativeErr(Double q8ToQ5RelativeErr) {
        this.q8ToQ5RelativeErr = q8ToQ5RelativeErr;
    }

    public Double getCanaryRelativeErr() {
        return canaryRelativeErr;
    }

    public void setCanaryRelativeErr(Double canaryRelativeErr) {
        this.canaryRelativeErr = canaryRelativeErr;
    }

    public Double getRank1CorrMax() {
        return rank1CorrMax;
    }

    public void setRank1CorrMax(Double rank1CorrMax) {
        this.rank1CorrMax = rank1CorrMax;
    }

    public Boolean getAntiCollusionEnforced() {
        return antiCollusionEnforced;
    }

    public void setAntiCollusionEnforced(Boolean antiCollusionEnforced) {
        this.antiCollusionEnforced = antiCollusionEnforced;
    }

    public String getMetricsJson() {
        return metricsJson;
    }

    public void setMetricsJson(String metricsJson) {
        this.metricsJson = metricsJson;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }
}

