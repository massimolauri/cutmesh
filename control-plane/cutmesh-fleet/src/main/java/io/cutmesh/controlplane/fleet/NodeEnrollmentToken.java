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
 * Enrollment token (`syn_node_live_...`) used by Docker GPU workers to authenticate
 * their heartbeats and associate discovered hardware with an owner user.
 */
@Entity
@Table(
    name = "node_enrollment_tokens",
    indexes = {
        @Index(name = "idx_node_token_hash", columnList = "token_sha256", unique = true),
        @Index(name = "idx_node_token_owner", columnList = "owner_user_id")
    }
)
public class NodeEnrollmentToken {

    @Id
    @Column(name = "id", nullable = false, updatable = false, length = 36)
    private String id;

    @Column(name = "owner_user_id", nullable = false, length = 36)
    private String ownerUserId;

    @Column(name = "label", nullable = false, length = 120)
    private String label;

    @Column(name = "token_prefix", nullable = false, length = 32)
    private String tokenPrefix;

    @Column(name = "token_sha256", nullable = false, unique = true, length = 64)
    private String tokenSha256;

    @Enumerated(EnumType.STRING)
    @Column(name = "default_pool_mode", nullable = false, length = 24)
    private PoolMode defaultPoolMode = PoolMode.SHARED_POOL;

    @Column(name = "revoked", nullable = false)
    private boolean revoked = false;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    @Column(name = "last_used_at")
    private Instant lastUsedAt;

    public NodeEnrollmentToken() {
    }

    public NodeEnrollmentToken(String ownerUserId, String label, String tokenPrefix, String tokenSha256, PoolMode defaultPoolMode) {
        this.id = UUID.randomUUID().toString();
        this.ownerUserId = ownerUserId;
        this.label = label;
        this.tokenPrefix = tokenPrefix;
        this.tokenSha256 = tokenSha256;
        this.defaultPoolMode = defaultPoolMode != null ? defaultPoolMode : PoolMode.SHARED_POOL;
        this.revoked = false;
        this.createdAt = Instant.now();
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

    public String getOwnerUserId() {
        return ownerUserId;
    }

    public void setOwnerUserId(String ownerUserId) {
        this.ownerUserId = ownerUserId;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getTokenPrefix() {
        return tokenPrefix;
    }

    public String getTokenSha256() {
        return tokenSha256;
    }

    public PoolMode getDefaultPoolMode() {
        return defaultPoolMode;
    }

    public void setDefaultPoolMode(PoolMode defaultPoolMode) {
        this.defaultPoolMode = defaultPoolMode;
    }

    public boolean isRevoked() {
        return revoked;
    }

    public void setRevoked(boolean revoked) {
        this.revoked = revoked;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getLastUsedAt() {
        return lastUsedAt;
    }

    public void setLastUsedAt(Instant lastUsedAt) {
        this.lastUsedAt = lastUsedAt;
    }
}
