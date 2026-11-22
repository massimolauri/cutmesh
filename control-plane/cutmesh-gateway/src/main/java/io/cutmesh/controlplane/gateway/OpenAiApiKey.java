package io.cutmesh.controlplane.gateway;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.Instant;
import java.util.UUID;

/**
 * OpenAI-compatible consumer API Key (`sk-cutmesh-...`) stored solely as a SHA-256 hash.
 */
@Entity
@Table(
    name = "openai_api_keys",
    indexes = {
        @Index(name = "idx_openai_key_sha256", columnList = "key_sha256", unique = true),
        @Index(name = "idx_openai_key_owner", columnList = "owner_user_id")
    }
)
public class OpenAiApiKey {

    @Id
    @Column(name = "id", nullable = false, updatable = false, length = 36)
    private String id;

    @Column(name = "owner_user_id", nullable = false, length = 36)
    private String ownerUserId;

    @Column(name = "name", nullable = false, length = 120)
    private String name;

    @Column(name = "key_prefix", nullable = false, length = 32)
    private String keyPrefix;

    @Column(name = "key_sha256", nullable = false, unique = true, length = 64)
    private String keySha256;

    @Column(name = "rate_limit_rpm", nullable = false)
    private int rateLimitRpm = 60;

    @Column(name = "total_requests", nullable = false)
    private long totalRequests = 0L;

    @Column(name = "total_tokens_generated", nullable = false)
    private long totalTokensGenerated = 0L;

    @Column(name = "revoked", nullable = false)
    private boolean revoked = false;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    @Column(name = "last_used_at")
    private Instant lastUsedAt;

    public OpenAiApiKey() {
    }

    public OpenAiApiKey(String ownerUserId, String name, String keyPrefix, String keySha256, int rateLimitRpm) {
        this.id = UUID.randomUUID().toString();
        this.ownerUserId = ownerUserId;
        this.name = name;
        this.keyPrefix = keyPrefix;
        this.keySha256 = keySha256;
        this.rateLimitRpm = rateLimitRpm;
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

    public String getName() {
        return name;
    }

    public String getKeyPrefix() {
        return keyPrefix;
    }

    public String getKeySha256() {
        return keySha256;
    }

    public int getRateLimitRpm() {
        return rateLimitRpm;
    }

    public long getTotalRequests() {
        return totalRequests;
    }

    public void setTotalRequests(long totalRequests) {
        this.totalRequests = totalRequests;
    }

    public long getTotalTokensGenerated() {
        return totalTokensGenerated;
    }

    public void setTotalTokensGenerated(long totalTokensGenerated) {
        this.totalTokensGenerated = totalTokensGenerated;
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
