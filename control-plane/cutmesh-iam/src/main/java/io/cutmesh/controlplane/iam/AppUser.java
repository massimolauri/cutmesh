package io.cutmesh.controlplane.iam;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.Instant;
import java.util.UUID;

/**
 * Authenticated user entity supporting Google OAuth2 SSO, Email/Password authentication,
 * 3-Role RBAC (SUPER_ADMIN, HARDWARE_PROVIDER, API_USER), and bilingual locale preferences (EN/IT).
 */
@Entity
@Table(
    name = "app_users",
    indexes = {
        @Index(name = "idx_app_users_email", columnList = "email", unique = true),
        @Index(name = "idx_app_users_oauth_subject", columnList = "oauth_provider,oauth_subject", unique = true)
    }
)
public class AppUser {

    @Id
    @Column(name = "id", nullable = false, updatable = false, length = 36)
    private String id;

    @Column(name = "email", nullable = false, unique = true, length = 180)
    private String email;

    @Column(name = "display_name", nullable = false, length = 120)
    private String displayName;

    @Column(name = "oauth_provider", nullable = false, length = 32)
    private String oauthProvider;

    @Column(name = "oauth_subject", nullable = false, length = 128)
    private String oauthSubject;

    @Column(name = "password_hash", length = 128)
    private String passwordHash;

    @Column(name = "avatar_url", length = 512)
    private String avatarUrl;

    @Column(name = "preferred_locale", nullable = false, length = 8)
    private String preferredLocale = "it";

    @Column(name = "role", nullable = false, length = 32)
    private String role = "SUPER_ADMIN";

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    @Column(name = "last_login_at", nullable = false)
    private Instant lastLoginAt;

    public AppUser() {
    }

    public AppUser(
        String email,
        String displayName,
        String oauthProvider,
        String oauthSubject,
        String avatarUrl,
        String preferredLocale
    ) {
        this(email, displayName, oauthProvider, oauthSubject, avatarUrl, preferredLocale, "SUPER_ADMIN", null);
    }

    public AppUser(
        String email,
        String displayName,
        String oauthProvider,
        String oauthSubject,
        String avatarUrl,
        String preferredLocale,
        String role,
        String passwordHash
    ) {
        this.id = UUID.randomUUID().toString();
        this.email = email;
        this.displayName = displayName;
        this.oauthProvider = oauthProvider;
        this.oauthSubject = oauthSubject;
        this.avatarUrl = avatarUrl;
        this.preferredLocale = preferredLocale != null ? preferredLocale : "it";
        this.role = role != null ? role : "API_USER";
        this.passwordHash = passwordHash;
        this.createdAt = Instant.now();
        this.lastLoginAt = Instant.now();
    }

    @PrePersist
    public void prePersist() {
        if (this.id == null) {
            this.id = UUID.randomUUID().toString();
        }
        if (this.createdAt == null) {
            this.createdAt = Instant.now();
        }
        if (this.lastLoginAt == null) {
            this.lastLoginAt = Instant.now();
        }
        if (this.role == null) {
            this.role = "API_USER";
        }
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getDisplayName() {
        return displayName;
    }

    public void setDisplayName(String displayName) {
        this.displayName = displayName;
    }

    public String getOauthProvider() {
        return oauthProvider;
    }

    public void setOauthProvider(String oauthProvider) {
        this.oauthProvider = oauthProvider;
    }

    public String getOauthSubject() {
        return oauthSubject;
    }

    public void setOauthSubject(String oauthSubject) {
        this.oauthSubject = oauthSubject;
    }

    public String getPasswordHash() {
        return passwordHash;
    }

    public void setPasswordHash(String passwordHash) {
        this.passwordHash = passwordHash;
    }

    public String getAvatarUrl() {
        return avatarUrl;
    }

    public void setAvatarUrl(String avatarUrl) {
        this.avatarUrl = avatarUrl;
    }

    public String getPreferredLocale() {
        return preferredLocale;
    }

    public void setPreferredLocale(String preferredLocale) {
        this.preferredLocale = preferredLocale;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }

    public Instant getLastLoginAt() {
        return lastLoginAt;
    }

    public void setLastLoginAt(Instant lastLoginAt) {
        this.lastLoginAt = lastLoginAt;
    }
}
