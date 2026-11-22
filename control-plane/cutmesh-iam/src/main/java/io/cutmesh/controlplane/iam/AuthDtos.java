package io.cutmesh.controlplane.iam;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import java.time.Instant;

public final class AuthDtos {

    private AuthDtos() {
    }

    public record GoogleOAuthLoginRequest(
        String idToken,
        @Email String email,
        String displayName,
        String googleSub,
        String avatarUrl,
        @Pattern(regexp = "it|en") String preferredLocale,
        @Pattern(regexp = "SUPER_ADMIN|HARDWARE_PROVIDER|API_USER") String requestedRole
    ) {
    }

    public record ApiKeyLoginRequest(
        @NotBlank String apiKey
    ) {
    }

    public record EmailPasswordLoginRequest(
        @NotBlank @Email String email,
        @NotBlank String password
    ) {
    }

    public record RegisterUserRequest(
        @NotBlank @Email String email,
        @NotBlank String password,
        @NotBlank String displayName,
        @Pattern(regexp = "SUPER_ADMIN|HARDWARE_PROVIDER|API_USER") String role,
        @Pattern(regexp = "it|en") String preferredLocale
    ) {
    }

    public record UpdateLocaleRequest(
        @NotBlank @Pattern(regexp = "it|en") String preferredLocale
    ) {
    }

    public record UpdateRoleRequest(
        @NotBlank @Pattern(regexp = "SUPER_ADMIN|HARDWARE_PROVIDER|API_USER") String role
    ) {
    }

    public record UserProfileResponse(
        String id,
        String email,
        String displayName,
        String oauthProvider,
        String avatarUrl,
        String preferredLocale,
        String role,
        Instant createdAt,
        Instant lastLoginAt,
        String jwtToken
    ) {
    }
}
