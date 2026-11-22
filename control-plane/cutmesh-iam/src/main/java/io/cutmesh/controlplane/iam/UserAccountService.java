package io.cutmesh.controlplane.iam;

import io.cutmesh.controlplane.shared.CryptoHashUtils;
import io.cutmesh.controlplane.audit.SecurityAuditService;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import io.cutmesh.controlplane.iam.AuthDtos.EmailPasswordLoginRequest;
import io.cutmesh.controlplane.iam.AuthDtos.GoogleOAuthLoginRequest;
import io.cutmesh.controlplane.iam.AuthDtos.RegisterUserRequest;
import io.cutmesh.controlplane.iam.AuthDtos.UserProfileResponse;

import jakarta.persistence.EntityNotFoundException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

@Service
public class UserAccountService {

    private final AppUserRepository userRepository;
    private final ObjectProvider<ApiKeyAuthLookup> apiKeyAuthLookupProvider;
    private final SecurityAuditService auditService;
    private final JwtTokenService jwtTokenService;
    private final ObjectMapper objectMapper;
    private final String googleClientId;
    private final HttpClient httpClient;

    public UserAccountService(
        AppUserRepository userRepository,
        ObjectProvider<ApiKeyAuthLookup> apiKeyAuthLookupProvider,
        SecurityAuditService auditService,
        JwtTokenService jwtTokenService,
        ObjectMapper objectMapper,
        @Value("${cutmesh.google.client-id:}") String googleClientId
    ) {
        this.userRepository = userRepository;
        this.apiKeyAuthLookupProvider = apiKeyAuthLookupProvider;
        this.auditService = auditService;
        this.jwtTokenService = jwtTokenService;
        this.objectMapper = objectMapper;
        this.googleClientId = googleClientId != null ? googleClientId.trim() : "";
        this.httpClient = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    }

    public boolean isGoogleOAuthEnabled() {
        return !this.googleClientId.isBlank();
    }

    @Transactional
    public UserProfileResponse loginOrRegisterGoogleUser(GoogleOAuthLoginRequest request) {
        if (!isGoogleOAuthEnabled()) {
            throw new ResponseStatusException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "Google OAuth2 non configurato su questo Control Plane (impostare CUTMESH_GOOGLE_CLIENT_ID)."
            );
        }
        if (request == null || request.idToken() == null || request.idToken().isBlank()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Google ID token mancante.");
        }

        JsonNode claims = verifyGoogleIdToken(request.idToken().trim());
        String verifiedSub = claims.path("sub").asString("");
        String verifiedEmail = claims.path("email").asString("").trim().toLowerCase(Locale.ROOT);
        String verifiedName = claims.path("name").asString("").trim();
        String verifiedPicture = claims.path("picture").asString("").trim();
        if (verifiedSub.isBlank() || verifiedEmail.isBlank()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Google ID token incompleto.");
        }
        String displayName = !verifiedName.isBlank() ? verifiedName : verifiedEmail;
        String avatarFallback = !verifiedPicture.isBlank()
            ? verifiedPicture
            : "https://api.dicebear.com/7.x/shapes/svg?seed=" + verifiedEmail;

        AppUser user = userRepository
            .findByOauthProviderAndOauthSubject("GOOGLE", verifiedSub)
            .or(() -> userRepository.findByEmailIgnoreCase(verifiedEmail))
            .map(existing -> {
                existing.setDisplayName(displayName);
                existing.setOauthSubject(verifiedSub);
                existing.setAvatarUrl(avatarFallback);
                if (request.preferredLocale() != null && !request.preferredLocale().isBlank()) {
                    existing.setPreferredLocale(request.preferredLocale());
                }
                existing.setLastLoginAt(Instant.now());
                return userRepository.save(existing);
            })
            .orElseGet(() -> {
                AppUser created = new AppUser(
                    verifiedEmail,
                    displayName,
                    "GOOGLE",
                    verifiedSub,
                    avatarFallback,
                    request.preferredLocale() != null ? request.preferredLocale() : "it",
                    "API_USER",
                    null
                );
                return userRepository.save(created);
            });

        auditService.recordEvent(
            "USER_SSO_AUTH",
            "INFO",
            "Google OAuth2 Session Authenticated (" + user.getEmail() + " - " + user.getRole() + ")",
            "Sessione Google OAuth2 Autenticata (" + user.getEmail() + " - " + user.getRole() + ")",
            "User '" + user.getDisplayName() + "' signed in via verified Google ID token with role '" + user.getRole() + "'.",
            "L'utente '" + user.getDisplayName() + "' ha effettuato l'accesso tramite Google ID token verificato con ruolo '" + user.getRole() + "'.",
            user.getId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );

        return toAuthenticatedProfileResponse(user);
    }

    private JsonNode verifyGoogleIdToken(String idToken) {
        try {
            String url = "https://oauth2.googleapis.com/tokeninfo?id_token="
                + URLEncoder.encode(idToken, StandardCharsets.UTF_8);
            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(5))
                .GET()
                .build();
            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) {
                throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Google ID token non valido.");
            }
            JsonNode claims = objectMapper.readTree(resp.body());
            String aud = claims.path("aud").asString("");
            boolean emailVerified = "true".equalsIgnoreCase(claims.path("email_verified").asString(""));
            if (!googleClientId.equals(aud) || !emailVerified) {
                throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Audience o email verificata del token Google non corrispondente.");
            }
            return claims;
        } catch (ResponseStatusException rse) {
            throw rse;
        } catch (Exception e) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Verifica Google ID token fallita: " + e.getMessage());
        }
    }

    @Transactional
    public UserProfileResponse loginWithEmailPassword(EmailPasswordLoginRequest request) {
        AppUser user = userRepository.findByEmailIgnoreCase(request.email().trim())
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Credenziali non valide."));

        if (user.getPasswordHash() == null || user.getPasswordHash().isBlank()
            || !CryptoHashUtils.verifyPassword(request.password(), user.getPasswordHash())) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Credenziali non valide.");
        }
        if (CryptoHashUtils.isLegacySha256Hash(user.getPasswordHash())) {
            user.setPasswordHash(CryptoHashUtils.hashPassword(request.password()));
        }

        user.setLastLoginAt(Instant.now());
        AppUser saved = userRepository.save(user);

        auditService.recordEvent(
            "USER_SSO_AUTH",
            "INFO",
            "User Login Authenticated (" + saved.getEmail() + " - " + saved.getRole() + ")",
            "Accesso Utente Autenticato (" + saved.getEmail() + " - " + saved.getRole() + ")",
            "User '" + saved.getDisplayName() + "' signed in with role '" + saved.getRole() + "'.",
            "L'utente '" + saved.getDisplayName() + "' ha effettuato l'accesso con ruolo '" + saved.getRole() + "'.",
            saved.getId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );

        return toAuthenticatedProfileResponse(saved);
    }

    @Transactional
    public UserProfileResponse loginWithApiKey(String rawApiKey) {
        if (rawApiKey == null || !rawApiKey.trim().startsWith("sk-cutmesh-")) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Formato chiave API Cutmesh non valido.");
        }
        String sha256 = CryptoHashUtils.sha256Hex(rawApiKey.trim());
        ApiKeyAuthLookup lookup = apiKeyAuthLookupProvider.getIfAvailable();
        String ownerUserId = (lookup != null ? lookup.resolveOwnerUserIdAndTouchBySha256(sha256) : Optional.<String>empty())
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Chiave API non valida o revocata."));
        AppUser owner = userRepository.findById(ownerUserId)
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Utente proprietario della chiave non trovato."));
        owner.setLastLoginAt(Instant.now());
        AppUser saved = userRepository.save(owner);
        return toAuthenticatedProfileResponse(saved);
    }

    @Transactional
    public UserProfileResponse registerLocalUser(RegisterUserRequest request) {
        String email = request.email().trim().toLowerCase(Locale.ROOT);
        if (userRepository.findByEmailIgnoreCase(email).isPresent()) {
            throw new IllegalArgumentException("Un account con questa email esiste già: " + email);
        }

        String requestedRole = (request.role() != null && !request.role().isBlank())
            ? request.role().trim().toUpperCase(Locale.ROOT)
            : "API_USER";
        if ("SUPER_ADMIN".equals(requestedRole)) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Registrazione autonoma come SUPER_ADMIN non consentita.");
        }
        String role = "HARDWARE_PROVIDER".equals(requestedRole) ? "HARDWARE_PROVIDER" : "API_USER";
        String locale = (request.preferredLocale() != null && !request.preferredLocale().isBlank()) ? request.preferredLocale() : "it";
        String passwordHash = CryptoHashUtils.hashPassword(request.password());
        String avatar = "https://api.dicebear.com/7.x/shapes/svg?seed=" + email;

        AppUser created = new AppUser(
            email,
            request.displayName().trim(),
            "LOCAL",
            "local|" + email,
            avatar,
            locale,
            role,
            passwordHash
        );
        AppUser saved = userRepository.save(created);

        auditService.recordEvent(
            "USER_SSO_AUTH",
            "VERIFIED",
            "Registered New User Account (" + saved.getEmail() + " -> " + saved.getRole() + ")",
            "Registrato Nuovo Account Utente (" + saved.getEmail() + " -> " + saved.getRole() + ")",
            "Created user '" + saved.getDisplayName() + "' with role " + saved.getRole() + ".",
            "Creato utente '" + saved.getDisplayName() + "' con ruolo " + saved.getRole() + ".",
            saved.getId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );

        return toAuthenticatedProfileResponse(saved);
    }

    @Transactional(readOnly = true)
    public AppUser requireRole(String authenticatedUserId, String... allowedRoles) {
        if (authenticatedUserId == null || authenticatedUserId.isBlank()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Sessione non autenticata.");
        }
        AppUser user = userRepository.findById(authenticatedUserId)
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Utente non trovato."));
        if (allowedRoles != null && allowedRoles.length > 0) {
            boolean matched = false;
            for (String allowed : allowedRoles) {
                if (allowed.equals(user.getRole())) {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                throw new ResponseStatusException(
                    HttpStatus.FORBIDDEN,
                    "Accesso negato per ruolo " + user.getRole()
                );
            }
        }
        return user;
    }

    @Transactional(readOnly = true)
    public UserProfileResponse getUserByIdOrDefault(String userId) {
        if (userId == null || userId.isBlank()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Sessione non autenticata");
        }
        return userRepository.findById(userId)
            .map(this::toAuthenticatedProfileResponse)
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Utente non trovato"));
    }

    @Transactional(readOnly = true)
    public UserProfileResponse getCurrentDefaultUser() {
        List<AppUser> all = userRepository.findAll();
        if (all.isEmpty()) {
            throw new EntityNotFoundException("No default user initialized");
        }
        return toProfileResponse(all.get(0));
    }

    @Transactional(readOnly = true)
    public AppUser getDefaultUserEntity() {
        return userRepository.findAll().stream().findFirst()
            .orElseThrow(() -> new EntityNotFoundException("No default user initialized"));
    }

    @Transactional(readOnly = true)
    public List<UserProfileResponse> listAllUsers() {
        return userRepository.findAll().stream()
            .map(UserAccountService::toProfileResponse)
            .toList();
    }

    @Transactional
    public UserProfileResponse updatePreferredLocale(String userId, String preferredLocale) {
        AppUser user = (userId != null && !userId.isBlank())
            ? userRepository.findById(userId).orElseGet(this::getDefaultUserEntity)
            : getDefaultUserEntity();
        user.setPreferredLocale(preferredLocale);
        userRepository.save(user);
        return toProfileResponse(user);
    }

    @Transactional
    public UserProfileResponse updateUserRole(String userId, String role) {
        AppUser user = (userId != null && !userId.isBlank())
            ? userRepository.findById(userId).orElseGet(this::getDefaultUserEntity)
            : getDefaultUserEntity();
        String oldRole = user.getRole();
        user.setRole(role);
        AppUser saved = userRepository.save(user);

        auditService.recordEvent(
            "USER_ROLE_CHANGE",
            "INFO",
            "Updated User Role for " + saved.getEmail() + ": " + oldRole + " -> " + role,
            "Aggiornato Ruolo Utente per " + saved.getEmail() + ": " + oldRole + " -> " + role,
            "RBAC permission role changed from " + oldRole + " to " + role + ".",
            "Il ruolo RBAC è stato modificato da " + oldRole + " a " + role + ".",
            saved.getId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );

        return toAuthenticatedProfileResponse(saved);
    }

    @Transactional
    public void deleteUser(String userId) {
        if (userRepository.count() <= 1) {
            throw new IllegalStateException("Impossibile eliminare l'ultimo amministratore di sistema.");
        }
        userRepository.deleteById(userId);
    }

    public UserProfileResponse toAuthenticatedProfileResponse(AppUser user) {
        return new UserProfileResponse(
            user.getId(),
            user.getEmail(),
            user.getDisplayName(),
            user.getOauthProvider(),
            user.getAvatarUrl(),
            user.getPreferredLocale(),
            user.getRole(),
            user.getCreatedAt(),
            user.getLastLoginAt(),
            jwtTokenService.generateToken(user)
        );
    }

    public static UserProfileResponse toProfileResponse(AppUser user) {
        return new UserProfileResponse(
            user.getId(),
            user.getEmail(),
            user.getDisplayName(),
            user.getOauthProvider(),
            user.getAvatarUrl(),
            user.getPreferredLocale(),
            user.getRole(),
            user.getCreatedAt(),
            user.getLastLoginAt(),
            null
        );
    }
}
