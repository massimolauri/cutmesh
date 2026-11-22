package io.cutmesh.controlplane.gateway;

import io.cutmesh.controlplane.shared.CryptoHashUtils;
import io.cutmesh.controlplane.iam.ApiKeyAuthLookup;
import io.cutmesh.controlplane.iam.JwtTokenService;
import io.cutmesh.controlplane.audit.SecurityAuditService;
import io.cutmesh.controlplane.iam.UserAccountService;
import io.cutmesh.controlplane.iam.AppUser;

import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.CreateOpenAiKeyRequest;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.CreatedOpenAiKeyResponse;

import jakarta.persistence.EntityNotFoundException;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

/**
 * Single-responsibility service managing OpenAI-compatible consumer API keys (`sk-cutmesh-...`):
 * issuance, SHA-256 hashing, RBAC revocation, and Bearer/JWT authentication.
 */
@Service
public class OpenAiApiKeyService implements ApiKeyAuthLookup {

    private final OpenAiApiKeyRepository apiKeyRepository;
    private final UserAccountService userAccountService;
    private final SecurityAuditService auditService;
    private final JwtTokenService jwtTokenService;

    public OpenAiApiKeyService(
        OpenAiApiKeyRepository apiKeyRepository,
        UserAccountService userAccountService,
        SecurityAuditService auditService,
        JwtTokenService jwtTokenService
    ) {
        this.apiKeyRepository = apiKeyRepository;
        this.userAccountService = userAccountService;
        this.auditService = auditService;
        this.jwtTokenService = jwtTokenService;
    }

    @Override
    @Transactional
    public Optional<String> resolveOwnerUserIdAndTouchBySha256(String sha256) {
        return apiKeyRepository.findByKeySha256AndRevokedFalse(sha256).map(key -> {
            key.setLastUsedAt(Instant.now());
            apiKeyRepository.save(key);
            return key.getOwnerUserId();
        });
    }

    @Transactional
    public CreatedOpenAiKeyResponse createApiKey(String userId, CreateOpenAiKeyRequest request) {
        AppUser defaultOwner = userAccountService.getDefaultUserEntity();
        String effectiveUserId = (userId != null && !userId.isBlank()) ? userId : defaultOwner.getId();

        String rawKey = CryptoHashUtils.generateOpenAiApiKey();
        String prefix = CryptoHashUtils.extractPrefix(rawKey, 16);
        String sha256 = CryptoHashUtils.sha256Hex(rawKey);

        OpenAiApiKey entity = new OpenAiApiKey(
            effectiveUserId,
            request.name(),
            prefix,
            sha256,
            request.rateLimitRpm()
        );
        OpenAiApiKey saved = apiKeyRepository.save(entity);

        auditService.recordEvent(
            "API_KEY_LIFECYCLE",
            "INFO",
            "Created OpenAI Consumer API Key '" + saved.getName() + "' (" + prefix + ")",
            "Creata Chiave API OpenAI Consumer '" + saved.getName() + "' (" + prefix + ")",
            "Issued SHA-256 hashed API key with rate limit " + saved.getRateLimitRpm() + " RPM.",
            "Emessa chiave API con hash SHA-256 e limite " + saved.getRateLimitRpm() + " RPM.",
            saved.getId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );

        return new CreatedOpenAiKeyResponse(
            saved.getId(),
            saved.getName(),
            rawKey,
            saved.getKeyPrefix(),
            saved.getRateLimitRpm(),
            saved.getCreatedAt()
        );
    }

    @Transactional(readOnly = true)
    public List<OpenAiApiKey> listApiKeys(String userId) {
        AppUser defaultOwner = userAccountService.getDefaultUserEntity();
        String effectiveUserId = (userId != null && !userId.isBlank()) ? userId : defaultOwner.getId();
        return apiKeyRepository.findByOwnerUserIdOrderByCreatedAtDesc(effectiveUserId);
    }

    @Transactional
    public void revokeApiKey(String keyId) {
        revokeApiKey(keyId, null, true);
    }

    @Transactional
    public void revokeApiKey(String keyId, String callerUserId, boolean isAdmin) {
        OpenAiApiKey key = apiKeyRepository.findById(keyId)
            .orElseThrow(() -> new EntityNotFoundException("API Key not found: " + keyId));
        if (!isAdmin && callerUserId != null && !callerUserId.equals(key.getOwnerUserId())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Non sei autorizzato a revocare questa API key.");
        }
        key.setRevoked(true);
        apiKeyRepository.save(key);

        auditService.recordEvent(
            "API_KEY_LIFECYCLE",
            "WARNING",
            "Revoked OpenAI Consumer API Key '" + key.getName() + "' (" + key.getKeyPrefix() + ")",
            "Revocata Chiave API OpenAI Consumer '" + key.getName() + "' (" + key.getKeyPrefix() + ")",
            "Key " + key.getKeyPrefix() + " was permanently revoked.",
            "La chiave " + key.getKeyPrefix() + " è stata revocata permanentemente.",
            key.getId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );
    }

    @Transactional
    public Optional<OpenAiApiKey> authenticateApiKeyOnly(String rawKey) {
        if (rawKey == null || !rawKey.startsWith("sk-cutmesh-")) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Invalid Cutmesh API key format");
        }
        String sha256 = CryptoHashUtils.sha256Hex(rawKey.trim());
        OpenAiApiKey key = apiKeyRepository.findByKeySha256AndRevokedFalse(sha256)
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Invalid or revoked Cutmesh API key"));
        return Optional.of(key);
    }

    @Transactional
    public Optional<OpenAiApiKey> authenticateBearerOrJwt(String authorizationHeader) {
        if (authorizationHeader == null || !authorizationHeader.startsWith("Bearer ")) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Missing or invalid Authorization: Bearer header");
        }
        String rawKey = authorizationHeader.substring("Bearer ".length()).trim();
        if (rawKey.startsWith("sk-cutmesh-")) {
            String sha256 = CryptoHashUtils.sha256Hex(rawKey);
            OpenAiApiKey key = apiKeyRepository.findByKeySha256AndRevokedFalse(sha256)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Invalid or revoked Cutmesh API key"));
            return Optional.of(key);
        }
        String userId = jwtTokenService.verifyAndExtractUserId(authorizationHeader);
        return apiKeyRepository.findByOwnerUserIdOrderByCreatedAtDesc(userId).stream()
            .filter(k -> !k.isRevoked())
            .findFirst();
    }
}
