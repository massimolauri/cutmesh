package io.cutmesh.controlplane.iam;

import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.ObjectMapper;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

/**
 * RFC 7519 HMAC-SHA256 (HS256) JWT issuer and verifier for the Control Plane API.
 */
@Service
public class JwtTokenService {

    private static final String HMAC_ALGO = "HmacSHA256";
    private static final long TOKEN_TTL_SECONDS = 86_400L; // 24 hours
    private static final Base64.Encoder B64_URL_ENC = Base64.getUrlEncoder().withoutPadding();
    private static final Base64.Decoder B64_URL_DEC = Base64.getUrlDecoder();

    private final ObjectMapper objectMapper;
    private final byte[] signingKeyBytes;

    public JwtTokenService(
        ObjectMapper objectMapper,
        @Value("${cutmesh.jwt.secret:}") String jwtSecret
    ) {
        this.objectMapper = objectMapper;
        if (jwtSecret == null || jwtSecret.isBlank()) {
            byte[] generated = new byte[32];
            new java.security.SecureRandom().nextBytes(generated);
            this.signingKeyBytes = generated;
            org.slf4j.LoggerFactory.getLogger(JwtTokenService.class).warn(
                "CUTMESH_JWT_SECRET is unset; generated an ephemeral runtime HMAC-SHA256 signing key. Set CUTMESH_JWT_SECRET in production so JWT sessions survive restarts."
            );
        } else {
            this.signingKeyBytes = jwtSecret.getBytes(StandardCharsets.UTF_8);
        }
    }

    public String generateToken(AppUser user) {
        try {
            long now = Instant.now().getEpochSecond();
            Map<String, Object> header = Map.of("alg", "HS256", "typ", "JWT");
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("sub", user.getId());
            payload.put("email", user.getEmail());
            payload.put("name", user.getDisplayName());
            payload.put("role", user.getRole());
            payload.put("iat", now);
            payload.put("exp", now + TOKEN_TTL_SECONDS);

            String headerB64 = B64_URL_ENC.encodeToString(objectMapper.writeValueAsBytes(header));
            String payloadB64 = B64_URL_ENC.encodeToString(objectMapper.writeValueAsBytes(payload));
            String signingInput = headerB64 + "." + payloadB64;
            String signatureB64 = signHmacSha256(signingInput);
            return signingInput + "." + signatureB64;
        } catch (Exception e) {
            throw new IllegalStateException("Impossibile generare il token JWT", e);
        }
    }

    public String verifyAndExtractUserId(String authorizationHeader) {
        if (authorizationHeader == null || !authorizationHeader.startsWith("Bearer ")) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Token JWT mancante nell'header Authorization");
        }
        String token = authorizationHeader.substring(7).trim();
        if (token.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Token JWT vuoto");
        }

        String[] parts = token.split("\\.");
        if (parts.length != 3) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Formato token JWT non valido");
        }

        String signingInput = parts[0] + "." + parts[1];
        String expectedSig = signHmacSha256(signingInput);
        if (!MessageDigest.isEqual(
            expectedSig.getBytes(StandardCharsets.UTF_8),
            parts[2].getBytes(StandardCharsets.UTF_8)
        )) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Firma crittografica del token JWT non valida");
        }

        try {
            byte[] payloadJson = B64_URL_DEC.decode(parts[1]);
            Map<String, Object> claims = objectMapper.readValue(
                payloadJson,
                new TypeReference<Map<String, Object>>() {}
            );
            Number exp = (Number) claims.get("exp");
            long now = Instant.now().getEpochSecond();
            if (exp == null || now >= exp.longValue()) {
                throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Sessione JWT scaduta. Effettua nuovamente il login.");
            }
            String sub = (String) claims.get("sub");
            if (sub == null || sub.isBlank()) {
                throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Claim 'sub' mancante nel token JWT");
            }
            return sub;
        } catch (ResponseStatusException rse) {
            throw rse;
        } catch (Exception e) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Impossibile decodificare il token JWT");
        }
    }

    private String signHmacSha256(String signingInput) {
        try {
            Mac mac = Mac.getInstance(HMAC_ALGO);
            mac.init(new SecretKeySpec(signingKeyBytes, HMAC_ALGO));
            byte[] sig = mac.doFinal(signingInput.getBytes(StandardCharsets.UTF_8));
            return B64_URL_ENC.encodeToString(sig);
        } catch (Exception e) {
            throw new IllegalStateException("HMAC-SHA256 non disponibile", e);
        }
    }
}

