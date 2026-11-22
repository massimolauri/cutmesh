package io.cutmesh.controlplane.shared;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.security.spec.InvalidKeySpecException;
import java.util.HexFormat;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.PBEKeySpec;

/**
 * Cryptographic token generation, PBKDF2-HMAC-SHA256 password hashing, and SHA-256 hashing utilities.
 * Raw secrets (`syn_node_live_...`, `sk-cutmesh-...`, and user passwords) are never stored in plaintext.
 */
public final class CryptoHashUtils {

    private static final SecureRandom SECURE_RANDOM = new SecureRandom();
    private static final String ALPHANUM = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private static final String PBKDF2_PREFIX = "pbkdf2$";
    private static final int PBKDF2_DEFAULT_ITERATIONS = 120_000;
    private static final int PBKDF2_SALT_BYTES = 16;
    private static final int PBKDF2_KEY_BITS = 256;

    private CryptoHashUtils() {
    }

    public static String hashPassword(String rawPassword) {
        if (rawPassword == null || rawPassword.isBlank()) {
            throw new IllegalArgumentException("Password must not be blank");
        }
        byte[] salt = new byte[PBKDF2_SALT_BYTES];
        SECURE_RANDOM.nextBytes(salt);
        byte[] derived = pbkdf2(rawPassword.toCharArray(), salt, PBKDF2_DEFAULT_ITERATIONS, PBKDF2_KEY_BITS);
        return PBKDF2_PREFIX
            + PBKDF2_DEFAULT_ITERATIONS
            + "$"
            + HexFormat.of().formatHex(salt)
            + "$"
            + HexFormat.of().formatHex(derived);
    }

    public static boolean verifyPassword(String rawPassword, String storedHash) {
        if (rawPassword == null || storedHash == null || storedHash.isBlank()) {
            return false;
        }
        if (storedHash.startsWith(PBKDF2_PREFIX)) {
            String[] parts = storedHash.split("\\$");
            if (parts.length != 4) {
                return false;
            }
            try {
                int iterations = Integer.parseInt(parts[1]);
                byte[] salt = HexFormat.of().parseHex(parts[2]);
                byte[] expected = HexFormat.of().parseHex(parts[3]);
                byte[] actual = pbkdf2(rawPassword.toCharArray(), salt, iterations, expected.length * 8);
                return MessageDigest.isEqual(actual, expected);
            } catch (IllegalArgumentException e) {
                return false;
            }
        }
        // Legacy unsalted SHA-256 verification fallback (caller upgrades to PBKDF2 on success).
        byte[] expectedBytes = storedHash.getBytes(StandardCharsets.UTF_8);
        byte[] actualBytes = sha256Hex(rawPassword).getBytes(StandardCharsets.UTF_8);
        return MessageDigest.isEqual(actualBytes, expectedBytes);
    }

    public static boolean isLegacySha256Hash(String storedHash) {
        return storedHash != null && !storedHash.startsWith(PBKDF2_PREFIX) && storedHash.length() == 64;
    }

    private static byte[] pbkdf2(char[] password, byte[] salt, int iterations, int keyBits) {
        PBEKeySpec spec = new PBEKeySpec(password, salt, iterations, keyBits);
        try {
            SecretKeyFactory skf = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256");
            return skf.generateSecret(spec).getEncoded();
        } catch (NoSuchAlgorithmException | InvalidKeySpecException e) {
            throw new IllegalStateException("PBKDF2WithHmacSHA256 not available", e);
        } finally {
            spec.clearPassword();
        }
    }

    public static String generateNodeEnrollmentToken() {
        return "syn_node_live_" + randomAlphaNumeric(32);
    }

    public static String generateOpenAiApiKey() {
        return "sk-cutmesh-" + randomAlphaNumeric(36);
    }

    public static String extractPrefix(String secret, int prefixLen) {
        if (secret == null || secret.length() <= prefixLen) {
            return secret;
        }
        return secret.substring(0, prefixLen) + "...";
    }

    public static String sha256Hex(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 algorithm not available", e);
        }
    }

    private static String randomAlphaNumeric(int length) {
        StringBuilder sb = new StringBuilder(length);
        for (int i = 0; i < length; i++) {
            sb.append(ALPHANUM.charAt(SECURE_RANDOM.nextInt(ALPHANUM.length())));
        }
        return sb.toString();
    }
}

