package io.cutmesh.controlplane.iam;

import java.util.Optional;

/**
 * DIP port owned by `cutmesh-iam` allowing `UserAccountService` to authenticate a consumer
 * API key (`sk-cutmesh-...`) without depending on `cutmesh-gateway`.
 */
@FunctionalInterface
public interface ApiKeyAuthLookup {
    Optional<String> resolveOwnerUserIdAndTouchBySha256(String sha256);
}
