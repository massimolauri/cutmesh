package io.cutmesh.controlplane.audit;

import java.util.Optional;

/**
 * DIP port owned by `cutmesh-audit` allowing `SecurityAuditService` to resolve a catalog model's
 * base GGUF path without depending on `cutmesh-orchestration`.
 */
@FunctionalInterface
public interface ModelPathResolver {
    Optional<String> resolveBaseGgufPath(String modelId);
}
