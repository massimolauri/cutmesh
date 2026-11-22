package io.cutmesh.controlplane.gateway;

import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface OpenAiApiKeyRepository extends JpaRepository<OpenAiApiKey, String> {
    Optional<OpenAiApiKey> findByKeySha256AndRevokedFalse(String keySha256);
    List<OpenAiApiKey> findByOwnerUserIdOrderByCreatedAtDesc(String ownerUserId);
}

