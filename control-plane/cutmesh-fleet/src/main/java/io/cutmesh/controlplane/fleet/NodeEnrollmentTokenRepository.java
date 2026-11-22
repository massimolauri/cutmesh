package io.cutmesh.controlplane.fleet;

import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface NodeEnrollmentTokenRepository extends JpaRepository<NodeEnrollmentToken, String> {
    Optional<NodeEnrollmentToken> findByTokenSha256AndRevokedFalse(String tokenSha256);
    List<NodeEnrollmentToken> findByOwnerUserIdOrderByCreatedAtDesc(String ownerUserId);
}

