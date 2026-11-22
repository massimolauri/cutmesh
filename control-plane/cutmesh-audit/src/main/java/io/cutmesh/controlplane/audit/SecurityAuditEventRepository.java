package io.cutmesh.controlplane.audit;

import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface SecurityAuditEventRepository extends JpaRepository<SecurityAuditEvent, String> {
    List<SecurityAuditEvent> findTop100ByOrderByCreatedAtDesc();
    List<SecurityAuditEvent> findByCategoryOrderByCreatedAtDesc(String category);
}

