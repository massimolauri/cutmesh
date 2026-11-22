package io.cutmesh.controlplane.orchestration;

import java.util.List;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface ManagedRigRepository extends JpaRepository<ManagedRig, String> {
    List<ManagedRig> findByLifecycleStateOrderByCreatedAtDesc(RigLifecycleState lifecycleState);
    List<ManagedRig> findByModelIdAndLifecycleState(String modelId, RigLifecycleState lifecycleState);
    List<ManagedRig> findAllByOrderByCreatedAtDesc();

    long countByLifecycleState(RigLifecycleState lifecycleState);

    @Query("""
        SELECT r FROM ManagedRig r
        WHERE (:state IS NULL OR r.lifecycleState = :state)
          AND (:modelId IS NULL OR :modelId = '' OR r.modelId = :modelId)
          AND (:query IS NULL OR :query = ''
               OR LOWER(r.rigName) LIKE LOWER(CONCAT('%', :query, '%'))
               OR LOWER(r.modelDisplayName) LIKE LOWER(CONCAT('%', :query, '%'))
               OR LOWER(r.modelId) LIKE LOWER(CONCAT('%', :query, '%')))
        """)
    Page<ManagedRig> searchRigs(
        @Param("state") RigLifecycleState state,
        @Param("modelId") String modelId,
        @Param("query") String query,
        Pageable pageable
    );
}
