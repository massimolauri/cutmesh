package io.cutmesh.controlplane.fleet;

import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface GpuWorkerNodeRepository extends JpaRepository<GpuWorkerNode, String> {
    Optional<GpuWorkerNode> findByIrohNodeId(String irohNodeId);
    List<GpuWorkerNode> findByOwnerUserIdOrderByLastHeartbeatAtDesc(String ownerUserId);
    List<GpuWorkerNode> findByAssignmentStateOrderByMaxAllocatableVramMbDesc(WorkerAssignmentState assignmentState);
    List<GpuWorkerNode> findByAssignmentStateAndPoolModeOrderByMaxAllocatableVramMbDesc(
        WorkerAssignmentState assignmentState,
        PoolMode poolMode
    );
    List<GpuWorkerNode> findByAssignedRigId(String assignedRigId);
}

