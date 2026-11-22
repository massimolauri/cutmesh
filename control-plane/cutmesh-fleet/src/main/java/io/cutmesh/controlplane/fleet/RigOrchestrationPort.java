package io.cutmesh.controlplane.fleet;

/**
 * Dependency Inversion Principle (DIP) port owned by `cutmesh-fleet` and implemented by
 * `cutmesh-orchestration` (`HybridRigOrchestratorService`).
 * Breaks the cyclic dependency between fleet telemetry/enrollment and rig orchestration
 * so each domain compiles as an independent Maven module.
 */
public interface RigOrchestrationPort {

    record WorkerShardAssignment(String modelUrl, int layerStart, int layerEndInclusive) {
    }

    boolean isRigCurrentlyServing(String rigId);

    WorkerShardAssignment resolveWorkerShardAssignment(GpuWorkerNode worker);

    void handleWorkerDisconnectedFromRig(String rigId, GpuWorkerNode disconnectedWorker);

    int reconcileAutoPilotRigs();
}
