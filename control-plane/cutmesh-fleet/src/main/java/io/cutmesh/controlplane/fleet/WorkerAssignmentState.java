package io.cutmesh.controlplane.fleet;

public enum WorkerAssignmentState {
    IDLE_STANDBY_POOL,
    LOCKED_IN_RIG,
    DRAINING,
    OFFLINE_STALE
}
