package io.cutmesh.controlplane.orchestration;

public enum RigLifecycleState {
    PROVISIONING,
    CLOSED_AND_LOCKED,
    SERVING,
    DEGRADED,
    DISSOLVED
}
