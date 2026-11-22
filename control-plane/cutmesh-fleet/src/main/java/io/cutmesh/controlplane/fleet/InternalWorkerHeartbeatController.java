package io.cutmesh.controlplane.fleet;

import io.cutmesh.controlplane.fleet.WorkerAndEnrollmentDtos.RustWorkerHeartbeatRequest;
import io.cutmesh.controlplane.fleet.WorkerAndEnrollmentDtos.RustWorkerHeartbeatResponse;

import jakarta.validation.Valid;
import java.util.Map;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Internal endpoint receiving 2-second heartbeats and immediate disconnect notifications
 * from the Rust GPU Worker (`src/discovery.rs`).
 */
@RestController
@RequestMapping("/api/v1/internal/workers")
public class InternalWorkerHeartbeatController {

    private final FleetAndEnrollmentService fleetService;
    private final RigOrchestrationPort orchestratorPort;

    public InternalWorkerHeartbeatController(
        FleetAndEnrollmentService fleetService,
        RigOrchestrationPort orchestratorPort
    ) {
        this.fleetService = fleetService;
        this.orchestratorPort = orchestratorPort;
    }

    @PostMapping("/heartbeat")
    public RustWorkerHeartbeatResponse receiveWorkerHeartbeat(
        @RequestHeader(value = "Authorization", required = false) String authorizationHeader,
        @Valid @RequestBody RustWorkerHeartbeatRequest request
    ) {
        RustWorkerHeartbeatResponse initial = fleetService.processWorkerHeartbeat(authorizationHeader, request);
        if ("IDLE_STANDBY_POOL".equals(initial.assignmentState())) {
            int provisioned = orchestratorPort.reconcileAutoPilotRigs();
            if (provisioned > 0) {
                return fleetService.processWorkerHeartbeat(authorizationHeader, request);
            }
        }
        return initial;
    }

    @PostMapping("/disconnect")
    public Map<String, Object> receiveWorkerDisconnect(
        @RequestHeader(value = "Authorization", required = false) String authorizationHeader,
        @RequestBody Map<String, String> payload
    ) {
        String irohNodeId = payload != null ? payload.get("iroh_node_id") : null;
        if (irohNodeId == null || irohNodeId.isBlank()) {
            throw new org.springframework.web.server.ResponseStatusException(
                org.springframework.http.HttpStatus.BAD_REQUEST,
                "Missing iroh_node_id"
            );
        }
        return fleetService.disconnectWorkerByIrohNodeId(authorizationHeader, irohNodeId);
    }
}
