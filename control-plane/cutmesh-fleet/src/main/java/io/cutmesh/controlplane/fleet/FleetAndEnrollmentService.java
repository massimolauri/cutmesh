package io.cutmesh.controlplane.fleet;

import io.cutmesh.controlplane.shared.ClusterLeaderLock;
import io.cutmesh.controlplane.shared.CryptoHashUtils;
import io.cutmesh.controlplane.audit.SecurityAuditService;
import io.cutmesh.controlplane.iam.UserAccountService;
import io.cutmesh.controlplane.iam.AppUser;

import io.cutmesh.controlplane.fleet.WorkerAndEnrollmentDtos.CreateEnrollmentTokenRequest;
import io.cutmesh.controlplane.fleet.WorkerAndEnrollmentDtos.EnrollmentTokenIssuedResponse;
import io.cutmesh.controlplane.fleet.WorkerAndEnrollmentDtos.RustWorkerHeartbeatRequest;
import io.cutmesh.controlplane.fleet.WorkerAndEnrollmentDtos.RustWorkerHeartbeatResponse;

import jakarta.persistence.EntityNotFoundException;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Handles GPU Provider enrollment tokens (`syn_node_live_...`), Docker connection command
 * generation, real-time silicon telemetry heartbeats, immediate disconnect detection,
 * and live Self-Healing Watchdog testing.
 */
@Service
public class FleetAndEnrollmentService {

    private final NodeEnrollmentTokenRepository tokenRepository;
    private final GpuWorkerNodeRepository workerRepository;
    private final UserAccountService userAccountService;
    private final SecurityAuditService auditService;
    private final RigOrchestrationPort orchestratorPort;
    private final ClusterLeaderLock clusterLeaderLock;
    private final String publicHubUrl;
    private final long heartbeatTimeoutSeconds;
    private final String workerDockerImage;
    private final Instant serviceStartedAt = Instant.now();

    public FleetAndEnrollmentService(
        NodeEnrollmentTokenRepository tokenRepository,
        GpuWorkerNodeRepository workerRepository,
        UserAccountService userAccountService,
        SecurityAuditService auditService,
        RigOrchestrationPort orchestratorPort,
        ClusterLeaderLock clusterLeaderLock,
        @Value("${cutmesh.control-plane.public-hub-url:http://localhost:8085}") String publicHubUrl,
        @Value("${cutmesh.control-plane.worker-heartbeat-timeout-seconds:30}") long heartbeatTimeoutSeconds,
        @Value("${cutmesh.worker.docker-image:ghcr.io/massimolauri/cutmesh-worker:latest}") String workerDockerImage
    ) {
        this.tokenRepository = tokenRepository;
        this.workerRepository = workerRepository;
        this.userAccountService = userAccountService;
        this.auditService = auditService;
        this.orchestratorPort = orchestratorPort;
        this.clusterLeaderLock = clusterLeaderLock;
        this.publicHubUrl = publicHubUrl;
        this.heartbeatTimeoutSeconds = heartbeatTimeoutSeconds;
        this.workerDockerImage = workerDockerImage;
    }

    @Transactional
    public EnrollmentTokenIssuedResponse issueEnrollmentToken(String userId, CreateEnrollmentTokenRequest request) {
        if (userId == null || userId.isBlank()) {
            throw new org.springframework.web.server.ResponseStatusException(
                org.springframework.http.HttpStatus.UNAUTHORIZED,
                "Utente non autenticato."
            );
        }

        String rawSecret = CryptoHashUtils.generateNodeEnrollmentToken();
        String prefix = CryptoHashUtils.extractPrefix(rawSecret, 18);
        String sha256 = CryptoHashUtils.sha256Hex(rawSecret);

        NodeEnrollmentToken entity = new NodeEnrollmentToken(
            userId,
            request.label(),
            prefix,
            sha256,
            request.defaultPoolMode()
        );
        NodeEnrollmentToken saved = tokenRepository.save(entity);

        auditService.recordEvent(
            "WORKER_ENROLLMENT_TOKEN",
            "INFO",
            "Issued GPU Enrollment Token '" + saved.getLabel() + "' (" + prefix + ")",
            "Generato Token di Arruolamento GPU '" + saved.getLabel() + "' (" + prefix + ")",
            "Created SHA-256 hashed enrollment token for pool mode " + saved.getDefaultPoolMode() + ".",
            "Creato token di arruolamento con hash SHA-256 per modalità pool " + saved.getDefaultPoolMode() + ".",
            saved.getId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );

        return buildIssuedResponse(saved, rawSecret);
    }

    @Transactional
    public void revokeEnrollmentToken(String tokenId, String callerUserId, boolean isAdmin) {
        NodeEnrollmentToken token = tokenRepository.findById(tokenId)
            .orElseThrow(() -> new EntityNotFoundException("Enrollment token not found: " + tokenId));
        if (!isAdmin && callerUserId != null && !callerUserId.equals(token.getOwnerUserId())) {
            throw new org.springframework.web.server.ResponseStatusException(
                org.springframework.http.HttpStatus.FORBIDDEN,
                "Non sei autorizzato a revocare questo token di arruolamento."
            );
        }
        token.setRevoked(true);
        tokenRepository.save(token);

        auditService.recordEvent(
            "WORKER_ENROLLMENT_TOKEN",
            "WARNING",
            "Revoked GPU Enrollment Token '" + token.getLabel() + "'",
            "Revocato Token di Arruolamento GPU '" + token.getLabel() + "'",
            "Token " + token.getTokenPrefix() + " was permanently revoked.",
            "Il token " + token.getTokenPrefix() + " è stato revocato permanentemente.",
            token.getId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );
    }

    /**
     * Live Chaos Engineering action: kills the local `ggml-rpc-server` process so the operator
     * can observe the Rust 2-second Self-Healing Watchdog respawn it in real time.
     */
    @Transactional
    public Map<String, Object> triggerWatchdogFaultTest(String workerId, String callerUserId, boolean isAdmin) {
        GpuWorkerNode worker = workerRepository.findById(workerId)
            .orElseThrow(() -> new EntityNotFoundException("GPU Worker not found: " + workerId));
        if (!isAdmin && callerUserId != null && !callerUserId.equals(worker.getOwnerUserId())) {
            throw new org.springframework.web.server.ResponseStatusException(
                org.springframework.http.HttpStatus.FORBIDDEN,
                "Non sei autorizzato a eseguire fault test su questo worker."
            );
        }

        boolean processKilled = false;
        try {
            boolean isWindows = System.getProperty("os.name", "").toLowerCase().contains("win");
            ProcessBuilder pb = isWindows
                ? new ProcessBuilder("taskkill", "/F", "/IM", "ggml-rpc-server.exe")
                : new ProcessBuilder("pkill", "-9", "-f", "ggml-rpc-server");
            Process p = pb.start();
            processKilled = p.waitFor(5, TimeUnit.SECONDS) && p.exitValue() == 0;
        } catch (Exception ignored) {
        }

        auditService.recordEvent(
            "SELF_HEALING_WATCHDOG",
            "WARNING",
            "Injected Chaos Fault on " + worker.getGpuName() + " (Killed ggml-rpc-server)",
            "Iniettato Fault Test su " + worker.getGpuName() + " (Terminato ggml-rpc-server)",
            "Terminated ggml-rpc-server child process to verify <2s Rust Self-Healing Watchdog recovery while keeping Iroh ticket active.",
            "Terminato il processo figlio ggml-rpc-server per verificare il ripristino in <2s del Self-Healing Watchdog mantenendo attivo il ticket Iroh.",
            worker.getId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );

        return Map.of(
            "workerId", worker.getId(),
            "gpuName", worker.getGpuName(),
            "rpcProcessTerminated", processKilled,
            "expectedRecoveryMs", 2000
        );
    }

    public EnrollmentTokenIssuedResponse buildIssuedResponse(NodeEnrollmentToken token, String rawSecret) {
        String poolModeEnv = token.getDefaultPoolMode() == PoolMode.PRIVATE_SOLO ? "PRIVATE_SOLO" : "SHARED_POOL";
        String poolFlag = token.getDefaultPoolMode() == PoolMode.PRIVATE_SOLO ? "private" : "shared";
        String safeNodeName = token.getLabel().replaceAll("[^a-zA-Z0-9_-]", "-");
        if (safeNodeName.isBlank()) {
            safeNodeName = "cutmesh-gpu-node";
        }

        String dockerLinux = String.format(
            "docker run -d --gpus all --restart unless-stopped --name %s -p 40000:40000/udp "
                + "-v cutmesh-cache:/data/cutmesh "
                + "-e CUTMESH_QUIC_PORT=40000 -e CUTMESH_HUB_URL=\"%s\" -e CUTMESH_NODE_TOKEN=\"%s\" "
                + "-e CUTMESH_NODE_NAME=\"%s\" -e CUTMESH_POOL_MODE=\"%s\" -e CUTMESH_POWER_PCT=100 -e CUTMESH_STRICT_DIRECT=1 "
                + "%s",
            safeNodeName, publicHubUrl, rawSecret, safeNodeName, poolModeEnv, workerDockerImage
        );

        String dockerWsl2 = String.format(
            "docker run -d --gpus all --restart unless-stopped --name %s -p 40000:40000/udp "
                + "-v cutmesh-cache:/data/cutmesh "
                + "-e CUTMESH_QUIC_PORT=40000 -e CUTMESH_HUB_URL=\"%s\" -e CUTMESH_NODE_TOKEN=\"%s\" "
                + "-e CUTMESH_NODE_NAME=\"%s\" -e CUTMESH_POOL_MODE=\"%s\" -e CUTMESH_POWER_PCT=100 -e CUTMESH_STRICT_DIRECT=1 "
                + "%s",
            safeNodeName, publicHubUrl, rawSecret, safeNodeName, poolModeEnv, workerDockerImage
        );

        String nativeCli = String.format(
            "./cutmesh worker --name %s --hub-url %s --node-token %s --pool-mode %s --wan-dht",
            safeNodeName, publicHubUrl, rawSecret, poolFlag
        );

        return new EnrollmentTokenIssuedResponse(
            token.getId(),
            token.getLabel(),
            rawSecret,
            token.getTokenPrefix(),
            token.getDefaultPoolMode(),
            dockerLinux,
            dockerWsl2,
            nativeCli,
            token.getCreatedAt()
        );
    }

    @Transactional(readOnly = true)
    public List<NodeEnrollmentToken> listTokensForUser(String userId, boolean isAdmin) {
        if (isAdmin) {
            return tokenRepository.findAll();
        }
        return tokenRepository.findByOwnerUserIdOrderByCreatedAtDesc(userId);
    }

    @Transactional(readOnly = true)
    public List<GpuWorkerNode> listAllWorkers() {
        return workerRepository.findAll();
    }

    @Transactional(readOnly = true)
    public List<GpuWorkerNode> listStandbyWorkers() {
        return workerRepository.findByAssignmentStateOrderByMaxAllocatableVramMbDesc(
            WorkerAssignmentState.IDLE_STANDBY_POOL
        );
    }

    private NodeEnrollmentToken requireValidWorkerToken(String bearerHeader) {
        if (bearerHeader == null || !bearerHeader.startsWith("Bearer ")) {
            throw new org.springframework.web.server.ResponseStatusException(
                org.springframework.http.HttpStatus.UNAUTHORIZED,
                "Missing Authorization: Bearer syn_node_live_... worker token"
            );
        }
        String rawToken = bearerHeader.substring("Bearer ".length()).trim();
        if (rawToken.isBlank() || !rawToken.startsWith("syn_node_live_")) {
            throw new org.springframework.web.server.ResponseStatusException(
                org.springframework.http.HttpStatus.UNAUTHORIZED,
                "Invalid worker token format"
            );
        }
        String hash = CryptoHashUtils.sha256Hex(rawToken);
        NodeEnrollmentToken token = tokenRepository.findByTokenSha256AndRevokedFalse(hash)
            .orElseThrow(() -> new org.springframework.web.server.ResponseStatusException(
                org.springframework.http.HttpStatus.UNAUTHORIZED,
                "Invalid or revoked worker enrollment token"
            ));
        token.setLastUsedAt(Instant.now());
        return tokenRepository.save(token);
    }

    /**
     * Processes the live 2-second heartbeat sent by the Rust GPU Worker (`src/discovery.rs`).
     */
    @Transactional
    public RustWorkerHeartbeatResponse processWorkerHeartbeat(String bearerHeader, RustWorkerHeartbeatRequest req) {
        NodeEnrollmentToken token = requireValidWorkerToken(bearerHeader);
        String ownerUserId = token.getOwnerUserId();
        String enrollmentTokenId = token.getId();

        PoolMode parsedPoolMode = "PRIVATE_SOLO".equalsIgnoreCase(req.poolMode()) || "private".equalsIgnoreCase(req.poolMode())
            ? PoolMode.PRIVATE_SOLO
            : PoolMode.SHARED_POOL;

        Optional<GpuWorkerNode> existingOpt = workerRepository.findByIrohNodeId(req.irohNodeId());
        if (existingOpt.isPresent()) {
            GpuWorkerNode existing = existingOpt.get();
            if (existing.getOwnerUserId() != null && !existing.getOwnerUserId().equals(ownerUserId)) {
                throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.FORBIDDEN,
                    "Worker node belongs to a different enrollment owner"
                );
            }
        }
        boolean isNewNode = existingOpt.isEmpty();
        boolean wasOffline = existingOpt.map(w -> w.getAssignmentState() == WorkerAssignmentState.OFFLINE_STALE).orElse(false);
        long previousRestarts = existingOpt.map(GpuWorkerNode::getWatchdogRestarts).orElse(0L);

        GpuWorkerNode worker = existingOpt.orElseGet(GpuWorkerNode::new);
        worker.setOwnerUserId(ownerUserId);
        if (enrollmentTokenId != null) {
            worker.setEnrollmentTokenId(enrollmentTokenId);
        }
        if (req.nodeName() != null && !req.nodeName().isBlank()) {
            worker.setNodeName(req.nodeName());
        }
        if (req.localRpcPort() != null && req.localRpcPort() > 0) {
            worker.setLocalRpcPort(req.localRpcPort());
        }
        if (req.device() != null && !req.device().isBlank()) {
            worker.setDevice(req.device());
        }
        if (req.effectiveTflops() != null) {
            worker.setEffectiveTflops(req.effectiveTflops());
        }
        if (req.effectiveBandwidthGbS() != null) {
            worker.setEffectiveBandwidthGbS(req.effectiveBandwidthGbS());
        }
        if (req.allocatedPowerWatts() != null) {
            worker.setAllocatedPowerWatts(req.allocatedPowerWatts());
        }
        if (req.powerBudgetPct() != null) {
            worker.setPowerBudgetPct(req.powerBudgetPct());
        }
        if (req.temperatureC() != null) {
            worker.setTemperatureC(req.temperatureC());
        }
        if (req.gpuUtilizationPct() != null) {
            worker.setGpuUtilizationPct(req.gpuUtilizationPct());
        }

        // Self-heal orphan LOCKED_IN_RIG workers whose Rig was dissolved or no longer exists
        boolean wasLockedInRig = worker.getAssignmentState() == WorkerAssignmentState.LOCKED_IN_RIG;
        if (wasLockedInRig) {
            if (!orchestratorPort.isRigCurrentlyServing(worker.getAssignedRigId())) {
                worker.setAssignmentState(WorkerAssignmentState.IDLE_STANDBY_POOL);
                worker.setAssignedRigId(null);
                worker.setAssignedModelId(null);
                worker.setAssignedRole(null);
            }
        }

        if (worker.getAssignmentState() == null
            || worker.getAssignmentState() == WorkerAssignmentState.OFFLINE_STALE
            || worker.getAssignmentState() == WorkerAssignmentState.DRAINING) {
            worker.setAssignmentState(WorkerAssignmentState.IDLE_STANDBY_POOL);
            worker.setAssignedRigId(null);
            worker.setAssignedModelId(null);
            worker.setAssignedRole(null);
        }

        worker.setIrohNodeId(req.irohNodeId());
        worker.setIrohTicket(req.irohTicket());
        worker.setGpuName(req.gpuName());
        worker.setFreeVramMb(req.freeVramMb());
        worker.setTotalVramMb(req.totalVramMb());
        if (wasLockedInRig || worker.getAssignmentState() == WorkerAssignmentState.LOCKED_IN_RIG) {
            // While serving a Rig, live free VRAM is reduced by the loaded model shard; preserve full allocatable capacity
            worker.setMaxAllocatableVramMb(Math.max(worker.getMaxAllocatableVramMb(), req.maxAllocatableVramMb()));
        } else {
            worker.setMaxAllocatableVramMb(req.maxAllocatableVramMb());
        }
        worker.setPoolMode(parsedPoolMode);
        worker.setRpcHealthy(req.rpcHealthy());
        worker.setWatchdogRestarts(req.watchdogRestarts());
        worker.setVersion(req.version());
        worker.setLastHeartbeatAt(Instant.now());

        GpuWorkerNode saved = workerRepository.save(worker);

        if (isNewNode || wasOffline) {
            auditService.recordEvent(
                "WORKER_ENROLLMENT_AND_HEARTBEAT",
                "VERIFIED",
                "GPU Worker Connected via Iroh P2P: " + saved.getNodeName() + " (" + saved.getGpuName() + ", " + saved.getFreeVramMb() + " MB Free)",
                "GPU Worker Connesso via Iroh P2P: " + saved.getNodeName() + " (" + saved.getGpuName() + ", " + saved.getFreeVramMb() + " MB Liberi)",
                "Node " + shortenNodeId(saved.getIrohNodeId()) + " (" + saved.getNodeName() + ") joined " + saved.getPoolMode()
                    + " with " + saved.getEffectiveTflops() + " TFLOPS, " + saved.getEffectiveBandwidthGbS() + " GB/s.",
                "Il nodo " + shortenNodeId(saved.getIrohNodeId()) + " (" + saved.getNodeName() + ") è entrato in " + saved.getPoolMode()
                    + " con " + saved.getEffectiveTflops() + " TFLOPS, " + saved.getEffectiveBandwidthGbS() + " GB/s.",
                saved.getId(),
                true,
                null,
                null,
                null,
                null,
                null,
                null
            );
        } else if (req.watchdogRestarts() > previousRestarts) {
            auditService.recordEvent(
                "SELF_HEALING_WATCHDOG",
                "VERIFIED",
                "Self-Healing Watchdog Recovered ggml-rpc-server on " + saved.getGpuName() + " (Restart #" + req.watchdogRestarts() + ")",
                "Self-Healing Watchdog ha ripristinato ggml-rpc-server su " + saved.getGpuName() + " (Riavvio #" + req.watchdogRestarts() + ")",
                "Worker " + shortenNodeId(saved.getIrohNodeId()) + " auto-restarted crashed RPC daemon on port " + saved.getLocalRpcPort() + " while preserving Iroh P2P ticket.",
                "Il worker " + shortenNodeId(saved.getIrohNodeId()) + " ha riavviato automaticamente il demone RPC sulla porta " + saved.getLocalRpcPort() + " mantenendo attivo il ticket P2P Iroh.",
                saved.getId(),
                true,
                null,
                null,
                null,
                null,
                null,
                null
            );
        }

        RigOrchestrationPort.WorkerShardAssignment shard =
            orchestratorPort.resolveWorkerShardAssignment(saved);

        return new RustWorkerHeartbeatResponse(
            true,
            saved.getAssignmentState().name(),
            saved.getAssignedRigId(),
            saved.getAssignedModelId(),
            saved.getAssignedRole(),
            shard != null ? shard.modelUrl() : null,
            shard != null ? shard.layerStart() : null,
            shard != null ? shard.layerEndInclusive() : null,
            2L
        );
    }

    /**
     * Immediate disconnect handler called by a Rust worker upon graceful shutdown (`Ctrl+C`)
     * or by the operator from the Web Dashboard.
     */
    @Transactional
    public GpuWorkerNode markWorkerDisconnectedImmediately(GpuWorkerNode worker, String reason) {
        String previousRigId = worker.getAssignedRigId();
        worker.setAssignmentState(WorkerAssignmentState.OFFLINE_STALE);
        worker.setRpcHealthy(false);
        worker.setAssignedRigId(null);
        worker.setAssignedModelId(null);
        worker.setAssignedRole(null);
        worker.setLastHeartbeatAt(Instant.now().minusSeconds(heartbeatTimeoutSeconds + 5));
        GpuWorkerNode saved = workerRepository.save(worker);

        auditService.recordEvent(
            "WORKER_DISCONNECTED",
            "WARNING",
            "GPU Node Disconnected: " + saved.getNodeName() + " (" + saved.getGpuName() + ")",
            "Nodo GPU Disconnesso: " + saved.getNodeName() + " (" + saved.getGpuName() + ")",
            "Worker " + saved.getNodeName() + " went OFFLINE_STALE (" + reason + ").",
            "Il worker " + saved.getNodeName() + " è passato in stato OFFLINE_STALE (" + reason + ").",
            saved.getId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );

        if (previousRigId != null && !previousRigId.isBlank()) {
            orchestratorPort.handleWorkerDisconnectedFromRig(previousRigId, saved);
        }
        return saved;
    }

    @Transactional
    public Map<String, Object> disconnectWorkerByIrohNodeId(String authorizationHeader, String irohNodeId) {
        NodeEnrollmentToken token = requireValidWorkerToken(authorizationHeader);
        Optional<GpuWorkerNode> opt = workerRepository.findByIrohNodeId(irohNodeId);
        if (opt.isPresent()) {
            GpuWorkerNode existing = opt.get();
            if (existing.getOwnerUserId() != null && !existing.getOwnerUserId().equals(token.getOwnerUserId())) {
                throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.FORBIDDEN,
                    "Worker ownership mismatch for enrollment token"
                );
            }
            GpuWorkerNode w = markWorkerDisconnectedImmediately(existing, "graceful worker shutdown signal");
            return Map.of("disconnected", true, "workerId", w.getId(), "nodeName", w.getNodeName());
        }
        return Map.of("disconnected", false);
    }

    @Transactional
    public GpuWorkerNode disconnectWorkerById(String workerId, String callerUserId, boolean isAdmin) {
        GpuWorkerNode worker = workerRepository.findById(workerId)
            .orElseThrow(() -> new EntityNotFoundException("GPU Worker not found: " + workerId));
        if (!isAdmin && callerUserId != null && !callerUserId.equals(worker.getOwnerUserId())) {
            throw new org.springframework.web.server.ResponseStatusException(
                org.springframework.http.HttpStatus.FORBIDDEN,
                "Non sei autorizzato a disconnettere questo worker."
            );
        }
        return markWorkerDisconnectedImmediately(worker, "manual operator disconnect");
    }

    @Transactional
    public void deleteWorkerNode(String workerId, String callerUserId, boolean isAdmin) {
        GpuWorkerNode worker = workerRepository.findById(workerId)
            .orElseThrow(() -> new EntityNotFoundException("GPU Worker not found: " + workerId));
        if (!isAdmin && callerUserId != null && !callerUserId.equals(worker.getOwnerUserId())) {
            throw new org.springframework.web.server.ResponseStatusException(
                org.springframework.http.HttpStatus.FORBIDDEN,
                "Non sei autorizzato a eliminare questo worker."
            );
        }
        String previousRigId = worker.getAssignedRigId();
        if (previousRigId != null && !previousRigId.isBlank()) {
            orchestratorPort.handleWorkerDisconnectedFromRig(previousRigId, worker);
        }
        workerRepository.delete(worker);
    }

    /**
     * Scans every 2 seconds for any worker (including LOCKED_IN_RIG workers!) whose heartbeat
     * is older than `heartbeatTimeoutSeconds` (8s) and immediately transitions it to OFFLINE_STALE
     * while healing any affected Rig.
     */
    @Scheduled(fixedDelayString = "2000", initialDelayString = "15000")
    @Transactional
    public void evictStaleWorkers() {
        if (Duration.between(serviceStartedAt, Instant.now()).getSeconds() < heartbeatTimeoutSeconds) {
            return;
        }
        if (!clusterLeaderLock.tryAcquireTransactionLock("cutmesh:fleet:evict-stale-workers")) {
            return;
        }
        Instant cutoff = Instant.now().minus(Duration.ofSeconds(heartbeatTimeoutSeconds));
        List<GpuWorkerNode> all = workerRepository.findAll();
        for (GpuWorkerNode w : all) {
            if (w.getAssignmentState() != WorkerAssignmentState.OFFLINE_STALE
                && w.getLastHeartbeatAt() != null
                && w.getLastHeartbeatAt().isBefore(cutoff)) {
                markWorkerDisconnectedImmediately(w, "heartbeat timeout > " + heartbeatTimeoutSeconds + "s");
            }
        }
    }

    private static String shortenNodeId(String nodeId) {
        if (nodeId == null || nodeId.length() <= 16) {
            return nodeId;
        }
        return nodeId.substring(0, 16) + "...";
    }
}
