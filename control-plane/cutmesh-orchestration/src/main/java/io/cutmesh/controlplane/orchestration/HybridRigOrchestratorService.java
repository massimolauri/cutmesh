package io.cutmesh.controlplane.orchestration;

import io.cutmesh.controlplane.audit.ModelPathResolver;
import io.cutmesh.controlplane.audit.SecurityAuditService;
import io.cutmesh.controlplane.fleet.RigOrchestrationPort;
import io.cutmesh.controlplane.shared.ClusterLeaderLock;
import io.cutmesh.controlplane.shared.ClusterStateProvider;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import io.cutmesh.controlplane.fleet.GpuWorkerNode;

import io.cutmesh.controlplane.audit.SecurityAuditEvent;

import io.cutmesh.controlplane.fleet.WorkerAssignmentState;
import io.cutmesh.controlplane.orchestration.CatalogAndRigDtos.CompileCovariantRequest;
import io.cutmesh.controlplane.orchestration.CatalogAndRigDtos.CovariantCompileResponse;
import io.cutmesh.controlplane.orchestration.CatalogAndRigDtos.CreateCatalogModelRequest;
import io.cutmesh.controlplane.orchestration.CatalogAndRigDtos.ManualCreateRigRequest;
import io.cutmesh.controlplane.orchestration.CatalogAndRigDtos.UpdateModelOrchestrationRequest;

import io.cutmesh.controlplane.fleet.GpuWorkerNodeRepository;

import jakarta.persistence.EntityNotFoundException;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Hybrid Auto-Pilot + Manual Override Rig Orchestrator & Real Rust GGUF Inspector/Compiler Bridge.
 */
@Service
public class HybridRigOrchestratorService implements RigOrchestrationPort, ModelPathResolver {

    private static final Logger log = LoggerFactory.getLogger(HybridRigOrchestratorService.class);

    private final CatalogModelRepository catalogRepository;
    private final GpuWorkerNodeRepository workerRepository;
    private final ManagedRigRepository rigRepository;
    private final SecurityAuditService auditService;
    private final ObjectMapper objectMapper;
    private final RigPlacementPlanner placementPlanner;
    private final ClusterLeaderLock clusterLeaderLock;
    private final String podAdvertiseHost;
    private final ConcurrentHashMap<String, Process> activeRigProcesses = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Long> rigRuntimeEpoch = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Boolean> rigLaunchPending = new ConcurrentHashMap<>();
    private final Map<String, String> rigFsmStateCodeById;
    private final Map<String, String> rigLastLogById;

    private RigPlacementPlanner.DbModelShardingPolicy parseDbModelPolicy(CatalogModel model) {
        return placementPlanner.parseDbModelPolicy(model);
    }

    private int snapPolicyCutLayer(
        int rawEndExcl,
        int minEndExcl,
        int maxEndExcl,
        RigPlacementPlanner.DbModelShardingPolicy policy
    ) {
        return placementPlanner.snapPolicyCutLayer(rawEndExcl, minEndExcl, maxEndExcl, policy);
    }

    public HybridRigOrchestratorService(
        CatalogModelRepository catalogRepository,
        GpuWorkerNodeRepository workerRepository,
        ManagedRigRepository rigRepository,
        SecurityAuditService auditService,
        ObjectMapper objectMapper,
        RigPlacementPlanner placementPlanner,
        ClusterLeaderLock clusterLeaderLock,
        ClusterStateProvider clusterStateProvider,
        @Value("${cutmesh.control-plane.pod-advertise-host:127.0.0.1}") String podAdvertiseHost
    ) {
        this.catalogRepository = catalogRepository;
        this.workerRepository = workerRepository;
        this.rigRepository = rigRepository;
        this.auditService = auditService;
        this.objectMapper = objectMapper;
        this.placementPlanner = placementPlanner;
        this.clusterLeaderLock = clusterLeaderLock;
        this.podAdvertiseHost = (podAdvertiseHost == null || podAdvertiseHost.isBlank()) ? "127.0.0.1" : podAdvertiseHost.trim();
        this.rigFsmStateCodeById = clusterStateProvider.getMap("cutmesh-rig-fsm-state");
        this.rigLastLogById = clusterStateProvider.getMap("cutmesh-rig-last-log");
    }

    @Override
    @Transactional(readOnly = true)
    public java.util.Optional<String> resolveBaseGgufPath(String modelId) {
        return catalogRepository.findById(modelId).map(CatalogModel::getBaseGgufPath);
    }

    @Transactional(readOnly = true)
    public List<CatalogModel> listCatalogModels() {
        List<CatalogModel> models = new ArrayList<>(catalogRepository.findAllByOrderByPriorityOrderAsc());
        models.sort(this::compareModelPriority);
        return models;
    }

    @Transactional(readOnly = true)
    public JsonNode getModelPolicy(String modelId) {
        CatalogModel model = catalogRepository.findById(modelId)
            .orElseThrow(() -> new EntityNotFoundException("Modello non trovato: " + modelId));
        try {
            if (model.getPolicyJson() != null && !model.getPolicyJson().isBlank()) {
                return objectMapper.readTree(model.getPolicyJson());
            }
        } catch (Exception ignored) {
        }
        return objectMapper.createObjectNode();
    }

    @Transactional
    public CatalogModel updateModelPolicy(String modelId, JsonNode policyNode) {
        CatalogModel model = catalogRepository.findById(modelId)
            .orElseThrow(() -> new EntityNotFoundException("Modello non trovato: " + modelId));
        try {
            String pretty = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(policyNode);
            model.setPolicyJson(pretty);
            JsonNode eng = policyNode.path("engine");
            if (!eng.isMissingNode() && eng.has("max_context_length")) {
                model.setMaxContextLength(eng.path("max_context_length").asInt(262144));
            }
            JsonNode sh = policyNode.path("sharding");
            if (!sh.isMissingNode() && sh.has("kv_vram_per_1k_tokens_mb")) {
                model.setKvVramPer1kTokensMb(sh.path("kv_vram_per_1k_tokens_mb").asLong(model.getKvVramPer1kTokensMb()));
            }
        } catch (Exception e) {
            throw new IllegalArgumentException("JSON policy non valido: " + e.getMessage());
        }
        return catalogRepository.save(model);
    }

    @Transactional
    public void deleteCatalogModel(String modelId) {
        if (!catalogRepository.existsById(modelId)) {
            throw new EntityNotFoundException("Modello non trovato: " + modelId);
        }
        List<ManagedRig> activeForModel = rigRepository.findByModelIdAndLifecycleState(
            modelId,
            RigLifecycleState.SERVING
        );
        for (ManagedRig rig : activeForModel) {
            dissolveRig(rig.getId());
        }
        catalogRepository.deleteById(modelId);
        reconcileAutoPilotRigs();
    }

    @Transactional(readOnly = true)
    public List<ManagedRig> listAllRigs() {
        return rigRepository.findAllByOrderByCreatedAtDesc();
    }

    @Transactional(readOnly = true)
    public List<ManagedRig> listActiveServingRigs() {
        return rigRepository.findByLifecycleStateOrderByCreatedAtDesc(RigLifecycleState.SERVING);
    }

    @Transactional(readOnly = true)
    public List<ManagedRig> listServingRigs() {
        return listActiveServingRigs();
    }

    @Transactional(readOnly = true)
    public long countServingRigs() {
        return rigRepository.countByLifecycleState(RigLifecycleState.SERVING);
    }

    @Transactional(readOnly = true)
    public org.springframework.data.domain.Page<ManagedRig> searchRigsPaged(
        String stateRaw,
        String modelId,
        String query,
        int page,
        int size
    ) {
        RigLifecycleState state = null;
        if (stateRaw != null && !stateRaw.isBlank() && !"ALL".equalsIgnoreCase(stateRaw.trim())) {
            try {
                state = RigLifecycleState.valueOf(stateRaw.trim().toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException ignored) {
                state = RigLifecycleState.SERVING;
            }
        }
        org.springframework.data.domain.Pageable pageable = org.springframework.data.domain.PageRequest.of(
            page,
            size,
            org.springframework.data.domain.Sort.by(org.springframework.data.domain.Sort.Direction.DESC, "createdAt")
        );
        String cleanModelId = (modelId == null || modelId.isBlank() || "ALL".equalsIgnoreCase(modelId.trim()))
            ? null
            : modelId.trim();
        String cleanQuery = (query == null || query.isBlank()) ? null : query.trim();
        return rigRepository.searchRigs(state, cleanModelId, cleanQuery, pageable);
    }

    /**
     * Registers a new Model in the Catalog and inspects its real `.gguf` file on disk via
     * `cutmesh inspect-gguf` if present.
     */
    @Transactional
    public CatalogModel registerNewCatalogModel(CreateCatalogModelRequest req) {
        CatalogModel model = new CatalogModel();
        model.setModelId(req.modelId().trim());
        model.setDisplayName(req.displayName().trim());
        model.setHuggingfaceRepo(req.huggingfaceRepo().trim());
        model.setBaseGgufPath(req.baseGgufPath().trim());

        String covPath = req.baseGgufPath().endsWith(".gguf")
            ? req.baseGgufPath().substring(0, req.baseGgufPath().length() - 5) + "-covariant.gguf"
            : req.baseGgufPath() + "-covariant.gguf";
        model.setCovariantGgufPath(covPath);
        model.setSecurityProfile(req.securityProfile());
        model.setOrchestrationMode(req.orchestrationMode());
        model.setMinActiveRigs(req.minActiveRigs());
        model.setMaxActiveRigs(req.maxActiveRigs());
        model.setPriorityOrder(req.priorityOrder());

        long requiredVramMb = 4096L;
        int totalLayers = 32;
        long kvPer1k = 48L;

        File rustExe = auditService.resolveRustBinary();
        File ggufFile = auditService.resolveModelFile(req.baseGgufPath());
        if (ggufFile != null && ggufFile.exists()) {
            long fileSizeMb = Math.max(256L, ggufFile.length() / (1024L * 1024L));
            requiredVramMb = (long) Math.ceil(fileSizeMb * 1.25);

            if (rustExe != null && rustExe.exists()) {
                try {
                    ProcessBuilder pb = new ProcessBuilder(
                        rustExe.getAbsolutePath(),
                        "inspect-gguf",
                        "--path", ggufFile.getAbsolutePath(),
                        "--vram-mb", "8188"
                    );
                    pb.redirectErrorStream(true);
                    Process process = pb.start();
                    StringBuilder out = new StringBuilder();
                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            out.append(line).append('\n');
                        }
                    }
                    if (process.waitFor(15, TimeUnit.SECONDS) && process.exitValue() == 0) {
                        String s = out.toString();
                        int jStart = s.indexOf('{');
                        int jEnd = s.lastIndexOf('}');
                        if (jStart >= 0 && jEnd > jStart) {
                            JsonNode root = objectMapper.readTree(s.substring(jStart, jEnd + 1));
                            totalLayers = root.path("total_layers").asInt(totalLayers);
                            if (root.has("catalog_entry") && !root.get("catalog_entry").isNull()) {
                                JsonNode cat = root.get("catalog_entry");
                                requiredVramMb = Math.round(cat.path("total_weights_vram_gb").asDouble(fileSizeMb / 1024.0) * 1024.0 * 1.25);
                            }
                        }
                    }
                } catch (Exception e) {
                    log.warn("GGUF inspection fallback: {}", e.getMessage());
                }
            }
        } else if (rustExe != null && rustExe.exists() && req.huggingfaceRepo() != null && req.huggingfaceRepo().startsWith("http")) {
            try {
                ProcessBuilder pb = new ProcessBuilder(
                    rustExe.getAbsolutePath(),
                    "inspect-gguf",
                    "--path", req.huggingfaceRepo().trim(),
                    "--vram-mb", "32768"
                );
                pb.redirectErrorStream(true);
                Process process = pb.start();
                StringBuilder out = new StringBuilder();
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        out.append(line).append('\n');
                    }
                }
                if (process.waitFor(20, TimeUnit.SECONDS) && process.exitValue() == 0) {
                    String s = out.toString();
                    int jStart = s.indexOf('{');
                    int jEnd = s.lastIndexOf('}');
                    if (jStart >= 0 && jEnd > jStart) {
                        JsonNode root = objectMapper.readTree(s.substring(jStart, jEnd + 1));
                        totalLayers = root.path("total_layers").asInt(totalLayers);
                        requiredVramMb = root.path("target_rig_vram_mb").asLong(requiredVramMb);
                    }
                }
            } catch (Exception e) {
                log.warn("Remote HTTP Range GGUF inspection fallback: {}", e.getMessage());
            }
        }

        model.setRequiredVramMb(Math.max(512L, requiredVramMb));
        model.setTotalLayers(totalLayers);
        model.setKvVramPer1kTokensMb(kvPer1k);
        model.setCovariantCompiled(false);
        model.setLastCompileMs(null);
        model.setLastCompileThroughputMbPerSec(null);
        model.setTransformedQ50Rows(null);
        model.setCalibratedRmsnormTensors(null);

        CatalogModel saved = catalogRepository.save(model);

        if (ggufFile != null && ggufFile.exists()) {
            try {
                compileModelCovariantGguf(
                    saved.getModelId(),
                    new CompileCovariantRequest("auto-covariant-" + saved.getModelId(), 2.0)
                );
            } catch (Exception e) {
                log.info("Automatic covariant preparation fallback for {}: {}", saved.getModelId(), e.getMessage());
            }
        }

        auditService.recordEvent(
            "MODEL_CATALOG_REGISTRATION",
            "VERIFIED",
            "Registered Model '" + saved.getDisplayName() + "' (" + saved.getTotalLayers() + " layers, " + saved.getRequiredVramMb() + " MB)",
            "Registrato Modello '" + saved.getDisplayName() + "' (" + saved.getTotalLayers() + " layer, " + saved.getRequiredVramMb() + " MB)",
            "Inspected GGUF '" + saved.getBaseGgufPath() + "' and configured " + saved.getOrchestrationMode() + " with " + saved.getSecurityProfile() + ".",
            "Ispezionato GGUF '" + saved.getBaseGgufPath() + "' e configurato " + saved.getOrchestrationMode() + " con " + saved.getSecurityProfile() + ".",
            saved.getModelId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );

        if (saved.getOrchestrationMode() == OrchestrationMode.AUTO_PILOT) {
            reconcileAutoPilotRigs();
        }
        return saved;
    }

    @Transactional
    public CatalogModel updateModelOrchestration(String modelId, UpdateModelOrchestrationRequest req) {
        CatalogModel model = catalogRepository.findById(modelId)
            .orElseThrow(() -> new EntityNotFoundException("Model not found: " + modelId));
        model.setOrchestrationMode(req.orchestrationMode());
        model.setMinActiveRigs(req.minActiveRigs());
        model.setMaxActiveRigs(req.maxActiveRigs());
        model.setSecurityProfile(req.securityProfile());
        if (req.priorityOrder() != null && req.priorityOrder() >= 1) {
            model.setPriorityOrder(req.priorityOrder());
        }
        CatalogModel saved = catalogRepository.save(model);

        auditService.recordEvent(
            "MODEL_ORCHESTRATION_POLICY",
            "INFO",
            "Updated Orchestration Policy for " + saved.getDisplayName() + " -> " + saved.getOrchestrationMode() + " (P" + saved.getPriorityOrder() + ")",
            "Aggiornata Policy di Orchestrazione per " + saved.getDisplayName() + " -> " + saved.getOrchestrationMode() + " (P" + saved.getPriorityOrder() + ")",
            "Set priorityOrder=" + saved.getPriorityOrder() + ", minActiveRigs=" + saved.getMinActiveRigs()
                + ", maxActiveRigs=" + saved.getMaxActiveRigs() + ", securityProfile=" + saved.getSecurityProfile() + ".",
            "Impostati priorityOrder=" + saved.getPriorityOrder() + ", minActiveRigs=" + saved.getMinActiveRigs()
                + ", maxActiveRigs=" + saved.getMaxActiveRigs() + ", securityProfile=" + saved.getSecurityProfile() + ".",
            saved.getModelId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );

        if (saved.getOrchestrationMode() == OrchestrationMode.MANUAL_ONLY) {
            List<ManagedRig> activeForModel = rigRepository.findByModelIdAndLifecycleState(
                saved.getModelId(),
                RigLifecycleState.SERVING
            );
            for (ManagedRig rig : activeForModel) {
                if (rig.getOrchestrationMode() == OrchestrationMode.AUTO_PILOT) {
                    dissolveRig(rig.getId());
                }
            }
        }
        reconcileAutoPilotRigs();
        return saved;
    }

    /**
     * Strict total ordering over CatalogModels for Policy 1 (Priority with Automatic Promotion):
     * 1. Lower `priorityOrder` wins (`1` beats `2` beats `5`).
     * 2. Equal `priorityOrder`: larger `requiredVramMb` wins so multi-GPU models are never starved by small single-GPU models.
     * 3. Deterministic tie-breaker by `modelId`.
     */
    private int compareModelPriority(CatalogModel a, CatalogModel b) {
        int pCmp = Integer.compare(a.getPriorityOrder(), b.getPriorityOrder());
        if (pCmp != 0) {
            return pCmp;
        }
        int vramCmp = Long.compare(b.getRequiredVramMb(), a.getRequiredVramMb());
        if (vramCmp != 0) {
            return vramCmp;
        }
        return a.getModelId().compareTo(b.getModelId());
    }

    private double computeWorkerComputeScore(GpuWorkerNode w) {
        return placementPlanner.computeWorkerComputeScore(w);
    }

    private List<GpuWorkerNode> selectBestFitWorkersForModel(
        CatalogModel model,
        List<GpuWorkerNode> candidatePool
    ) {
        return placementPlanner.selectBestFitWorkersForModel(model, candidatePool);
    }

    public boolean isRigCurrentlyServing(String rigId) {
        if (rigId == null || rigId.isBlank()) {
            return false;
        }
        return rigRepository.findById(rigId)
            .map(r -> r.getLifecycleState() == RigLifecycleState.SERVING)
            .orElse(false);
    }

    /**
     * Multi-Model Policy 1 Scheduler (Priority with Automatic Promotion + Best-Fit + Fair-Share Coverage):
     * - PASS 1 (Primary Coverage 0 -> 1 Rig + Automatic Priority Promotion):
     *   Ensures each active AUTO_PILOT model gets its 1st Rig in priority order. If a higher-priority model
     *   has 0 Rigs and cannot fit in IDLE_STANDBY_POOL alone, checks if combining IDLE_STANDBY_POOL with
     *   idle lower-priority AUTO_PILOT Rigs (or duplicate replicas) satisfies `model.getRequiredVramMb()`.
     *   If so, minimally preempts only the required lower-priority Rigs, promotes the GPUs to the higher-priority
     *   model, and then re-provisions the lower-priority models on any remaining GPUs in the same pass.
     * - PASS 2 (Scale-Out Replicas 1 -> minActiveRigs):
     *   Allocates leftover IDLE_STANDBY_POOL GPUs to additional replicas up to `minActiveRigs`.
     */
    @Scheduled(fixedDelayString = "${cutmesh.control-plane.autopilot-interval-ms:6000}")
    @Transactional
    @Override
    public synchronized int reconcileAutoPilotRigs() {
        if (!clusterLeaderLock.tryAcquireTransactionLock("cutmesh:orchestration:autopilot")) {
            return 0;
        }
        // 0. Clean up any duplicate/ghost SERVING Rigs whose GPUs were reassigned to a newer Rig
        List<ManagedRig> allCurrentlyServing = rigRepository.findByLifecycleStateOrderByCreatedAtDesc(RigLifecycleState.SERVING);
        Set<String> claimedWorkerIds = new HashSet<>();
        Set<String> validServingRigIds = new HashSet<>();
        for (ManagedRig servingRig : allCurrentlyServing) {
            List<GpuWorkerNode> assigned = workerRepository.findByAssignedRigId(servingRig.getId());
            int expectedCount = servingRig.getAllocations() != null ? servingRig.getAllocations().size() : 0;
            boolean overlapping = assigned.stream().anyMatch(w -> !claimedWorkerIds.add(w.getId()));
            if (assigned.isEmpty() || (expectedCount > 0 && assigned.size() < expectedCount) || overlapping) {
                servingRig.setLifecycleState(RigLifecycleState.DISSOLVED);
                servingRig.setClosedAt(Instant.now());
                rigRepository.save(servingRig);
                stopRigRuntime(servingRig.getId());
                for (GpuWorkerNode w : assigned) {
                    if (servingRig.getId().equals(w.getAssignedRigId())) {
                        w.setAssignmentState(WorkerAssignmentState.IDLE_STANDBY_POOL);
                        w.setAssignedRigId(null);
                        w.setAssignedModelId(null);
                        w.setAssignedRole(null);
                        workerRepository.save(w);
                    }
                }
            } else {
                validServingRigIds.add(servingRig.getId());
            }
        }

        // 0b. Self-heal any orphan LOCKED_IN_RIG workers pointing to a DISSOLVED or missing Rig
        List<GpuWorkerNode> lockedWorkers = workerRepository.findByAssignmentStateOrderByMaxAllocatableVramMbDesc(
            WorkerAssignmentState.LOCKED_IN_RIG
        );
        for (GpuWorkerNode w : lockedWorkers) {
            if (w.getAssignedRigId() == null || !validServingRigIds.contains(w.getAssignedRigId())) {
                w.setAssignmentState(WorkerAssignmentState.IDLE_STANDBY_POOL);
                w.setAssignedRigId(null);
                w.setAssignedModelId(null);
                w.setAssignedRole(null);
                workerRepository.save(w);
            }
        }

        List<CatalogModel> allModels = new ArrayList<>(catalogRepository.findAllByOrderByPriorityOrderAsc());
        Map<String, CatalogModel> modelById = new HashMap<>();
        List<CatalogModel> autoPilotModels = new ArrayList<>();
        for (CatalogModel m : allModels) {
            modelById.put(m.getModelId(), m);
            if (m.getOrchestrationMode() == OrchestrationMode.AUTO_PILOT
                && m.getMinActiveRigs() > 0
                && m.getMaxActiveRigs() > 0) {
                autoPilotModels.add(m);
            }
        }
        autoPilotModels.sort(this::compareModelPriority);

        int newlyProvisioned = 0;

        // PASS 1: Give at least 1 active Rig to each AUTO_PILOT model in Priority Order (with Automatic Promotion)
        for (CatalogModel model : autoPilotModels) {
            List<ManagedRig> activeForModel = rigRepository.findByModelIdAndLifecycleState(
                model.getModelId(),
                RigLifecycleState.SERVING
            );
            if (!activeForModel.isEmpty()) {
                for (ManagedRig existingRig : activeForModel) {
                    String ep = existingRig.getInferenceEndpointUrl();
                    boolean ownedByThisPod = ep == null
                        || ep.contains("://" + podAdvertiseHost + ":")
                        || ep.contains("://127.0.0.1:");
                    if (ownedByThisPod && !isRigProcessAlive(existingRig.getId())) {
                        List<GpuWorkerNode> rigWorkers = orderWorkersByHostLocality(
                            workerRepository.findByAssignedRigId(existingRig.getId()),
                            parseDbModelPolicy(model).nonBlockMasterWeightsMib()
                        );
                        if (!rigWorkers.isEmpty()) {
                            int port = 8089;
                            if (ep != null && ep.contains(":")) {
                                try {
                                    port = Integer.parseInt(ep.substring(ep.lastIndexOf(':') + 1).replaceAll("[^0-9]", ""));
                                } catch (Exception ignored) {
                                }
                            }
                            launchRigRuntimeAsync(
                                existingRig,
                                model,
                                rigWorkers,
                                port,
                                Math.max(4096, existingRig.getAutoMaximizedContextSize())
                            );
                        }
                    }
                }
                continue;
            }

            List<GpuWorkerNode> standby = workerRepository.findByAssignmentStateOrderByMaxAllocatableVramMbDesc(
                WorkerAssignmentState.IDLE_STANDBY_POOL
            );
            List<GpuWorkerNode> selectedFromStandby = selectBestFitWorkersForModel(model, standby);
            if (!selectedFromStandby.isEmpty()) {
                String rigName = String.format("autopilot-%s-rig-1", model.getModelId());
                assembleAndLockRig(model, selectedFromStandby, rigName, OrchestrationMode.AUTO_PILOT);
                newlyProvisioned++;
                continue;
            }

            // Automatic Priority Promotion: check if reclaiming duplicate replicas or strictly lower-priority
            // idle AUTO_PILOT Rigs allows this unserved higher-priority model to close its Rig.
            List<ManagedRig> allServing = rigRepository.findByLifecycleStateOrderByCreatedAtDesc(RigLifecycleState.SERVING);
            Map<String, Integer> servingCountByModel = new HashMap<>();
            for (ManagedRig r : allServing) {
                servingCountByModel.merge(r.getModelId(), 1, Integer::sum);
            }

            List<ManagedRig> preemptibleRigs = allServing.stream()
                .filter(r -> r.getOrchestrationMode() == OrchestrationMode.AUTO_PILOT)
                .filter(r -> r.getActiveInFlightRequests() == 0)
                .filter(r -> !r.getModelId().equals(model.getModelId()))
                .filter(r -> {
                    int countForRigModel = servingCountByModel.getOrDefault(r.getModelId(), 1);
                    if (countForRigModel > 1) {
                        return true;
                    }
                    CatalogModel rigModel = modelById.get(r.getModelId());
                    return rigModel == null || compareModelPriority(model, rigModel) < 0;
                })
                .sorted((r1, r2) -> {
                    boolean r1Duplicate = servingCountByModel.getOrDefault(r1.getModelId(), 1) > 1;
                    boolean r2Duplicate = servingCountByModel.getOrDefault(r2.getModelId(), 1) > 1;
                    if (r1Duplicate != r2Duplicate) {
                        return r1Duplicate ? -1 : 1;
                    }
                    CatalogModel m1 = modelById.get(r1.getModelId());
                    CatalogModel m2 = modelById.get(r2.getModelId());
                    if (m1 != null && m2 != null) {
                        // Sacrifice lowest-priority (highest compareModelPriority rank) Rigs first
                        return compareModelPriority(m2, m1);
                    }
                    return Long.compare(r1.getTotalAllocatedVramMb(), r2.getTotalAllocatedVramMb());
                })
                .collect(Collectors.toList());

            if (preemptibleRigs.isEmpty()) {
                continue;
            }

            List<GpuWorkerNode> candidatePool = new ArrayList<>(standby);
            Map<String, ManagedRig> rigByWorkerId = new HashMap<>();
            List<GpuWorkerNode> promotedSelection = List.of();

            for (ManagedRig preRig : preemptibleRigs) {
                List<GpuWorkerNode> rigWorkers = workerRepository.findByAssignedRigId(preRig.getId());
                for (GpuWorkerNode rw : rigWorkers) {
                    if (rw.isRpcHealthy() && rw.getAssignmentState() != WorkerAssignmentState.OFFLINE_STALE) {
                        candidatePool.add(rw);
                        rigByWorkerId.put(rw.getId(), preRig);
                    }
                }
                List<GpuWorkerNode> attempt = selectBestFitWorkersForModel(model, candidatePool);
                if (!attempt.isEmpty()) {
                    promotedSelection = attempt;
                    break;
                }
            }

            if (!promotedSelection.isEmpty()) {
                // Minimal Preemption: dissolve ONLY the Rigs whose workers were actually selected
                Set<String> rigsToDissolveIds = new HashSet<>();
                List<String> dissolvedNames = new ArrayList<>();
                for (GpuWorkerNode chosen : promotedSelection) {
                    ManagedRig ownerRig = rigByWorkerId.get(chosen.getId());
                    if (ownerRig != null && rigsToDissolveIds.add(ownerRig.getId())) {
                        dissolvedNames.add(ownerRig.getRigName());
                        dissolveRig(ownerRig.getId());
                    }
                }

                auditService.recordEvent(
                    "RIG_PRIORITY_PROMOTION",
                    "INFO",
                    "Priority Promotion: Reallocated GPUs from " + String.join(", ", dissolvedNames) + " -> " + model.getDisplayName(),
                    "Promozione Automatica Priorità: Riassegnate GPU da " + String.join(", ", dissolvedNames) + " -> " + model.getDisplayName(),
                    "Higher-priority model '" + model.getDisplayName() + "' (P" + model.getPriorityOrder() + ", "
                        + model.getRequiredVramMb() + " MB) reached sufficient cluster VRAM and preempted lower-priority Auto-Pilot rig(s).",
                    "Il modello a priorità superiore '" + model.getDisplayName() + "' (P" + model.getPriorityOrder() + ", "
                        + model.getRequiredVramMb() + " MB) ha raggiunto la VRAM necessaria nel cluster promuovendo le GPU.",
                    model.getModelId(),
                    true,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null
                );

                String rigName = String.format("autopilot-%s-rig-1", model.getModelId());
                assembleAndLockRig(model, promotedSelection, rigName, OrchestrationMode.AUTO_PILOT);
                newlyProvisioned++;
            }
        }

        // PASS 2: Scale-Out additional replicas (rig-2 .. maxActiveRigs) only after all feasible models have 1 Rig
        for (CatalogModel model : autoPilotModels) {
            List<ManagedRig> activeForModel = rigRepository.findByModelIdAndLifecycleState(
                model.getModelId(),
                RigLifecycleState.SERVING
            );
            int currentCount = activeForModel.size();
            while (currentCount < model.getMaxActiveRigs()) {
                List<GpuWorkerNode> standby = workerRepository.findByAssignmentStateOrderByMaxAllocatableVramMbDesc(
                    WorkerAssignmentState.IDLE_STANDBY_POOL
                );
                List<GpuWorkerNode> selected = selectBestFitWorkersForModel(model, standby);
                if (selected.isEmpty()) {
                    break;
                }
                String rigName = String.format("autopilot-%s-rig-%d", model.getModelId(), currentCount + 1);
                assembleAndLockRig(model, selected, rigName, OrchestrationMode.AUTO_PILOT);
                newlyProvisioned++;
                currentCount++;
            }
        }

        return newlyProvisioned;
    }

    @Transactional
    public ManagedRig createManualRig(ManualCreateRigRequest request) {
        CatalogModel model = catalogRepository.findById(request.modelId())
            .orElseThrow(() -> new EntityNotFoundException("Catalog model not found: " + request.modelId()));

        Set<String> rigsToDissolve = new HashSet<>();
        for (String workerId : request.workerNodeIds()) {
            GpuWorkerNode worker = workerRepository.findById(workerId)
                .orElseThrow(() -> new EntityNotFoundException("GPU Worker not found: " + workerId));
            if (worker.getAssignmentState() == WorkerAssignmentState.OFFLINE_STALE) {
                throw new IllegalStateException(
                    "GPU Worker " + worker.getNodeName() + " (" + worker.getGpuName() + ") is OFFLINE"
                );
            }
            if (worker.getAssignedRigId() != null && !worker.getAssignedRigId().isBlank()) {
                rigsToDissolve.add(worker.getAssignedRigId());
            }
        }

        for (String oldRigId : rigsToDissolve) {
            rigRepository.findById(oldRigId).ifPresent(oldRig -> {
                if (oldRig.getLifecycleState() == RigLifecycleState.SERVING) {
                    dissolveRig(oldRigId);
                }
            });
        }

        List<GpuWorkerNode> selectedWorkers = new ArrayList<>();
        long totalAllocatable = 0L;
        for (String workerId : request.workerNodeIds()) {
            GpuWorkerNode worker = workerRepository.findById(workerId)
                .orElseThrow(() -> new EntityNotFoundException("GPU Worker not found: " + workerId));
            selectedWorkers.add(worker);
            totalAllocatable += worker.getMaxAllocatableVramMb();
        }

        int maxGpusForModel = parseDbModelPolicy(model).maxWanNodes();
        if (selectedWorkers.size() > maxGpusForModel) {
            throw new IllegalArgumentException(
                String.format(
                    "Too many GPUs selected (%d): model '%s' allows at most %d GPUs (1, 2, or %d GPUs). Rig closed/rejected.",
                    selectedWorkers.size(), model.getDisplayName(), maxGpusForModel, maxGpusForModel
                )
            );
        }

        if (totalAllocatable < model.getRequiredVramMb()) {
            throw new IllegalArgumentException(
                String.format(
                    "Insufficient aggregate VRAM: selected GPUs provide %d MB, but model '%s' requires %d MB. Rig cannot be formed.",
                    totalAllocatable, model.getDisplayName(), model.getRequiredVramMb()
                )
            );
        }

        return assembleAndLockRig(model, selectedWorkers, request.rigName(), OrchestrationMode.MANUAL_ONLY);
    }

    @Transactional
    public void handleWorkerDisconnectedFromRig(String rigId, GpuWorkerNode disconnectedWorker) {
        rigRepository.findById(rigId).ifPresent(rig -> {
            if (rig.getLifecycleState() != RigLifecycleState.SERVING) {
                return;
            }
            rig.setLifecycleState(RigLifecycleState.DISSOLVED);
            rig.setClosedAt(Instant.now());
            rigRepository.save(rig);
            stopRigRuntime(rig.getId());

            List<GpuWorkerNode> remainingOnline = new ArrayList<>();
            List<GpuWorkerNode> assignedWorkers = workerRepository.findByAssignedRigId(rig.getId());
            for (GpuWorkerNode w : assignedWorkers) {
                if (!w.getId().equals(disconnectedWorker.getId())
                    && w.getAssignmentState() != WorkerAssignmentState.OFFLINE_STALE) {
                    w.setAssignmentState(WorkerAssignmentState.IDLE_STANDBY_POOL);
                    w.setAssignedRigId(null);
                    w.setAssignedModelId(null);
                    w.setAssignedRole(null);
                    workerRepository.save(w);
                    remainingOnline.add(w);
                }
            }

            auditService.recordEvent(
                "RIG_TOPOLOGY_HEALED",
                "WARNING",
                "Topology Reconfigured: GPU '" + disconnectedWorker.getNodeName() + "' Disconnected from Rig '" + rig.getRigName() + "'",
                "Topologia Riconfigurata: GPU '" + disconnectedWorker.getNodeName() + "' Disconnessa dal Rig '" + rig.getRigName() + "'",
                "Rig '" + rig.getRigName() + "' released " + remainingOnline.size() + " surviving GPU(s) for immediate self-healing.",
                "Il Rig '" + rig.getRigName() + "' ha rilasciato " + remainingOnline.size() + " GPU superstiti per il ripristino immediato.",
                rig.getId(),
                true,
                null,
                null,
                null,
                null,
                null,
                null
            );

            catalogRepository.findById(rig.getModelId()).ifPresent(model -> {
                int maxGpusForModel = parseDbModelPolicy(model).maxWanNodes();
                long survivingVram = remainingOnline.stream().mapToLong(GpuWorkerNode::getMaxAllocatableVramMb).sum();
                if (!remainingOnline.isEmpty()
                    && remainingOnline.size() <= maxGpusForModel
                    && survivingVram >= model.getRequiredVramMb()) {
                    String healedRigName = rig.getRigName() + "-healed";
                    assembleAndLockRig(model, remainingOnline, healedRigName, rig.getOrchestrationMode());
                } else if (model.getOrchestrationMode() == OrchestrationMode.AUTO_PILOT) {
                    reconcileAutoPilotRigs();
                }
            });
        });
    }

    @Transactional
    public ManagedRig dissolveRig(String rigId) {
        ManagedRig rig = rigRepository.findById(rigId)
            .orElseThrow(() -> new EntityNotFoundException("ManagedRig not found: " + rigId));

        rig.setLifecycleState(RigLifecycleState.DISSOLVED);
        rig.setClosedAt(Instant.now());
        stopRigRuntime(rig.getId());

        List<GpuWorkerNode> workers = workerRepository.findByAssignedRigId(rig.getId());
        for (GpuWorkerNode worker : workers) {
            if (worker.getAssignmentState() != WorkerAssignmentState.OFFLINE_STALE) {
                worker.setAssignmentState(WorkerAssignmentState.IDLE_STANDBY_POOL);
            }
            worker.setAssignedRigId(null);
            worker.setAssignedModelId(null);
            worker.setAssignedRole(null);
            workerRepository.save(worker);
        }

        ManagedRig saved = rigRepository.save(rig);

        auditService.recordEvent(
            "RIG_DISSOLVED",
            "INFO",
            "Dissolved Rig '" + saved.getRigName() + "' -> Returned " + workers.size() + " GPUs to Standby Pool",
            "Sciolto Rig '" + saved.getRigName() + "' -> Rilasciate " + workers.size() + " GPU nello Standby Pool",
            "All workers assigned to rig '" + saved.getRigName() + "' returned to IDLE_STANDBY_POOL.",
            "Tutti i worker assegnati al rig '" + saved.getRigName() + "' sono tornati in IDLE_STANDBY_POOL.",
            saved.getId(),
            true,
            null,
            null,
            null,
            null,
            null,
            null
        );

        return saved;
    }

    /**
     * Invokes the real Rust `cutmesh compile-covariant-gguf --input <in> --output <out> --secret-seed <seed> --kappa-lambda <kappa>`
     * binary and parses the real `CovariantGgufCompileReport` JSON from stdout.
     */
    @Transactional
    public CovariantCompileResponse compileModelCovariantGguf(String modelId, CompileCovariantRequest req) {
        CatalogModel model = catalogRepository.findById(modelId)
            .orElseThrow(() -> new EntityNotFoundException("Catalog model not found: " + modelId));

        String secretSeed = (req != null && req.secretSeed() != null && !req.secretSeed().isBlank())
            ? req.secretSeed()
            : "cutmesh-covariant-session-secret-2026";
        double kappaLambda = (req != null && req.kappaLambda() != null && req.kappaLambda() >= 1.0)
            ? req.kappaLambda()
            : 2.0;

        boolean compiledPhysical = false;
        double elapsedMs = 0.0;
        double throughputMbS = 0.0;
        long transformedRows = 0L;
        int calibratedRmsnorm = 0;
        String profileApplied = kappaLambda > 1.0001
            ? String.format(Locale.ROOT, "Covariant-GL(d) (kappa_Lambda = %.2f)", kappaLambda)
            : "Strict-Fidelity O(d) (kappa_Lambda = 1.00)";
        double empiricalRmsnormKappa = 0.0;
        String outputHeaderBlake3 = "";
        String clientKeyManifestPath = model.getCovariantGgufPath() + ".client_key.json";
        String rawReportJson = null;

        File rustExe = auditService.resolveRustBinary();
        File inputGguf = auditService.resolveModelFile(model.getBaseGgufPath());
        File outputGguf = resolveOutputModelFile(model.getCovariantGgufPath(), inputGguf);

        if (rustExe != null && rustExe.exists() && inputGguf != null && inputGguf.exists()) {
            try {
                ProcessBuilder pb = new ProcessBuilder(
                    rustExe.getAbsolutePath(),
                    "compile-covariant-gguf",
                    "--input", inputGguf.getAbsolutePath(),
                    "--output", outputGguf.getAbsolutePath(),
                    "--secret-seed", secretSeed,
                    "--kappa-lambda", String.format(Locale.ROOT, "%.2f", kappaLambda)
                );
                pb.redirectErrorStream(true);
                Process process = pb.start();
                StringBuilder out = new StringBuilder();
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        out.append(line).append('\n');
                    }
                }
                boolean done = process.waitFor(60, TimeUnit.SECONDS);
                if (done && process.exitValue() == 0 && outputGguf.exists()) {
                    compiledPhysical = true;
                    String s = out.toString();
                    int jStart = s.indexOf('{');
                    int jEnd = s.lastIndexOf('}');
                    if (jStart >= 0 && jEnd > jStart) {
                        rawReportJson = s.substring(jStart, jEnd + 1);
                        JsonNode report = objectMapper.readTree(rawReportJson);
                        elapsedMs = report.path("compile_wall_clock_ms").asDouble(0.0);
                        throughputMbS = report.path("effective_throughput_mb_s").asDouble(0.0);
                        transformedRows = report.path("token_embd_rows_transformed").asLong(0L);
                        calibratedRmsnorm = report.path("rmsnorm_tensors_calibrated").asInt(0);
                        profileApplied = report.path("profile_applied").asString(profileApplied);
                        empiricalRmsnormKappa = report.path("empirical_rmsnorm_kappa").asDouble(0.0);
                        outputHeaderBlake3 = report.path("output_header_blake3").asString("");
                        clientKeyManifestPath = report.path("client_key_manifest_path").asString(clientKeyManifestPath);
                    }
                }
            } catch (Exception e) {
                log.warn("Physical GGUF compiler failure for {}: {}", modelId, e.getMessage());
            }
        }

        if (!compiledPhysical) {
            throw new org.springframework.web.server.ResponseStatusException(
                org.springframework.http.HttpStatus.PRECONDITION_REQUIRED,
                "Physical GGUF compiler binary or base model file not available on host for model " + modelId
            );
        }

        model.setCovariantCompiled(true);
        model.setLastCompileMs(elapsedMs);
        model.setLastCompileThroughputMbPerSec(throughputMbS);
        model.setTransformedQ50Rows(transformedRows);
        model.setCalibratedRmsnormTensors(calibratedRmsnorm);
        catalogRepository.save(model);

        SecurityAuditEvent auditEvent = auditService.recordEvent(
            "COVARIANT_GGUF_COMPILATION",
            "VERIFIED",
            String.format(Locale.ROOT, "Compiled Physical Covariant GGUF for %s (%.1f ms, %.1f MB/s)", model.getDisplayName(), elapsedMs, throughputMbS),
            String.format(Locale.ROOT, "Compilato GGUF Covariante Fisico per %s (%.1f ms, %.1f MB/s)", model.getDisplayName(), elapsedMs, throughputMbS),
            String.format(Locale.ROOT, "Profile=%s, Transformed %,d Q5_0 rows, calibrated %d F32 RMSNorm tensors (kappa=%.6f), BLAKE3=%s.",
                profileApplied, transformedRows, calibratedRmsnorm, empiricalRmsnormKappa, outputHeaderBlake3.substring(0, Math.min(16, outputHeaderBlake3.length()))),
            String.format(Locale.ROOT, "Profilo=%s, Trasformate %,d righe Q5_0, calibrati %d tensori F32 RMSNorm (kappa=%.6f), BLAKE3=%s.",
                profileApplied, transformedRows, calibratedRmsnorm, empiricalRmsnormKappa, outputHeaderBlake3.substring(0, Math.min(16, outputHeaderBlake3.length()))),
            model.getModelId(),
            true,
            null,
            null,
            null,
            null,
            true,
            rawReportJson
        );

        return new CovariantCompileResponse(
            model.getModelId(),
            model.getBaseGgufPath(),
            model.getCovariantGgufPath(),
            clientKeyManifestPath,
            profileApplied,
            kappaLambda,
            empiricalRmsnormKappa,
            outputHeaderBlake3,
            compiledPhysical,
            elapsedMs,
            throughputMbS,
            transformedRows,
            calibratedRmsnorm,
            auditEvent.getId()
        );
    }

    private ManagedRig assembleAndLockRig(
        CatalogModel model,
        List<GpuWorkerNode> rawWorkers,
        String rigName,
        OrchestrationMode mode
    ) {
        RigPlacementPlanner.DbModelShardingPolicy policy = parseDbModelPolicy(model);
        double stage0OverheadMb = policy.nonBlockMasterWeightsMib();
        List<GpuWorkerNode> workers = orderWorkersByHostLocality(rawWorkers, stage0OverheadMb);
        long totalAllocatable = workers.stream().mapToLong(GpuWorkerNode::getMaxAllocatableVramMb).sum();
        long requiredVram = model.getRequiredVramMb();
        long targetAllocatedTotal = Math.min(totalAllocatable, Math.max(requiredVram, (long) (totalAllocatable * 0.85)));
        int autoMaximizedContext = placementPlanner.computeAutoMaximizedContext(totalAllocatable, requiredVram, policy);

        Set<String> distinctNodes = new HashSet<>();
        for (GpuWorkerNode w : workers) {
            distinctNodes.add(w.getIrohNodeId());
        }
        boolean antiCollusionVerified = workers.size() == 1 || distinctNodes.size() == workers.size();

        List<ManagedRig> currentlyServing = rigRepository.findByLifecycleStateOrderByCreatedAtDesc(RigLifecycleState.SERVING);
        Set<String> usedEndpoints = new HashSet<>();
        for (ManagedRig existing : currentlyServing) {
            if (existing.getInferenceEndpointUrl() != null) {
                usedEndpoints.add(existing.getInferenceEndpointUrl());
            }
        }
        int dynamicPort = 8089;
        while (usedEndpoints.contains("http://" + podAdvertiseHost + ":" + dynamicPort)
            || usedEndpoints.contains("http://127.0.0.1:" + dynamicPort)) {
            dynamicPort++;
        }

        ManagedRig rig = new ManagedRig();
        rig.setRigName(rigName);
        rig.setModelId(model.getModelId());
        rig.setModelDisplayName(model.getDisplayName());
        rig.setLifecycleState(RigLifecycleState.SERVING);
        rig.setOrchestrationMode(mode);
        rig.setSecurityProfile(model.getSecurityProfile());
        rig.setTotalAllocatedVramMb(targetAllocatedTotal);
        rig.setRequiredModelVramMb(requiredVram);
        rig.setAutoMaximizedContextSize(autoMaximizedContext);
        rig.setAntiCollusionVerified(antiCollusionVerified);
        rig.setInferenceEndpointUrl("http://" + podAdvertiseHost + ":" + dynamicPort);

        int n = workers.size();
        int totalLayers = model.getTotalLayers();
        long totalModelBytes = requiredVram * 1024L * 1024L;

        // Compute-Weighted Water-Filling driven 100% by the model's DB policy (`policyJson`)
        double nonBlockReserveMb = (n > 1) ? policy.nonBlockMasterWeightsMib() : 0.0;
        int pipelineLayers = policy.trunkLayers();
        int groupAlign = policy.layerGroupAlignment();
        List<Double> layerWeightsMib = policy.layerWeightsMib();
        double totalTrunkWeightsMib = 0.0;
        for (double mib : layerWeightsMib) {
            totalTrunkWeightsMib += mib;
        }
        if (totalTrunkWeightsMib <= 0.0) {
            totalTrunkWeightsMib = Math.max(256.0, (double) requiredVram - nonBlockReserveMb);
        }

        double[] capsMb = new double[n];
        double[] scores = new double[n];
        double[] allocMb = new double[n];
        boolean[] frozen = new boolean[n];
        double totalCapsMb = 0.0;
        for (int i = 0; i < n; i++) {
            double rawCap = Math.max(256.0, (double) workers.get(i).getMaxAllocatableVramMb());
            double gpuReserveDeltaMb = (i == 0) ? 1024.0 : 0.0;
            double extraStage0 = (i == 0 && n > 1) ? nonBlockReserveMb : 0.0;
            capsMb[i] = Math.max(256.0, rawCap - gpuReserveDeltaMb - extraStage0);
            totalCapsMb += capsMb[i];
        }
        boolean highVramPressure = totalTrunkWeightsMib / Math.max(1.0, totalCapsMb) >= 0.50;
        for (int i = 0; i < n; i++) {
            scores[i] = highVramPressure
                ? capsMb[i]
                : Math.max(1.0, computeWorkerComputeScore(workers.get(i)));
        }
        double remainingMb = Math.max(256.0, totalTrunkWeightsMib);
        for (int iter = 0; iter < n; iter++) {
            double unfrozenScoreSum = 0.0;
            for (int i = 0; i < n; i++) {
                if (!frozen[i]) {
                    unfrozenScoreSum += scores[i];
                }
            }
            if (unfrozenScoreSum <= 0.0 || remainingMb <= 0.0) {
                break;
            }
            boolean cappedAny = false;
            for (int i = 0; i < n; i++) {
                if (!frozen[i]) {
                    double desired = remainingMb * (scores[i] / unfrozenScoreSum);
                    if (desired > capsMb[i]) {
                        allocMb[i] = capsMb[i];
                        frozen[i] = true;
                        remainingMb = Math.max(0.0, remainingMb - capsMb[i]);
                        cappedAny = true;
                    }
                }
            }
            if (!cappedAny) {
                for (int i = 0; i < n; i++) {
                    if (!frozen[i]) {
                        allocMb[i] = remainingMb * (scores[i] / unfrozenScoreSum);
                        frozen[i] = true;
                    }
                }
                break;
            }
        }
        double totalWaterFilledMb = 0.0;
        for (double v : allocMb) {
            totalWaterFilledMb += v;
        }
        totalWaterFilledMb = Math.max(1.0, totalWaterFilledMb);

        List<String> splitFractionsFormatted = new ArrayList<>(n);
        int currentLayerCursor = 0;
        long currentByteCursor = 0L;
        int minGpuMaxContext = policy.maxContextLength();
        double llamaActGpuLayers = (double) Math.max(1, pipelineLayers) + 1.5;
        long prevCumPct = 0L;

        for (int i = 0; i < n; i++) {
            GpuWorkerNode w = workers.get(i);
            double fraction = Math.max(0.05, allocMb[i] / totalWaterFilledMb);
            double stageTotalMb = allocMb[i] + (i == 0 ? nonBlockReserveMb : 0.0);
            long allocatedMb = Math.max(256L, Math.round(stageTotalMb));

            int layerStart = currentLayerCursor;
            int layerEnd;
            if (i == n - 1) {
                layerEnd = totalLayers;
            } else {
                int remainingPeers = n - i - 1;
                int minReservedAfter = remainingPeers * groupAlign;
                int maxEndExcl = Math.max(layerStart + groupAlign, pipelineLayers - minReservedAfter);
                double targetSliceMib = totalTrunkWeightsMib * fraction;

                double accMib = 0.0;
                int rawEndExcl = layerStart + groupAlign;
                for (int l = layerStart; l < maxEndExcl; l++) {
                    double lMib = (l < layerWeightsMib.size())
                        ? layerWeightsMib.get(l)
                        : (totalTrunkWeightsMib / Math.max(1, pipelineLayers));
                    if (accMib + (lMib * 0.5) > targetSliceMib && l > layerStart) {
                        rawEndExcl = l;
                        break;
                    }
                    accMib += lMib;
                    rawEndExcl = l + 1;
                }
                layerEnd = snapPolicyCutLayer(rawEndExcl, layerStart + groupAlign, maxEndExcl, policy);
            }
            currentLayerCursor = layerEnd;

            double blockSliceMib = 0.0;
            int sliceEndTrunk = Math.min(layerEnd, pipelineLayers);
            for (int l = layerStart; l < sliceEndTrunk; l++) {
                blockSliceMib += (l < layerWeightsMib.size())
                    ? layerWeightsMib.get(l)
                    : (totalTrunkWeightsMib / Math.max(1, pipelineLayers));
            }
            long cumPct = (i == n - 1)
                ? 100L
                : Math.max(prevCumPct + 1L, Math.min(99L, Math.round((((double) layerEnd - 0.5) / llamaActGpuLayers) * 100.0)));
            long splitPct = Math.max(1L, cumPct - prevCumPct);
            prevCumPct = cumPct;
            splitFractionsFormatted.add(String.valueOf(splitPct));

            double actualSliceWeightsMib = ((i == 0) ? nonBlockReserveMb : 0.0) + blockSliceMib;
            double layerFrac = Math.max(0.05, (double) Math.max(1, layerEnd - layerStart) / Math.max(1, totalLayers));
            long kvPer1k = Math.max(8L, policy.kvVramPer1kTokensMb());
            double gpuKvPer1kMb = Math.max(1.0, (double) kvPer1k * layerFrac);
            double gpuReserveDeltaMb = (i == 0) ? 1024.0 : 0.0;
            double gpuSurplusMb = Math.max(128.0, (double) w.getMaxAllocatableVramMb() - actualSliceWeightsMib - gpuReserveDeltaMb);
            int gpuMaxCtx = (int) Math.floor((gpuSurplusMb / gpuKvPer1kMb) * 1024.0);
            minGpuMaxContext = Math.min(minGpuMaxContext, gpuMaxCtx);

            long bytesForNode = (i == n - 1)
                ? (totalModelBytes - currentByteCursor)
                : Math.round(fraction * totalModelBytes);
            long byteStart = currentByteCursor;
            long byteEnd = Math.min(totalModelBytes, byteStart + bytesForNode);
            currentByteCursor = byteEnd;

            String role;
            if (n == 1) {
                role = "EMBEDDING_INGRESS_AND_LOGITS_EGRESS";
            } else if (i == 0) {
                role = "EMBEDDING_INGRESS_AND_OUTPUT_HEAD (STAGE_0)";
            } else if (i == n - 1) {
                role = "TAIL_BLOCKS_EGRESS (STAGE_" + i + ")";
            } else {
                role = "COVARIANT_MIDDLE_TRUNK_" + i;
            }

            RigMemberAllocation alloc = new RigMemberAllocation();
            alloc.setWorkerNodeId(w.getId());
            alloc.setIrohNodeId(w.getIrohNodeId());
            alloc.setGpuName(w.getGpuName());
            alloc.setPipelineStageIndex(i);
            alloc.setAssignedRole(role);
            alloc.setAllocatedVramMb(allocatedMb);
            alloc.setTensorSplitFraction(fraction);
            alloc.setLayerStart(layerStart);
            alloc.setLayerEnd(layerEnd);
            alloc.setByteRangeStart(byteStart);
            alloc.setByteRangeEnd(byteEnd);
            rig.addAllocation(alloc);
        }

        autoMaximizedContext = Math.min(policy.maxContextLength(), Math.max(4096, (minGpuMaxContext / 1024) * 1024));
        rig.setAutoMaximizedContextSize(autoMaximizedContext);
        rig.setTensorSplitCsv(String.join(",", splitFractionsFormatted));
        ManagedRig savedRig = rigRepository.save(rig);

        for (int i = 0; i < n; i++) {
            GpuWorkerNode w = workers.get(i);
            RigMemberAllocation alloc = savedRig.getAllocations().get(i);
            w.setAssignmentState(WorkerAssignmentState.LOCKED_IN_RIG);
            w.setAssignedRigId(savedRig.getId());
            w.setAssignedModelId(model.getModelId());
            w.setAssignedRole(alloc.getAssignedRole());
            workerRepository.save(w);
        }

        auditService.recordEvent(
            "RIG_CLOSED_AND_LOCKED",
            "VERIFIED",
            String.format(Locale.ROOT, "Closed & Locked Rig '%s' for %s (%d GPUs, -c %d)",
                savedRig.getRigName(), model.getDisplayName(), n, autoMaximizedContext),
            String.format(Locale.ROOT, "Chiuso & Bloccato Rig '%s' per %s (%d GPU, -c %d)",
                savedRig.getRigName(), model.getDisplayName(), n, autoMaximizedContext),
            String.format(Locale.ROOT, "Mode=%s, TensorSplit=[%s], AllocatedVRAM=%d MB, AntiCollusionVerified=%s, Security=%s.",
                mode, savedRig.getTensorSplitCsv(), savedRig.getTotalAllocatedVramMb(), antiCollusionVerified, model.getSecurityProfile()),
            String.format(Locale.ROOT, "Modalità=%s, TensorSplit=[%s], VRAM Allocata=%d MB, AntiCollusioneVerificata=%s, Sicurezza=%s.",
                mode, savedRig.getTensorSplitCsv(), savedRig.getTotalAllocatedVramMb(), antiCollusionVerified, model.getSecurityProfile()),
            savedRig.getId(),
            true,
            null,
            null,
            null,
            null,
            antiCollusionVerified,
            null
        );

        launchRigRuntimeAsync(savedRig, model, workers, dynamicPort, autoMaximizedContext);
        return savedRig;
    }

    @Transactional(readOnly = true)
    public WorkerShardAssignment resolveWorkerShardAssignment(GpuWorkerNode worker) {
        if (worker == null || worker.getAssignedRigId() == null || worker.getAssignedModelId() == null) {
            return null;
        }
        CatalogModel model = catalogRepository.findById(worker.getAssignedModelId()).orElse(null);
        ManagedRig rig = rigRepository.findById(worker.getAssignedRigId()).orElse(null);
        if (model == null || rig == null) {
            return null;
        }
        String url = resolveCanonicalHuggingFaceUrl(model);
        if (url == null || url.isBlank()) {
            return null;
        }
        for (RigMemberAllocation alloc : rig.getAllocations()) {
            if (worker.getId().equals(alloc.getWorkerNodeId())
                || (worker.getIrohNodeId() != null && worker.getIrohNodeId().equals(alloc.getIrohNodeId()))) {
                int maxLayer = Math.max(0, model.getTotalLayers() - 1);
                int s = Math.max(0, alloc.getLayerStart() - 2);
                int e = Math.min(maxLayer, Math.max(s, alloc.getLayerEnd() - 1) + 2);
                return new WorkerShardAssignment(url, s, e);
            }
        }
        return null;
    }

    private String resolveCanonicalHuggingFaceUrl(CatalogModel model) {
        String hf = model.getHuggingfaceRepo();
        if (hf != null && (hf.startsWith("http://") || hf.startsWith("https://"))) {
            return hf.trim();
        }
        String base = model.getBaseGgufPath();
        if (base != null && (base.startsWith("http://") || base.startsWith("https://"))) {
            return base.trim();
        }
        return null;
    }

    private void launchRigRuntimeAsync(
        ManagedRig savedRig,
        CatalogModel model,
        List<GpuWorkerNode> workers,
        int dynamicPort,
        int autoMaximizedContext
    ) {
        File rustExe = auditService.resolveRustBinary();
        if (rustExe == null || !rustExe.exists()) {
            return;
        }
        String remoteUrl = resolveCanonicalHuggingFaceUrl(model);
        File localGguf = auditService.resolveModelFile(model.getBaseGgufPath());
        String modelArg = (localGguf != null && localGguf.exists())
            ? localGguf.getAbsolutePath()
            : (remoteUrl != null ? remoteUrl : model.getBaseGgufPath());
        if (modelArg == null || modelArg.isBlank()) {
            return;
        }

        List<String> tickets = new ArrayList<>();
        for (GpuWorkerNode w : workers) {
            if (w.getIrohTicket() != null && w.getIrohTicket().startsWith("iroh://")) {
                tickets.add(w.getIrohTicket().trim());
            }
        }
        if (tickets.isEmpty()) {
            return;
        }

        String rigId = savedRig.getId();
        String ticketsCsv = String.join(",", tickets);
        final long launchEpoch;
        synchronized (activeRigProcesses) {
            stopRigRuntime(rigId);
            launchEpoch = rigRuntimeEpoch.merge(rigId, 1L, Long::sum);
            rigLaunchPending.put(rigId, Boolean.TRUE);
        }
        rigFsmStateCodeById.put(rigId, RigRuntimePhaseState.STEP_1_LAYERS_ALLOCATED.name());
        transitionRigStateMachine(
            rigId,
            RigRuntimePhaseState.STEP_2_PREWARMING_SHARDS,
            "Avvio orchestratore P2P auto-rig e verifica shard GGUF sui worker..."
        );

        Thread runner = new Thread(() -> {
            Process proc = null;
            try {
                List<String> cmd = new ArrayList<>();
                cmd.add(rustExe.getAbsolutePath());
                cmd.add("auto-rig");
                cmd.add("--model");
                cmd.add(modelArg);
                if (remoteUrl != null && !remoteUrl.isBlank()) {
                    cmd.add("--model-url");
                    cmd.add(remoteUrl);
                }
                cmd.add("--tickets");
                cmd.add(ticketsCsv);
                cmd.add("--min-nodes");
                cmd.add(String.valueOf(Math.max(1, tickets.size())));
                cmd.add("--port");
                cmd.add(String.valueOf(dynamicPort));
                cmd.add("--ctx-size");
                cmd.add(String.valueOf(Math.max(4096, autoMaximizedContext)));
                cmd.add("--max-rtt-ms");
                cmd.add("500.0");

                ProcessBuilder pb = new ProcessBuilder(cmd);
                String strictDirect = System.getenv("CUTMESH_STRICT_DIRECT");
                pb.environment().put(
                    "CUTMESH_STRICT_DIRECT",
                    (strictDirect != null && !strictDirect.isBlank()) ? strictDirect.trim() : "1"
                );
                if (model.getPolicyJson() != null && !model.getPolicyJson().isBlank()) {
                    pb.environment().put("CUTMESH_MODEL_POLICY_JSON", model.getPolicyJson());
                }
                if (rustExe.getParentFile() != null
                    && "release".equalsIgnoreCase(rustExe.getParentFile().getName())
                    && rustExe.getParentFile().getParentFile() != null
                    && rustExe.getParentFile().getParentFile().getParentFile() != null) {
                    pb.directory(rustExe.getParentFile().getParentFile().getParentFile());
                }
                pb.redirectErrorStream(true);
                synchronized (activeRigProcesses) {
                    if (!Long.valueOf(launchEpoch).equals(rigRuntimeEpoch.get(rigId))) {
                        return;
                    }
                    proc = pb.start();
                    activeRigProcesses.put(rigId, proc);
                    rigLaunchPending.remove(rigId);
                }
                log.info("Auto-launched cutmesh auto-rig for Rig {} on port {}", savedRig.getRigName(), dynamicPort);

                // Drain stdout/stderr continuously so the OS pipe buffer never blocks auto-rig,
                // and drive the Rig Runtime Finite State Machine (FSM) in real time.
                try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(proc.getInputStream(), StandardCharsets.UTF_8)
                )) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (!Long.valueOf(launchEpoch).equals(rigRuntimeEpoch.get(rigId))) {
                            terminateProcessTree(proc);
                            break;
                        }
                        String trimmed = line.trim();
                        if (trimmed.isEmpty() || trimmed.startsWith("{") || trimmed.startsWith("}") || trimmed.startsWith("\"")) {
                            continue;
                        }
                        String clean = trimmed.replaceAll("\\u001B\\[[;\\d]*m", "");
                        rigLastLogById.put(rigId, clean);
                        if (clean.contains("READY (200 OK)") || clean.contains("[RIG_PHASE:READY]")) {
                            transitionRigStateMachine(rigId, RigRuntimePhaseState.STEP_4_OPERATIONAL_READY, clean);
                        } else if (clean.contains("[RIG_PHASE:STARTING_LLAMA]")
                            || clean.contains("AutoRig Effective Context")
                            || clean.contains("spawn")
                            || clean.contains("listening on")) {
                            transitionRigStateMachine(rigId, RigRuntimePhaseState.STEP_3_STARTING_LLAMA_VRAM, clean);
                        } else if (clean.contains("Download")
                            || clean.contains("Pre-warm")
                            || clean.contains("[RIG_PHASE:PREWARMING]")) {
                            transitionRigStateMachine(rigId, RigRuntimePhaseState.STEP_2_PREWARMING_SHARDS, clean);
                        }
                    }
                }
            } catch (Exception e) {
                if (Long.valueOf(launchEpoch).equals(rigRuntimeEpoch.get(rigId))) {
                    transitionRigStateMachine(rigId, RigRuntimePhaseState.ERROR_DEGRADED, "Nota avvio runtime: " + e.getMessage());
                }
                log.warn("Could not auto-launch Rig runtime for {}: {}", savedRig.getRigName(), e.getMessage());
            } finally {
                synchronized (activeRigProcesses) {
                    if (Long.valueOf(launchEpoch).equals(rigRuntimeEpoch.get(rigId))) {
                        rigLaunchPending.remove(rigId);
                    }
                    if (proc != null) {
                        activeRigProcesses.remove(rigId, proc);
                        terminateProcessTree(proc);
                    }
                }
            }
        }, "rig-runtime-" + savedRig.getRigName());
        runner.setDaemon(true);
        runner.start();
    }

    /**
     * Executes a deterministic transition in the Rig Runtime Finite State Machine (`RigRuntimePhaseState`).
     */
    public synchronized RigRuntimePhaseState transitionRigStateMachine(
        String rigId,
        RigRuntimePhaseState nextState,
        String detailLog
    ) {
        if (rigId == null || nextState == null) {
            return RigRuntimePhaseState.STEP_1_LAYERS_ALLOCATED;
        }
        if (detailLog != null && !detailLog.isBlank()) {
            rigLastLogById.put(rigId, detailLog);
        }
        RigRuntimePhaseState current = getRigFsmState(rigId);
        if (current.canTransitionTo(nextState)) {
            rigFsmStateCodeById.put(rigId, nextState.name());
            log.info("Rig FSM [{}]: {} -> {} ({})", rigId, current.getCode(), nextState.getCode(), detailLog);
            return nextState;
        }
        return current;
    }

    public boolean isRigProcessAlive(String rigId) {
        if (rigId == null) {
            return false;
        }
        if (Boolean.TRUE.equals(rigLaunchPending.get(rigId))) {
            return true;
        }
        Process p = activeRigProcesses.get(rigId);
        return p != null && p.isAlive();
    }

    public RigRuntimePhaseState getRigFsmState(String rigId) {
        if (rigId == null) {
            return RigRuntimePhaseState.STEP_1_LAYERS_ALLOCATED;
        }
        String code = rigFsmStateCodeById.get(rigId);
        if (code == null || code.isBlank()) {
            return RigRuntimePhaseState.STEP_1_LAYERS_ALLOCATED;
        }
        try {
            return RigRuntimePhaseState.valueOf(code);
        } catch (IllegalArgumentException ignored) {
            return RigRuntimePhaseState.STEP_1_LAYERS_ALLOCATED;
        }
    }

    public String getRigRuntimePhase(String rigId) {
        return getRigFsmState(rigId).getCode();
    }

    public String getRigLastLog(String rigId) {
        if (rigId == null) {
            return "";
        }
        return rigLastLogById.getOrDefault(rigId, "");
    }

    private void stopRigRuntime(String rigId) {
        if (rigId == null) {
            return;
        }
        synchronized (activeRigProcesses) {
            rigRuntimeEpoch.merge(rigId, 1L, Long::sum);
            rigLaunchPending.remove(rigId);
            rigFsmStateCodeById.remove(rigId);
            rigLastLogById.remove(rigId);
            Process proc = activeRigProcesses.remove(rigId);
            terminateProcessTree(proc);
        }
    }

    private static void terminateProcessTree(Process proc) {
        if (proc == null) {
            return;
        }
        try {
            proc.descendants().forEach(ProcessHandle::destroyForcibly);
        } catch (Exception ignored) {
        }
        if (proc.isAlive()) {
            proc.destroyForcibly();
        }
    }

    /**
     * Orders Rig workers so:
     * 1. Stage 0 (Elected Master) is the GPU with the highest Silicon Compute Score (TFLOPS + Bandwidth + VRAM).
     * 2. Remaining GPUs on the same physical host / public IP as the Master stay contiguous (Stage 0 -> Stage 1)
     *    before crossing the WAN, sorted by Silicon Compute Score descending.
     */
    private List<GpuWorkerNode> orderWorkersByHostLocality(
        List<GpuWorkerNode> rawWorkers,
        double stage0OverheadMb
    ) {
        return placementPlanner.orderWorkersByHostLocality(rawWorkers, stage0OverheadMb);
    }

    private File resolveOutputModelFile(String relativeOutPath, File inputFile) {
        if (inputFile != null && inputFile.getParentFile() != null) {
            return new File(inputFile.getParentFile(), new File(relativeOutPath).getName());
        }
        return new File(relativeOutPath);
    }
}
