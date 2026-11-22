package io.cutmesh.controlplane.orchestration;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import io.cutmesh.controlplane.fleet.GpuWorkerNode;
import io.cutmesh.controlplane.fleet.WorkerAssignmentState;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * Pure, stateless placement, Stage-0 Master election, cut-point snapping, and context-window
 * planning component extracted from {@link HybridRigOrchestratorService} (Single Responsibility).
 */
@Component
public class RigPlacementPlanner {

    private static final Logger log = LoggerFactory.getLogger(RigPlacementPlanner.class);

    private final ObjectMapper objectMapper;

    public RigPlacementPlanner(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public record DbModelShardingPolicy(
        int trunkLayers,
        int layerGroupAlignment,
        List<Integer> preferredCutLayers,
        List<Integer> allowedCutLayers,
        List<Double> layerWeightsMib,
        double nonBlockMasterWeightsMib,
        long kvVramPer1kTokensMb,
        int maxWanNodes,
        int maxLanNodes,
        int maxContextLength
    ) {
    }

    public DbModelShardingPolicy parseDbModelPolicy(CatalogModel model) {
        int trunkLayers = Math.max(1, model.getTotalLayers());
        int align = 1;
        List<Integer> preferredCuts = new ArrayList<>();
        List<Integer> allowedCuts = new ArrayList<>();
        List<Double> layerMibs = new ArrayList<>();
        double nonBlockMib = 1024.0;
        long kvPer1k = Math.max(8L, model.getKvVramPer1kTokensMb());
        int maxWan = 3;
        int maxLan = 4;
        int maxCtx = model.getMaxContextLength() > 0 ? model.getMaxContextLength() : 262144;

        String raw = model.getPolicyJson();
        if (raw != null && !raw.isBlank()) {
            try {
                JsonNode root = objectMapper.readTree(raw);
                JsonNode sh = root.path("sharding");
                if (!sh.isMissingNode()) {
                    trunkLayers = sh.path("trunk_layers").asInt(trunkLayers);
                    align = Math.max(1, sh.path("layer_group_alignment").asInt(align));
                    nonBlockMib = sh.path("non_block_master_weights_mib").asDouble(nonBlockMib);
                    kvPer1k = Math.max(8L, sh.path("kv_vram_per_1k_tokens_mb").asLong(kvPer1k));
                    maxWan = Math.max(1, sh.path("max_wan_nodes").asInt(maxWan));
                    maxLan = Math.max(1, sh.path("max_lan_nodes").asInt(maxLan));
                    for (JsonNode n : sh.path("preferred_cut_layers")) {
                        preferredCuts.add(n.asInt());
                    }
                    for (JsonNode n : sh.path("allowed_cut_layers")) {
                        allowedCuts.add(n.asInt());
                    }
                    for (JsonNode n : sh.path("layer_weights_mib")) {
                        layerMibs.add(n.asDouble());
                    }
                }
                JsonNode spec = root.path("speculative");
                if (!spec.isMissingNode() && spec.path("enabled").asBoolean(false)) {
                    nonBlockMib += spec.path("draft_weights_mib").asDouble(0.0);
                }
                JsonNode eng = root.path("engine");
                if (!eng.isMissingNode()) {
                    maxCtx = eng.path("max_context_length").asInt(maxCtx);
                }
            } catch (Exception e) {
                log.warn("Could not parse policyJson for {}: {}", model.getModelId(), e.getMessage());
            }
        }
        return new DbModelShardingPolicy(
            trunkLayers,
            align,
            preferredCuts,
            allowedCuts,
            layerMibs,
            nonBlockMib,
            kvPer1k,
            maxWan,
            maxLan,
            maxCtx
        );
    }

    public int snapPolicyCutLayer(int rawEndExcl, int minEndExcl, int maxEndExcl, DbModelShardingPolicy policy) {
        int lo = Math.min(minEndExcl, maxEndExcl);
        int hi = Math.max(lo, maxEndExcl);

        int bestPref = -1;
        int bestPrefDist = Integer.MAX_VALUE;
        for (int c : policy.preferredCutLayers()) {
            if (c >= lo && c <= hi) {
                int dist = Math.abs(c - rawEndExcl);
                if (dist < bestPrefDist) {
                    bestPrefDist = dist;
                    bestPref = c;
                }
            }
        }
        if (bestPref >= 0 && bestPrefDist <= 3) {
            return bestPref;
        }

        int bestAllowed = -1;
        int bestAllowedDist = Integer.MAX_VALUE;
        for (int c : policy.allowedCutLayers()) {
            if (c >= lo && c <= hi) {
                int dist = Math.abs(c - rawEndExcl);
                if (dist < bestAllowedDist) {
                    bestAllowedDist = dist;
                    bestAllowed = c;
                }
            }
        }
        if (bestAllowed >= 0) {
            return bestAllowed;
        }

        int align = Math.max(1, policy.layerGroupAlignment());
        if (align > 1) {
            int snapped = ((rawEndExcl + align / 2) / align) * align;
            return Math.max(lo, Math.min(hi, snapped));
        }
        return Math.max(lo, Math.min(hi, rawEndExcl));
    }

    public double computeWorkerComputeScore(GpuWorkerNode w) {
        if (w == null) {
            return 0.0;
        }
        double tflops = Math.max(0.5, w.getEffectiveTflops());
        double bandwidth = Math.max(50.0, w.getEffectiveBandwidthGbS());
        double allocMb = Math.max(512.0, (double) w.getMaxAllocatableVramMb());
        double vramFactor = 1.0 + (allocMb / 8192.0);
        return bandwidth * Math.sqrt(tflops) * vramFactor;
    }

    public double computeStage0MasterScore(GpuWorkerNode w, double stage0OverheadMb) {
        double base = computeWorkerComputeScore(w);
        if (w == null || stage0OverheadMb <= 0.0) {
            return base;
        }
        double masterExtraReserveMb = w.getFreeVramMb() > 0
            ? Math.max(0.0, 2560.0 - Math.max(0.0, (double) (w.getFreeVramMb() - w.getMaxAllocatableVramMb())))
            : 1024.0;
        double allocMb = Math.max(512.0, (double) w.getMaxAllocatableVramMb() - masterExtraReserveMb);
        double headroom = allocMb - stage0OverheadMb;
        if (headroom < 4096.0) {
            double viability = 0.05 * Math.max(0.05, Math.max(0.0, headroom) / 4096.0);
            return base * viability;
        }
        return base;
    }

    public List<GpuWorkerNode> selectBestFitWorkersForModel(
        CatalogModel model,
        List<GpuWorkerNode> candidatePool
    ) {
        long requiredVram = model.getRequiredVramMb();
        DbModelShardingPolicy policy = parseDbModelPolicy(model);
        int maxGpusForModel = policy.maxWanNodes();
        List<GpuWorkerNode> healthy = candidatePool.stream()
            .filter(w -> w != null && w.isRpcHealthy() && w.getAssignmentState() != WorkerAssignmentState.OFFLINE_STALE)
            .collect(Collectors.toList());
        long totalHealthyVram = healthy.stream().mapToLong(GpuWorkerNode::getMaxAllocatableVramMb).sum();
        if (totalHealthyVram < requiredVram) {
            return List.of();
        }

        // Case A: Single-GPU Best-Fit Tier + Highest TFLOPS Score
        List<GpuWorkerNode> singleGpuFits = healthy.stream()
            .filter(w -> w.getMaxAllocatableVramMb() >= requiredVram)
            .sorted((a, b) -> {
                long tierA = a.getMaxAllocatableVramMb() / 4096L;
                long tierB = b.getMaxAllocatableVramMb() / 4096L;
                if (tierA != tierB) {
                    return Long.compare(tierA, tierB);
                }
                int scoreCmp = Double.compare(computeWorkerComputeScore(b), computeWorkerComputeScore(a));
                if (scoreCmp != 0) {
                    return scoreCmp;
                }
                return Long.compare(a.getMaxAllocatableVramMb(), b.getMaxAllocatableVramMb());
            })
            .collect(Collectors.toList());
        if (!singleGpuFits.isEmpty()) {
            return List.of(singleGpuFits.get(0));
        }

        // Case B: Multi-GPU Host-Aware + TFLOPS Score (requires 2..maxGpusForModel GPUs)
        Map<String, List<GpuWorkerNode>> byHost = new LinkedHashMap<>();
        for (GpuWorkerNode w : healthy) {
            String hk = extractWorkerHostKey(w);
            byHost.computeIfAbsent(hk, k -> new ArrayList<>()).add(w);
        }

        List<Map.Entry<String, List<GpuWorkerNode>>> singleHostCandidates = byHost.entrySet().stream()
            .filter(e -> e.getValue().stream().mapToLong(GpuWorkerNode::getMaxAllocatableVramMb).sum() >= requiredVram)
            .sorted((e1, e2) -> {
                double s1 = e1.getValue().stream().mapToDouble(this::computeWorkerComputeScore).sum();
                double s2 = e2.getValue().stream().mapToDouble(this::computeWorkerComputeScore).sum();
                return Double.compare(s2, s1);
            })
            .collect(Collectors.toList());

        List<GpuWorkerNode> orderedPool;
        if (!singleHostCandidates.isEmpty()) {
            orderedPool = new ArrayList<>(singleHostCandidates.get(0).getValue());
            orderedPool.sort((a, b) -> Double.compare(computeWorkerComputeScore(b), computeWorkerComputeScore(a)));
        } else {
            List<Map.Entry<String, List<GpuWorkerNode>>> sortedHosts = new ArrayList<>(byHost.entrySet());
            sortedHosts.sort((e1, e2) -> Double.compare(
                e2.getValue().stream().mapToDouble(this::computeWorkerComputeScore).sum(),
                e1.getValue().stream().mapToDouble(this::computeWorkerComputeScore).sum()
            ));
            orderedPool = new ArrayList<>(healthy.size());
            for (Map.Entry<String, List<GpuWorkerNode>> entry : sortedHosts) {
                List<GpuWorkerNode> hostWorkers = new ArrayList<>(entry.getValue());
                hostWorkers.sort((a, b) -> Double.compare(computeWorkerComputeScore(b), computeWorkerComputeScore(a)));
                orderedPool.addAll(hostWorkers);
            }
        }

        List<GpuWorkerNode> selected = new ArrayList<>();
        long accumulated = 0L;
        for (int i = 0; i < orderedPool.size() && selected.size() < maxGpusForModel; i++) {
            long deficit = requiredVram - accumulated;
            if (deficit <= 0) {
                break;
            }
            List<GpuWorkerNode> remaining = orderedPool.subList(i, orderedPool.size());
            String preferredHost = selected.isEmpty() ? "" : extractWorkerHostKey(selected.get(0));
            List<GpuWorkerNode> closingFits = remaining.stream()
                .filter(w -> w.getMaxAllocatableVramMb() >= deficit)
                .sorted((a, b) -> {
                    boolean aSame = !preferredHost.isEmpty() && preferredHost.equals(extractWorkerHostKey(a));
                    boolean bSame = !preferredHost.isEmpty() && preferredHost.equals(extractWorkerHostKey(b));
                    if (aSame != bSame) {
                        return aSame ? -1 : 1;
                    }
                    long tierA = a.getMaxAllocatableVramMb() / 4096L;
                    long tierB = b.getMaxAllocatableVramMb() / 4096L;
                    if (tierA != tierB) {
                        return Long.compare(tierA, tierB);
                    }
                    return Double.compare(computeWorkerComputeScore(b), computeWorkerComputeScore(a));
                })
                .collect(Collectors.toList());

            if (!closingFits.isEmpty()) {
                GpuWorkerNode bestCloser = closingFits.get(0);
                selected.add(bestCloser);
                accumulated += bestCloser.getMaxAllocatableVramMb();
                break;
            }

            GpuWorkerNode nextBest = orderedPool.get(i);
            selected.add(nextBest);
            accumulated += nextBest.getMaxAllocatableVramMb();
        }

        return (accumulated >= requiredVram && selected.size() <= maxGpusForModel) ? selected : List.of();
    }

    public int computeAutoMaximizedContext(long totalAllocatable, long requiredVram, DbModelShardingPolicy policy) {
        long kvPer1k = Math.max(8L, policy.kvVramPer1kTokensMb());
        long baseFootprintMb;
        if (!policy.layerWeightsMib().isEmpty()) {
            double layersSum = policy.layerWeightsMib().stream().mapToDouble(Double::doubleValue).sum();
            baseFootprintMb = Math.round(policy.nonBlockMasterWeightsMib() + layersSum + 1024.0);
        } else {
            long fullPolicyKvMb = (long) Math.ceil((policy.maxContextLength() / 1024.0) * kvPer1k);
            baseFootprintMb = (requiredVram > fullPolicyKvMb + 4096L)
                ? (requiredVram - fullPolicyKvMb)
                : requiredVram;
        }
        long residualForKvMb = Math.max(512L, totalAllocatable - baseFootprintMb);
        int rawMaxContext = (int) ((residualForKvMb * 1024L) / kvPer1k);
        return Math.min(policy.maxContextLength(), Math.max(4096, (rawMaxContext / 1024) * 1024));
    }

    public List<GpuWorkerNode> orderWorkersByHostLocality(
        List<GpuWorkerNode> rawWorkers,
        double stage0OverheadMb
    ) {
        if (rawWorkers == null) {
            return List.of();
        }
        if (rawWorkers.size() <= 1) {
            return rawWorkers;
        }
        List<GpuWorkerNode> ordered = new ArrayList<>(rawWorkers);
        ordered.sort((a, b) -> Double.compare(
            computeStage0MasterScore(b, stage0OverheadMb),
            computeStage0MasterScore(a, stage0OverheadMb)
        ));
        if (ordered.size() == 2) {
            return ordered;
        }
        String masterHostKey = extractWorkerHostKey(ordered.get(0));
        ordered.subList(1, ordered.size()).sort((a, b) -> {
            boolean aSameHost = masterHostKey.equals(extractWorkerHostKey(a));
            boolean bSameHost = masterHostKey.equals(extractWorkerHostKey(b));
            if (aSameHost != bSameHost) {
                return aSameHost ? -1 : 1;
            }
            return Double.compare(computeWorkerComputeScore(b), computeWorkerComputeScore(a));
        });
        return ordered;
    }

    public String extractWorkerHostKey(GpuWorkerNode w) {
        if (w == null) {
            return "";
        }
        String tk = w.getIrohTicket();
        if (tk != null && tk.startsWith("iroh://")) {
            try {
                String hex = tk.substring("iroh://".length()).trim();
                byte[] bytes = new byte[hex.length() / 2];
                for (int i = 0; i < bytes.length; i++) {
                    bytes[i] = (byte) Integer.parseInt(hex.substring(i * 2, i * 2 + 2), 16);
                }
                JsonNode root = objectMapper.readTree(new String(bytes, StandardCharsets.UTF_8));
                for (JsonNode addr : root.path("addrs")) {
                    String ipPort = addr.path("Ip").asString("");
                    int colon = ipPort.lastIndexOf(':');
                    if (colon > 0) {
                        String ip = ipPort.substring(0, colon);
                        if (!ip.startsWith("172.") && !ip.startsWith("10.") && !ip.startsWith("192.168.") && !ip.startsWith("127.")) {
                            return ip;
                        }
                    }
                }
            } catch (Exception ignored) {
            }
        }
        String name = w.getNodeName() != null ? w.getNodeName() : "";
        int dash = name.lastIndexOf('-');
        return dash > 0 ? name.substring(0, dash) : name;
    }
}
