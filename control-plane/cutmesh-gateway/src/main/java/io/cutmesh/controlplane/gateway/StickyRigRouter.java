package io.cutmesh.controlplane.gateway;

import io.cutmesh.controlplane.orchestration.ManagedRig;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import org.springframework.stereotype.Component;

/**
 * Single-responsibility load-balancer & sticky-session router for active GPU Rigs:
 * delegates conversation-to-rig KV cache affinity to a pluggable `RouterStateStore`
 * (in-memory LRU by default, Redis/Valkey ready for multi-pod Kubernetes clusters),
 * and tracks in-flight counters, EWMA latency, and live prefill/decode throughput per rig.
 */
@Component
public class StickyRigRouter {

    private final RouterStateStore stateStore;
    private final Map<String, AtomicInteger> inFlightByRigId = new ConcurrentHashMap<>();
    private final Map<String, Double> ewmaLatencyMsByRigId = new ConcurrentHashMap<>();
    private final Map<String, Double> lastPrefillTokPerSecByRigId = new ConcurrentHashMap<>();
    private final Map<String, Double> lastDecodeTokPerSecByRigId = new ConcurrentHashMap<>();

    public StickyRigRouter(RouterStateStore stateStore) {
        this.stateStore = stateStore;
    }

    public String getPinnedRigId(String conversationId) {
        return stateStore.getPinnedRigId(conversationId);
    }

    public void pinConversationToRig(String conversationId, String rigId) {
        stateStore.pinConversationToRig(conversationId, rigId);
    }

    public int getActiveStickyConversationCount() {
        return stateStore.getActiveStickyConversationCount();
    }

    /**
     * Deterministic Rendezvous-hash selector for stateless multi-pod KV affinity:
     * when a pod has not yet cached `conversationId` locally, all pods still converge
     * on the same healthy Rig replica for that conversation.
     */
    public ManagedRig selectConsistentRigForConversation(
        String conversationId,
        List<ManagedRig> candidates,
        Predicate<ManagedRig> isHealthy
    ) {
        if (conversationId == null || conversationId.isBlank() || candidates.size() <= 1) {
            return selectLeastLoadedRig(candidates, isHealthy);
        }
        List<ManagedRig> healthy = candidates.stream().filter(isHealthy).toList();
        List<ManagedRig> pool = healthy.isEmpty() ? candidates : healthy;
        return pool.stream()
            .max(Comparator.comparingInt(r -> (conversationId + ":" + r.getId()).hashCode()))
            .orElse(pool.get(0));
    }

    public ManagedRig selectLeastLoadedRig(List<ManagedRig> candidates, Predicate<ManagedRig> isHealthy) {
        return candidates.stream()
            .min(Comparator.comparingDouble(r -> computeRigLoadScore(r, isHealthy.test(r))))
            .orElse(candidates.get(0));
    }

    public List<ManagedRig> sortRigsByLoadScore(
        List<ManagedRig> candidates,
        String preferredRigId,
        Predicate<ManagedRig> isHealthy
    ) {
        return candidates.stream()
            .sorted((a, b) -> {
                if (a.getId().equals(preferredRigId)) return -1;
                if (b.getId().equals(preferredRigId)) return 1;
                return Double.compare(
                    computeRigLoadScore(a, isHealthy.test(a)),
                    computeRigLoadScore(b, isHealthy.test(b))
                );
            })
            .toList();
    }

    /**
     * Score = ((inFlight + 1) * ewmaLatencyMs) / max(1.0, allocatedVramGb) + UnhealthyPenalty.
     * Healthy rigs with fewer in-flight requests, lower latency, and larger VRAM capacity win.
     */
    public double computeRigLoadScore(ManagedRig rig, boolean healthy) {
        int inFlight = getInFlightCount(rig.getId());
        double ewmaMs = getRigAvgLatencyMs(rig.getId());
        double capacityGb = Math.max(1.0, rig.getTotalAllocatedVramMb() / 1024.0);
        double healthPenalty = healthy ? 0.0 : 1_000_000.0;
        return healthPenalty + (((inFlight + 1) * ewmaMs) / capacityGb);
    }

    public int getInFlightCount(String rigId) {
        AtomicInteger counter = inFlightByRigId.get(rigId);
        return counter != null ? Math.max(0, counter.get()) : 0;
    }

    public double getRigAvgLatencyMs(String rigId) {
        return ewmaLatencyMsByRigId.getOrDefault(rigId, 150.0);
    }

    public double getRigLastPrefillTokPerSec(String rigId) {
        return lastPrefillTokPerSecByRigId.getOrDefault(rigId, 0.0);
    }

    public double getRigLastDecodeTokPerSec(String rigId) {
        return lastDecodeTokPerSecByRigId.getOrDefault(rigId, 0.0);
    }

    public void recordRigSpeedMetrics(String rigId, double prefillTokPerSec, double decodeTokPerSec) {
        if (prefillTokPerSec > 0.0) {
            lastPrefillTokPerSecByRigId.put(rigId, Math.round(prefillTokPerSec * 100.0) / 100.0);
        }
        if (decodeTokPerSec > 0.0) {
            lastDecodeTokPerSecByRigId.put(rigId, Math.round(decodeTokPerSec * 100.0) / 100.0);
        }
    }

    public void incrementInFlight(String rigId) {
        inFlightByRigId.computeIfAbsent(rigId, k -> new AtomicInteger(0)).incrementAndGet();
    }

    public void decrementInFlight(String rigId) {
        AtomicInteger c = inFlightByRigId.get(rigId);
        if (c != null) {
            c.updateAndGet(v -> Math.max(0, v - 1));
        }
    }

    public void recordRigLatencyEwma(String rigId, double sampleMs) {
        ewmaLatencyMsByRigId.compute(rigId, (k, prev) ->
            prev == null ? sampleMs : (prev * 0.7 + sampleMs * 0.3)
        );
    }
}
