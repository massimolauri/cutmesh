package io.cutmesh.controlplane.gateway;

import io.cutmesh.controlplane.shared.ClusterStateProvider;
import java.util.Map;
import org.springframework.stereotype.Component;

/**
 * Pluggable state store for sticky conversation-to-rig KV cache affinity.
 * Backed by `ClusterStateProvider`: automatically uses Hazelcast `IMap` when
 * `cutmesh.cluster.hazelcast.enabled=true` for multi-pod Kubernetes clusters,
 * or a bounded in-memory LRU cache in single-node deployments.
 */
public interface RouterStateStore {

    String getPinnedRigId(String conversationId);

    void pinConversationToRig(String conversationId, String rigId);

    int getActiveStickyConversationCount();

    @Component
    class ClusterBackedRouterStateStore implements RouterStateStore {

        private final Map<String, String> stickyConversationToRig;

        public ClusterBackedRouterStateStore(ClusterStateProvider clusterStateProvider) {
            this.stickyConversationToRig = clusterStateProvider.getBoundedLruMap("cutmesh-sticky-conversations", 2048);
        }

        @Override
        public String getPinnedRigId(String conversationId) {
            if (conversationId == null || conversationId.isBlank()) {
                return null;
            }
            return stickyConversationToRig.get(conversationId);
        }

        @Override
        public void pinConversationToRig(String conversationId, String rigId) {
            if (conversationId != null && !conversationId.isBlank() && rigId != null && !rigId.isBlank()) {
                stickyConversationToRig.put(conversationId, rigId);
            }
        }

        @Override
        public int getActiveStickyConversationCount() {
            return stickyConversationToRig.size();
        }
    }
}
