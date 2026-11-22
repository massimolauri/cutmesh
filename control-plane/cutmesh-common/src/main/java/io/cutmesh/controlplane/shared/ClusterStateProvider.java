package io.cutmesh.controlplane.shared;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.stereotype.Component;

/**
 * Abstraction for cluster-shared key-value state across horizontally scaled Control Plane pods.
 * When Hazelcast is enabled (`cutmesh.cluster.hazelcast.enabled=true`), returns replicated
 * Hazelcast `IMap` instances; otherwise falls back to lightweight local thread-safe maps.
 */
public interface ClusterStateProvider {

    <K, V> Map<K, V> getMap(String mapName);

    <K, V> Map<K, V> getBoundedLruMap(String mapName, int maxEntries);

    boolean isDistributed();

    @Component
    class LocalClusterStateProvider implements ClusterStateProvider {

        private final Map<String, Map<?, ?>> maps = new ConcurrentHashMap<>();

        @Override
        @SuppressWarnings("unchecked")
        public <K, V> Map<K, V> getMap(String mapName) {
            return (Map<K, V>) maps.computeIfAbsent(mapName, k -> new ConcurrentHashMap<>());
        }

        @Override
        @SuppressWarnings("unchecked")
        public <K, V> Map<K, V> getBoundedLruMap(String mapName, int maxEntries) {
            return (Map<K, V>) maps.computeIfAbsent(
                mapName,
                k -> Collections.synchronizedMap(new LinkedHashMap<>(16, 0.75f, true) {
                    @Override
                    protected boolean removeEldestEntry(Map.Entry<Object, Object> eldest) {
                        return size() > maxEntries;
                    }
                })
            );
        }

        @Override
        public boolean isDistributed() {
            return false;
        }
    }
}
