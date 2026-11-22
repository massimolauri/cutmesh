package io.cutmesh.controlplane.shared;

import com.hazelcast.config.Config;
import com.hazelcast.config.EvictionConfig;
import com.hazelcast.config.EvictionPolicy;
import com.hazelcast.config.JoinConfig;
import com.hazelcast.config.MapConfig;
import com.hazelcast.config.MaxSizePolicy;
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

/**
 * Embedded Hazelcast cluster configuration for multi-pod Kubernetes Control Plane deployments.
 * Enabled via `CUTMESH_HAZELCAST_ENABLED=true` (`cutmesh.cluster.hazelcast.enabled=true`).
 * Supports Kubernetes DNS discovery (`CUTMESH_HAZELCAST_K8S_SERVICE`) or TCP/Multicast fallback.
 */
@Configuration
@ConditionalOnProperty(name = "cutmesh.cluster.hazelcast.enabled", havingValue = "true")
public class HazelcastClusterConfig {

    private static final Logger log = LoggerFactory.getLogger(HazelcastClusterConfig.class);

    @Bean(destroyMethod = "shutdown")
    public HazelcastInstance cutmeshHazelcastInstance(
        @Value("${cutmesh.cluster.hazelcast.cluster-name:cutmesh-control-plane}") String clusterName,
        @Value("${cutmesh.cluster.hazelcast.k8s-service-name:}") String k8sServiceName,
        @Value("${cutmesh.cluster.hazelcast.k8s-namespace:}") String k8sNamespace,
        @Value("${cutmesh.cluster.hazelcast.tcp-members:}") String tcpMembers
    ) {
        Config config = new Config();
        config.setClusterName(clusterName);
        config.setProperty("hazelcast.logging.type", "slf4j");

        MapConfig stickyMapConfig = new MapConfig("cutmesh-sticky-conversations")
            .setBackupCount(1)
            .setTimeToLiveSeconds(7200)
            .setEvictionConfig(new EvictionConfig()
                .setEvictionPolicy(EvictionPolicy.LRU)
                .setMaxSizePolicy(MaxSizePolicy.PER_NODE)
                .setSize(4096));
        config.addMapConfig(stickyMapConfig);

        MapConfig rigFsmConfig = new MapConfig("cutmesh-rig-fsm-state")
            .setBackupCount(1)
            .setTimeToLiveSeconds(86400);
        config.addMapConfig(rigFsmConfig);

        MapConfig rigLogConfig = new MapConfig("cutmesh-rig-last-log")
            .setBackupCount(1)
            .setTimeToLiveSeconds(86400);
        config.addMapConfig(rigLogConfig);

        JoinConfig join = config.getNetworkConfig().getJoin();
        join.getMulticastConfig().setEnabled(false);

        if (k8sServiceName != null && !k8sServiceName.isBlank()) {
            join.getKubernetesConfig().setEnabled(true)
                .setProperty("service-dns", k8sServiceName.trim());
            if (k8sNamespace != null && !k8sNamespace.isBlank()) {
                join.getKubernetesConfig().setProperty("namespace", k8sNamespace.trim());
            }
            log.info("Hazelcast cluster '{}' configured with Kubernetes DNS discovery ({})", clusterName, k8sServiceName);
        } else if (tcpMembers != null && !tcpMembers.isBlank()) {
            join.getTcpIpConfig().setEnabled(true);
            for (String member : tcpMembers.split(",")) {
                String trimmed = member.trim();
                if (!trimmed.isBlank()) {
                    join.getTcpIpConfig().addMember(trimmed);
                }
            }
            log.info("Hazelcast cluster '{}' configured with TCP-IP members ({})", clusterName, tcpMembers);
        } else {
            join.getMulticastConfig().setEnabled(true);
            log.info("Hazelcast cluster '{}' configured with Multicast auto-discovery", clusterName);
        }

        return Hazelcast.newHazelcastInstance(config);
    }

    @Bean
    @Primary
    public ClusterStateProvider hazelcastClusterStateProvider(HazelcastInstance hazelcastInstance) {
        return new ClusterStateProvider() {
            @Override
            public <K, V> Map<K, V> getMap(String mapName) {
                return hazelcastInstance.getMap(mapName);
            }

            @Override
            public <K, V> Map<K, V> getBoundedLruMap(String mapName, int maxEntries) {
                return hazelcastInstance.getMap(mapName);
            }

            @Override
            public boolean isDistributed() {
                return true;
            }
        };
    }
}
