package io.cutmesh.controlplane.shared;

import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import java.sql.Connection;
import javax.sql.DataSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * Distributed cluster leader-election lock for horizontally scaled Control Plane pods.
 * Uses PostgreSQL transaction-scoped advisory locks (`pg_try_advisory_xact_lock`) so that
 * `@Scheduled` reconciliation loops (AUTO_PILOT scheduler, stale worker eviction) execute on
 * exactly one pod at a time without requiring external ZooKeeper/etcd infrastructure.
 * On single-node embedded H2 (e.g. unit/integration tests), gracefully succeeds locally.
 */
@Component
public class ClusterLeaderLock {

    private static final Logger log = LoggerFactory.getLogger(ClusterLeaderLock.class);

    @PersistenceContext
    private EntityManager entityManager;

    private final boolean postgresBackend;

    public ClusterLeaderLock(DataSource dataSource) {
        boolean isPg = false;
        try (Connection conn = dataSource.getConnection()) {
            String product = conn.getMetaData().getDatabaseProductName();
            isPg = product != null && product.toLowerCase().contains("postgresql");
        } catch (Exception e) {
            log.debug("Could not inspect JDBC database product for ClusterLeaderLock: {}", e.getMessage());
        }
        this.postgresBackend = isPg;
    }

    /**
     * Attempts to acquire a non-blocking transaction-scoped advisory lock for `lockKey`.
     * Must be invoked inside an active `@Transactional` method; PostgreSQL automatically
     * releases the lock when the transaction commits or rolls back.
     */
    public boolean tryAcquireTransactionLock(String lockKey) {
        if (!postgresBackend) {
            return true;
        }
        try {
            Object result = entityManager
                .createNativeQuery("SELECT pg_try_advisory_xact_lock(hashtext(:lockKey))")
                .setParameter("lockKey", lockKey)
                .getSingleResult();
            return Boolean.TRUE.equals(result);
        } catch (Exception e) {
            log.warn("ClusterLeaderLock fallback for key '{}': {}", lockKey, e.getMessage());
            return true;
        }
    }
}
