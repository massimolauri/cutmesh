package io.cutmesh.controlplane.orchestration;

import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface CatalogModelRepository extends JpaRepository<CatalogModel, String> {
    List<CatalogModel> findAllByOrderByPriorityOrderAsc();
}

