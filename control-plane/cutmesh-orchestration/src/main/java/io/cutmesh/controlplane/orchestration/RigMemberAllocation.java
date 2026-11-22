package io.cutmesh.controlplane.orchestration;

import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.util.UUID;

/**
 * Per-worker proportional VRAM allocation, layer range, and role pinning inside a ManagedRig.
 */
@Entity
@Table(name = "rig_member_allocations")
public class RigMemberAllocation {

    @Id
    @Column(name = "id", nullable = false, updatable = false, length = 36)
    private String id;

    @JsonIgnore
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "rig_id", nullable = false)
    private ManagedRig rig;

    @Column(name = "worker_node_id", nullable = false, length = 36)
    private String workerNodeId;

    @Column(name = "iroh_node_id", nullable = false, length = 128)
    private String irohNodeId;

    @Column(name = "gpu_name", nullable = false, length = 160)
    private String gpuName;

    @Column(name = "pipeline_stage_index", nullable = false)
    private int pipelineStageIndex;

    @Column(name = "assigned_role", nullable = false, length = 48)
    private String assignedRole;

    @Column(name = "allocated_vram_mb", nullable = false)
    private long allocatedVramMb;

    @Column(name = "tensor_split_fraction", nullable = false)
    private double tensorSplitFraction;

    @Column(name = "layer_start", nullable = false)
    private int layerStart;

    @Column(name = "layer_end", nullable = false)
    private int layerEnd;

    @Column(name = "byte_range_start", nullable = false)
    private long byteRangeStart;

    @Column(name = "byte_range_end", nullable = false)
    private long byteRangeEnd;

    public RigMemberAllocation() {
    }

    @PrePersist
    public void prePersist() {
        if (this.id == null) {
            this.id = UUID.randomUUID().toString();
        }
    }

    public String getId() {
        return id;
    }

    public ManagedRig getRig() {
        return rig;
    }

    public void setRig(ManagedRig rig) {
        this.rig = rig;
    }

    public String getWorkerNodeId() {
        return workerNodeId;
    }

    public void setWorkerNodeId(String workerNodeId) {
        this.workerNodeId = workerNodeId;
    }

    public String getIrohNodeId() {
        return irohNodeId;
    }

    public void setIrohNodeId(String irohNodeId) {
        this.irohNodeId = irohNodeId;
    }

    public String getGpuName() {
        return gpuName;
    }

    public void setGpuName(String gpuName) {
        this.gpuName = gpuName;
    }

    public int getPipelineStageIndex() {
        return pipelineStageIndex;
    }

    public void setPipelineStageIndex(int pipelineStageIndex) {
        this.pipelineStageIndex = pipelineStageIndex;
    }

    public String getAssignedRole() {
        return assignedRole;
    }

    public void setAssignedRole(String assignedRole) {
        this.assignedRole = assignedRole;
    }

    public long getAllocatedVramMb() {
        return allocatedVramMb;
    }

    public void setAllocatedVramMb(long allocatedVramMb) {
        this.allocatedVramMb = allocatedVramMb;
    }

    public double getTensorSplitFraction() {
        return tensorSplitFraction;
    }

    public void setTensorSplitFraction(double tensorSplitFraction) {
        this.tensorSplitFraction = tensorSplitFraction;
    }

    public int getLayerStart() {
        return layerStart;
    }

    public void setLayerStart(int layerStart) {
        this.layerStart = layerStart;
    }

    public int getLayerEnd() {
        return layerEnd;
    }

    public void setLayerEnd(int layerEnd) {
        this.layerEnd = layerEnd;
    }

    public long getByteRangeStart() {
        return byteRangeStart;
    }

    public void setByteRangeStart(long byteRangeStart) {
        this.byteRangeStart = byteRangeStart;
    }

    public long getByteRangeEnd() {
        return byteRangeEnd;
    }

    public void setByteRangeEnd(long byteRangeEnd) {
        this.byteRangeEnd = byteRangeEnd;
    }
}
