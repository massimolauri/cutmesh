package io.cutmesh.controlplane.orchestration;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import java.util.List;

public final class CatalogAndRigDtos {

    private CatalogAndRigDtos() {
    }

    public record CreateCatalogModelRequest(
        @NotBlank String modelId,
        @NotBlank String displayName,
        @NotBlank String huggingfaceRepo,
        @NotBlank String baseGgufPath,
        @NotNull SecurityProfile securityProfile,
        @NotNull OrchestrationMode orchestrationMode,
        @Min(0) int minActiveRigs,
        @Min(1) int maxActiveRigs,
        @Min(1) int priorityOrder
    ) {
    }

    public record UpdateModelOrchestrationRequest(
        @NotNull OrchestrationMode orchestrationMode,
        @Min(0) int minActiveRigs,
        @Min(1) int maxActiveRigs,
        @NotNull SecurityProfile securityProfile,
        Integer priorityOrder
    ) {
    }

    public record CompileCovariantRequest(
        String secretSeed,
        Double kappaLambda
    ) {
    }

    public record ManualCreateRigRequest(
        @NotBlank String rigName,
        @NotBlank String modelId,
        @NotEmpty List<String> workerNodeIds
    ) {
    }

    public record CovariantCompileResponse(
        String modelId,
        String inputPath,
        String outputPath,
        String clientKeyManifestPath,
        String profileApplied,
        double kappaLambdaAnisotropy,
        double empiricalRmsnormKappa,
        String outputHeaderBlake3,
        boolean compiledPhysicalFile,
        double elapsedMs,
        double throughputMbPerSec,
        long transformedQ50Rows,
        int calibratedRmsnormTensors,
        String auditEventId
    ) {
    }
}
