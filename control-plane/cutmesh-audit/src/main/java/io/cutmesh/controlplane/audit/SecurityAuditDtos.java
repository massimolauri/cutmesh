package io.cutmesh.controlplane.audit;

import tools.jackson.databind.JsonNode;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;

public final class SecurityAuditDtos {

    private SecurityAuditDtos() {
    }

    public record RunCovariantAuditRequest(
        @Min(64) @Max(8192) int dimension,
        Long seed,
        String secretSeed,
        String modelId
    ) {
    }

    public record CovariantAuditReportDto(
        String auditEventId,
        int dimension,
        int intermediateDimension,
        double f32RelativeErr,
        double q80ToQ50RelativeErr,
        double q80CanaryRelativeErr,
        double rank1EmpiricalCorrMax,
        boolean antiCollusionEnforced,
        boolean canaryVerified,
        boolean auditPassed,
        String executionSource,
        long elapsedMs,
        JsonNode rawProtocolAudit
    ) {
    }
}
