package io.cutmesh.controlplane.api;

import io.cutmesh.controlplane.iam.JwtAuthInterceptor;
import io.cutmesh.controlplane.orchestration.CatalogModel;
import io.cutmesh.controlplane.fleet.GpuWorkerNode;
import io.cutmesh.controlplane.orchestration.ManagedRig;
import io.cutmesh.controlplane.fleet.NodeEnrollmentToken;
import io.cutmesh.controlplane.gateway.OpenAiApiKey;
import io.cutmesh.controlplane.audit.SecurityAuditEvent;
import io.cutmesh.controlplane.orchestration.RigRuntimePhaseState;
import io.cutmesh.controlplane.iam.AuthDtos.ApiKeyLoginRequest;
import io.cutmesh.controlplane.iam.AuthDtos.EmailPasswordLoginRequest;
import io.cutmesh.controlplane.iam.AuthDtos.GoogleOAuthLoginRequest;
import io.cutmesh.controlplane.iam.AuthDtos.RegisterUserRequest;
import io.cutmesh.controlplane.iam.AuthDtos.UpdateLocaleRequest;
import io.cutmesh.controlplane.iam.AuthDtos.UpdateRoleRequest;
import io.cutmesh.controlplane.iam.AuthDtos.UserProfileResponse;
import io.cutmesh.controlplane.orchestration.CatalogAndRigDtos.CompileCovariantRequest;
import io.cutmesh.controlplane.orchestration.CatalogAndRigDtos.CovariantCompileResponse;
import io.cutmesh.controlplane.orchestration.CatalogAndRigDtos.CreateCatalogModelRequest;
import io.cutmesh.controlplane.orchestration.CatalogAndRigDtos.ManualCreateRigRequest;
import io.cutmesh.controlplane.orchestration.CatalogAndRigDtos.UpdateModelOrchestrationRequest;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.CreateOpenAiKeyRequest;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.CreatedOpenAiKeyResponse;
import io.cutmesh.controlplane.audit.SecurityAuditDtos.CovariantAuditReportDto;
import io.cutmesh.controlplane.audit.SecurityAuditDtos.RunCovariantAuditRequest;
import io.cutmesh.controlplane.fleet.WorkerAndEnrollmentDtos.CreateEnrollmentTokenRequest;
import io.cutmesh.controlplane.fleet.WorkerAndEnrollmentDtos.EnrollmentTokenIssuedResponse;
import io.cutmesh.controlplane.fleet.FleetAndEnrollmentService;
import io.cutmesh.controlplane.orchestration.HybridRigOrchestratorService;
import io.cutmesh.controlplane.gateway.OpenAiGatewayService;
import io.cutmesh.controlplane.audit.SecurityAuditService;
import io.cutmesh.controlplane.iam.UserAccountService;
import jakarta.validation.Valid;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestAttribute;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * Primary Control Plane REST API consumed by the Next.js Web Dashboard.
 */
@RestController
@RequestMapping("/api/v1")
public class ControlPlaneApiController {

    private final UserAccountService userAccountService;
    private final FleetAndEnrollmentService fleetService;
    private final HybridRigOrchestratorService orchestratorService;
    private final OpenAiGatewayService openAiGatewayService;
    private final SecurityAuditService auditService;

    public ControlPlaneApiController(
        UserAccountService userAccountService,
        FleetAndEnrollmentService fleetService,
        HybridRigOrchestratorService orchestratorService,
        OpenAiGatewayService openAiGatewayService,
        SecurityAuditService auditService
    ) {
        this.userAccountService = userAccountService;
        this.fleetService = fleetService;
        this.orchestratorService = orchestratorService;
        this.openAiGatewayService = openAiGatewayService;
        this.auditService = auditService;
    }

    // =========================================================================
    // 0. PUBLIC HEALTH STATUS & AUTHENTICATION / USER/ROLE MANAGEMENT
    // =========================================================================

    @GetMapping("/public/status")
    public Map<String, Object> publicStatus() {
        return Map.of(
            "status", "UP",
            "service", "cutmesh-control-plane",
            "googleOAuthEnabled", userAccountService.isGoogleOAuthEnabled(),
            "timestamp", java.time.Instant.now().toString()
        );
    }

    @GetMapping("/auth/me")
    public UserProfileResponse getCurrentUser(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId
    ) {
        return userAccountService.getUserByIdOrDefault(userId);
    }

    @PostMapping("/auth/login")
    public UserProfileResponse loginWithEmailPassword(@Valid @RequestBody EmailPasswordLoginRequest request) {
        return userAccountService.loginWithEmailPassword(request);
    }

    @PostMapping("/auth/register")
    public UserProfileResponse registerUser(@Valid @RequestBody RegisterUserRequest request) {
        return userAccountService.registerLocalUser(request);
    }

    @PostMapping("/auth/google")
    public UserProfileResponse loginWithGoogle(@RequestBody GoogleOAuthLoginRequest request) {
        return userAccountService.loginOrRegisterGoogleUser(request);
    }

    @PostMapping("/auth/api-key")
    public UserProfileResponse loginWithApiKey(@Valid @RequestBody ApiKeyLoginRequest request) {
        return userAccountService.loginWithApiKey(request.apiKey());
    }

    @PutMapping("/auth/locale")
    public UserProfileResponse updateLocale(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @Valid @RequestBody UpdateLocaleRequest request
    ) {
        userAccountService.requireRole(userId);
        return userAccountService.updatePreferredLocale(userId, request.preferredLocale());
    }

    @PutMapping("/auth/role")
    public UserProfileResponse updateRole(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @Valid @RequestBody UpdateRoleRequest request
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        return userAccountService.updateUserRole(userId, request.role());
    }

    @GetMapping("/users")
    public List<UserProfileResponse> listAllUsers(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        return userAccountService.listAllUsers();
    }

    @PutMapping("/users/{targetUserId}/role")
    public UserProfileResponse updateTargetUserRole(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String targetUserId,
        @Valid @RequestBody UpdateRoleRequest request
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        return userAccountService.updateUserRole(targetUserId, request.role());
    }

    @DeleteMapping("/users/{targetUserId}")
    public Map<String, Object> deleteTargetUser(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String targetUserId
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        userAccountService.deleteUser(targetUserId);
        return Map.of("deleted", true, "userId", targetUserId);
    }

    // =========================================================================
    // 2. GPU PROVIDER FLOW: DOCKER ENROLLMENT TOKENS, FLEET & WATCHDOG CHAOS TEST
    // =========================================================================

    @GetMapping("/enrollment-tokens")
    public List<NodeEnrollmentToken> listEnrollmentTokens(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId
    ) {
        var caller = userAccountService.requireRole(userId, "SUPER_ADMIN", "HARDWARE_PROVIDER");
        return fleetService.listTokensForUser(caller.getId(), "SUPER_ADMIN".equals(caller.getRole()));
    }

    @PostMapping("/enrollment-tokens")
    public EnrollmentTokenIssuedResponse createEnrollmentToken(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @Valid @RequestBody CreateEnrollmentTokenRequest request
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN", "HARDWARE_PROVIDER");
        return fleetService.issueEnrollmentToken(userId, request);
    }

    @DeleteMapping("/enrollment-tokens/{tokenId}")
    public Map<String, Object> revokeEnrollmentToken(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String tokenId
    ) {
        var caller = userAccountService.requireRole(userId, "SUPER_ADMIN", "HARDWARE_PROVIDER");
        fleetService.revokeEnrollmentToken(tokenId, caller.getId(), "SUPER_ADMIN".equals(caller.getRole()));
        return Map.of("revoked", true, "tokenId", tokenId);
    }

    @GetMapping("/workers")
    public List<GpuWorkerNode> listAllWorkers(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId
    ) {
        userAccountService.requireRole(userId);
        return fleetService.listAllWorkers();
    }

    @GetMapping("/workers/standby")
    public List<GpuWorkerNode> listStandbyWorkers(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId
    ) {
        userAccountService.requireRole(userId);
        return fleetService.listStandbyWorkers();
    }

    @PostMapping("/workers/{workerId}/test-watchdog")
    public Map<String, Object> triggerWatchdogFaultTest(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String workerId
    ) {
        var caller = userAccountService.requireRole(userId, "SUPER_ADMIN", "HARDWARE_PROVIDER");
        return fleetService.triggerWatchdogFaultTest(workerId, caller.getId(), "SUPER_ADMIN".equals(caller.getRole()));
    }

    @PostMapping("/workers/{workerId}/disconnect")
    public GpuWorkerNode disconnectWorker(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String workerId
    ) {
        var caller = userAccountService.requireRole(userId, "SUPER_ADMIN", "HARDWARE_PROVIDER");
        return fleetService.disconnectWorkerById(workerId, caller.getId(), "SUPER_ADMIN".equals(caller.getRole()));
    }

    @DeleteMapping("/workers/{workerId}")
    public Map<String, Object> deleteWorker(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String workerId
    ) {
        var caller = userAccountService.requireRole(userId, "SUPER_ADMIN", "HARDWARE_PROVIDER");
        fleetService.deleteWorkerNode(workerId, caller.getId(), "SUPER_ADMIN".equals(caller.getRole()));
        return Map.of("deleted", true, "workerId", workerId);
    }

    // =========================================================================
    // 3. MODEL CATALOG & HYBRID AUTO-PILOT / MANUAL RIG ORCHESTRATOR
    // =========================================================================

    @GetMapping("/catalog/models")
    public List<CatalogModel> listCatalogModels(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId
    ) {
        userAccountService.requireRole(userId);
        return orchestratorService.listCatalogModels();
    }

    @GetMapping("/catalog/models/{modelId}/policy")
    public tools.jackson.databind.JsonNode getCatalogModelPolicy(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String modelId
    ) {
        userAccountService.requireRole(userId);
        return orchestratorService.getModelPolicy(modelId);
    }

    @PutMapping("/catalog/models/{modelId}/policy")
    public CatalogModel updateCatalogModelPolicy(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String modelId,
        @RequestBody tools.jackson.databind.JsonNode policyNode
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        return orchestratorService.updateModelPolicy(modelId, policyNode);
    }

    @PostMapping("/catalog/models")
    public CatalogModel registerCatalogModel(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @Valid @RequestBody CreateCatalogModelRequest request
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        return orchestratorService.registerNewCatalogModel(request);
    }

    @DeleteMapping("/catalog/models/{modelId}")
    public Map<String, Object> deleteCatalogModel(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String modelId
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        orchestratorService.deleteCatalogModel(modelId);
        return Map.of("deleted", true, "modelId", modelId);
    }

    @PutMapping("/catalog/models/{modelId}/orchestration")
    public CatalogModel updateModelOrchestration(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String modelId,
        @Valid @RequestBody UpdateModelOrchestrationRequest request
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        return orchestratorService.updateModelOrchestration(modelId, request);
    }

    @PostMapping("/catalog/models/{modelId}/compile-covariant")
    public CovariantCompileResponse compileModelCovariant(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String modelId,
        @RequestBody(required = false) CompileCovariantRequest request
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        return orchestratorService.compileModelCovariantGguf(modelId, request);
    }

    @GetMapping("/rigs")
    public List<ManagedRig> listAllRigs(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @RequestParam(value = "activeOnly", required = false, defaultValue = "false") boolean activeOnly
    ) {
        userAccountService.requireRole(userId);
        List<ManagedRig> rigs = activeOnly
            ? orchestratorService.listServingRigs()
            : orchestratorService.listAllRigs();
        rigs.forEach(this::enrichRigRuntimeMetrics);
        return rigs;
    }

    @GetMapping("/rigs/paged")
    public Map<String, Object> listRigsPaged(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @RequestParam(value = "page", defaultValue = "0") int page,
        @RequestParam(value = "size", defaultValue = "25") int size,
        @RequestParam(value = "state", required = false, defaultValue = "SERVING") String state,
        @RequestParam(value = "modelId", required = false) String modelId,
        @RequestParam(value = "q", required = false) String query
    ) {
        userAccountService.requireRole(userId);
        int clampedSize = Math.max(1, Math.min(200, size));
        int clampedPage = Math.max(0, page);
        org.springframework.data.domain.Page<ManagedRig> result = orchestratorService.searchRigsPaged(
            state,
            modelId,
            query,
            clampedPage,
            clampedSize
        );
        result.getContent().forEach(this::enrichRigRuntimeMetrics);
        return Map.of(
            "items", result.getContent(),
            "page", result.getNumber(),
            "size", result.getSize(),
            "totalElements", result.getTotalElements(),
            "totalPages", result.getTotalPages(),
            "totalServing", orchestratorService.countServingRigs()
        );
    }

    private void enrichRigRuntimeMetrics(ManagedRig rig) {
        int httpCode = openAiGatewayService.getCachedRigEndpointHttpCode(rig.getId());
        boolean procAlive = orchestratorService.isRigProcessAlive(rig.getId());
        boolean isServingState = rig.getLifecycleState().name().equals("SERVING");

        boolean ready = isServingState && httpCode == 200;
        boolean llamaAlive = isServingState && (httpCode == 200 || httpCode == 503 || procAlive);

        RigRuntimePhaseState fsmState = orchestratorService.getRigFsmState(rig.getId());
        String lastLog = orchestratorService.getRigLastLog(rig.getId());
        String statusMsg = fsmState.getTitleIt() + " — " + fsmState.getDescriptionIt()
            + " (" + rig.getInferenceEndpointUrl() + ")";

        rig.setInferenceEngineReady(ready);
        rig.setLlamaProcessAlive(llamaAlive);
        rig.setRuntimePhase(fsmState.getCode());
        rig.setRuntimeStep(fsmState.getStepNumber());
        rig.setRuntimeLastLog(lastLog);
        rig.setActiveInFlightRequests(openAiGatewayService.getInFlightCount(rig.getId()));
        rig.setAvgLatencyMs(Math.round(openAiGatewayService.getRigAvgLatencyMs(rig.getId()) * 10.0) / 10.0);
        rig.setLastPrefillTokPerSec(openAiGatewayService.getRigLastPrefillTokPerSec(rig.getId()));
        rig.setLastDecodeTokPerSec(openAiGatewayService.getRigLastDecodeTokPerSec(rig.getId()));
        rig.setOperationalStatusMessage(statusMsg);
    }

    @GetMapping("/router/status")
    public Map<String, Object> getLoadBalancerRouterStatus(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId
    ) {
        userAccountService.requireRole(userId);
        return openAiGatewayService.getRouterLoadBalancerStatus();
    }

    @PostMapping("/rigs/manual")
    public ManagedRig createManualRig(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @Valid @RequestBody ManualCreateRigRequest request
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        return orchestratorService.createManualRig(request);
    }

    @PostMapping("/rigs/reconcile-autopilot")
    public Map<String, Object> triggerAutoPilotReconcile(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        int provisioned = orchestratorService.reconcileAutoPilotRigs();
        return Map.of("provisionedRigs", provisioned, "activeRigs", orchestratorService.listActiveServingRigs().size());
    }

    @DeleteMapping("/rigs/{rigId}")
    public ManagedRig dissolveRig(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String rigId
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        return orchestratorService.dissolveRig(rigId);
    }

    // =========================================================================
    // 4. OPENAI CONSUMER API KEYS (`sk-cutmesh-...`)
    // =========================================================================

    @GetMapping("/api-keys")
    public List<OpenAiApiKey> listApiKeys(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId
    ) {
        userAccountService.requireRole(userId);
        return openAiGatewayService.listApiKeys(userId);
    }

    @PostMapping("/api-keys")
    public CreatedOpenAiKeyResponse createApiKey(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @Valid @RequestBody CreateOpenAiKeyRequest request
    ) {
        userAccountService.requireRole(userId);
        return openAiGatewayService.createApiKey(userId, request);
    }

    @DeleteMapping("/api-keys/{keyId}")
    public Map<String, Object> revokeApiKey(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @PathVariable String keyId
    ) {
        var caller = userAccountService.requireRole(userId);
        openAiGatewayService.revokeApiKey(keyId, caller.getId(), "SUPER_ADMIN".equals(caller.getRole()));
        return Map.of("revoked", true, "keyId", keyId);
    }

    // =========================================================================
    // 5. ZERO-WEIGHT-DISCLOSURE COVARIANT AUDIT & IMMUTABLE AUDIT TRAIL
    // =========================================================================

    @GetMapping("/audit/events")
    public List<SecurityAuditEvent> listAuditEvents(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @RequestParam(value = "category", required = false) String category
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        return auditService.listRecentAuditEvents(category);
    }

    @PostMapping("/audit/run-covariant")
    public CovariantAuditReportDto runCovariantProtocolAudit(
        @RequestAttribute(JwtAuthInterceptor.AUTHENTICATED_USER_ID_ATTR) String userId,
        @Valid @RequestBody RunCovariantAuditRequest request
    ) {
        userAccountService.requireRole(userId, "SUPER_ADMIN");
        return auditService.executeCovariantProtocolAudit(request);
    }
}
