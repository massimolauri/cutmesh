package io.cutmesh.controlplane.gateway;

import io.cutmesh.controlplane.orchestration.CatalogModel;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.ChatCompletionRequest;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.ChatCompletionResponse;
import io.cutmesh.controlplane.orchestration.HybridRigOrchestratorService;

import jakarta.validation.Valid;
import java.util.List;
import java.util.Map;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * OpenAI-compatible `/v1/models` and `/v1/chat/completions` gateway protected by `Bearer sk-cutmesh-...` or JWT.
 */
@RestController
@RequestMapping("/v1")
public class OpenAiV1GatewayController {

    private final OpenAiGatewayService openAiGatewayService;
    private final HybridRigOrchestratorService orchestratorService;

    public OpenAiV1GatewayController(
        OpenAiGatewayService openAiGatewayService,
        HybridRigOrchestratorService orchestratorService
    ) {
        this.openAiGatewayService = openAiGatewayService;
        this.orchestratorService = orchestratorService;
    }

    @GetMapping("/models")
    public Map<String, Object> listOpenAiModels(
        @RequestHeader(value = "Authorization", required = false) String authorizationHeader
    ) {
        openAiGatewayService.authenticateBearerOrJwt(authorizationHeader);
        List<CatalogModel> models = orchestratorService.listCatalogModels();
        List<Map<String, Object>> data = models.stream()
            .map(m -> Map.<String, Object>of(
                "id", m.getModelId(),
                "object", "model",
                "created", m.getCreatedAt().getEpochSecond(),
                "owned_by", "cutmesh-covariant-swarm",
                "required_vram_mb", m.getRequiredVramMb(),
                "security_profile", m.getSecurityProfile().name()
            ))
            .toList();
        return Map.of("object", "list", "data", data);
    }

    @PostMapping("/chat/completions")
    public ChatCompletionResponse createChatCompletion(
        @RequestHeader(value = "Authorization", required = false) String authorizationHeader,
        @Valid @RequestBody ChatCompletionRequest request
    ) {
        return openAiGatewayService.handleChatCompletion(authorizationHeader, request);
    }

    @PostMapping(value = "/chat/completions/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter createChatCompletionStream(
        @RequestHeader(value = "Authorization", required = false) String authorizationHeader,
        @Valid @RequestBody ChatCompletionRequest request
    ) {
        return openAiGatewayService.handleChatCompletionStream(authorizationHeader, request);
    }
}

