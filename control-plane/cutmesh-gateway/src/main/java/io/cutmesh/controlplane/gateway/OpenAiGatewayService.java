package io.cutmesh.controlplane.gateway;

import io.cutmesh.controlplane.orchestration.HybridRigOrchestratorService;
import io.cutmesh.controlplane.orchestration.RigRuntimePhaseState;
import io.cutmesh.controlplane.audit.SecurityAuditService;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import io.cutmesh.controlplane.iam.AppUser;
import io.cutmesh.controlplane.orchestration.CatalogModel;
import io.cutmesh.controlplane.orchestration.ManagedRig;

import io.cutmesh.controlplane.orchestration.RigMemberAllocation;
import io.cutmesh.controlplane.orchestration.RigLifecycleState;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.ChatCompletionChoice;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.ChatCompletionRequest;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.ChatCompletionResponse;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.ChatCompletionUsage;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.ChatMessageDto;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.CreateOpenAiKeyRequest;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.CreatedOpenAiKeyResponse;
import io.cutmesh.controlplane.gateway.OpenAiGatewayDtos.CutmeshSwarmTelemetry;
import io.cutmesh.controlplane.orchestration.CatalogModelRepository;
import io.cutmesh.controlplane.orchestration.ManagedRigRepository;

import jakarta.annotation.PreDestroy;
import jakarta.persistence.EntityNotFoundException;
import java.io.File;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.time.Instant;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * OpenAI-compatible API Key lifecycle and `/v1/chat/completions` Sticky-Session Router
 * backed by the real `bin/llamacpp/llama-server` CUDA/RPC engine on the GPU Rig.
 */
@Service
public class OpenAiGatewayService {

    private static final Logger log = LoggerFactory.getLogger(OpenAiGatewayService.class);
    private static final int LLAMA_SERVER_PORT = 8089;

    private final OpenAiApiKeyRepository apiKeyRepository;
    private final OpenAiApiKeyService apiKeyService;
    private final StickyRigRouter stickyRigRouter;
    private final ManagedRigRepository rigRepository;
    private final CatalogModelRepository catalogRepository;
    private final HybridRigOrchestratorService orchestratorService;
    private final SecurityAuditService auditService;
    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;

    private final Map<String, List<ChatMessageDto>> conversationHistoryById = boundedLruMap(512);
    private final ConcurrentHashMap<String, Integer> cachedRigHttpCodeById = new ConcurrentHashMap<>();
    private Process llamaServerProcess;

    private static <K, V> Map<K, V> boundedLruMap(int maxEntries) {
        return java.util.Collections.synchronizedMap(new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
                return size() > maxEntries;
            }
        });
    }

    public OpenAiGatewayService(
        OpenAiApiKeyRepository apiKeyRepository,
        OpenAiApiKeyService apiKeyService,
        StickyRigRouter stickyRigRouter,
        ManagedRigRepository rigRepository,
        CatalogModelRepository catalogRepository,
        HybridRigOrchestratorService orchestratorService,
        SecurityAuditService auditService,
        ObjectMapper objectMapper
    ) {
        this.apiKeyRepository = apiKeyRepository;
        this.apiKeyService = apiKeyService;
        this.stickyRigRouter = stickyRigRouter;
        this.rigRepository = rigRepository;
        this.catalogRepository = catalogRepository;
        this.orchestratorService = orchestratorService;
        this.auditService = auditService;
        this.objectMapper = objectMapper;
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3))
            .build();
    }

    @PreDestroy
    public void stopLlamaServer() {
        if (llamaServerProcess != null && llamaServerProcess.isAlive()) {
            llamaServerProcess.destroyForcibly();
        }
    }

    @Transactional
    public CreatedOpenAiKeyResponse createApiKey(String userId, CreateOpenAiKeyRequest request) {
        return apiKeyService.createApiKey(userId, request);
    }

    @Transactional(readOnly = true)
    public List<OpenAiApiKey> listApiKeys(String userId) {
        return apiKeyService.listApiKeys(userId);
    }

    @Transactional
    public void revokeApiKey(String keyId) {
        apiKeyService.revokeApiKey(keyId);
    }

    @Transactional
    public void revokeApiKey(String keyId, String callerUserId, boolean isAdmin) {
        apiKeyService.revokeApiKey(keyId, callerUserId, isAdmin);
    }

    @Transactional
    public Optional<OpenAiApiKey> authenticateApiKeyOnly(String rawKey) {
        return apiKeyService.authenticateApiKeyOnly(rawKey);
    }

    @Transactional
    public Optional<OpenAiApiKey> authenticateBearerOrJwt(String authorizationHeader) {
        return apiKeyService.authenticateBearerOrJwt(authorizationHeader);
    }

    private record SelectedRigContext(ManagedRig rig, boolean stickyHit, CatalogModel modelMeta, List<ManagedRig> failoverPool) {
    }

    private SelectedRigContext resolveTargetRig(ChatCompletionRequest request) {
        orchestratorService.reconcileAutoPilotRigs();

        List<ManagedRig> activeForModel = rigRepository.findByModelIdAndLifecycleState(
            request.model(),
            RigLifecycleState.SERVING
        );

        if (activeForModel.isEmpty()) {
            List<ManagedRig> anyActive = rigRepository.findByLifecycleStateOrderByCreatedAtDesc(RigLifecycleState.SERVING);
            if (anyActive.isEmpty()) {
                throw new ResponseStatusException(
                    HttpStatus.SERVICE_UNAVAILABLE,
                    "Nessun Rig GPU attivo è attualmente in stato SERVING per il modello '" + request.model() + "'."
                );
            }
            activeForModel = anyActive;
        }

        boolean stickyHit = false;
        ManagedRig targetRig = null;
        if (request.conversationId() != null && !request.conversationId().isBlank()) {
            String pinnedRigId = stickyRigRouter.getPinnedRigId(request.conversationId());
            if (pinnedRigId != null) {
                Optional<ManagedRig> match = activeForModel.stream()
                    .filter(r -> r.getId().equals(pinnedRigId))
                    .findFirst();
                if (match.isPresent() && getInFlightCount(match.get().getId()) < 4) {
                    targetRig = match.get();
                    stickyHit = true;
                }
            }
        }

        if (targetRig == null) {
            targetRig = selectLeastLoadedRig(activeForModel);
            stickyRigRouter.pinConversationToRig(request.conversationId(), targetRig.getId());
        }

        CatalogModel modelMeta = catalogRepository.findById(targetRig.getModelId()).orElse(null);
        List<ManagedRig> failoverPool = sortRigsByLoadScore(activeForModel, targetRig.getId());
        return new SelectedRigContext(targetRig, stickyHit, modelMeta, failoverPool);
    }

    private boolean resolveEnableThinking(ChatCompletionRequest request) {
        if (request.reasoningEffort() != null && "off".equalsIgnoreCase(request.reasoningEffort().trim())) {
            return false;
        }
        if (request.enableThinking() != null) {
            return request.enableThinking();
        }
        if (request.chatTemplateKwargs() != null && request.chatTemplateKwargs().get("enable_thinking") instanceof Boolean b) {
            return b;
        }
        return true;
    }

    private String resolveReasoningEffort(ChatCompletionRequest request) {
        String raw = request.reasoningEffort();
        if ((raw == null || raw.isBlank()) && request.chatTemplateKwargs() != null) {
            Object kw = request.chatTemplateKwargs().get("reasoning_effort");
            if (kw instanceof String s && !s.isBlank()) {
                raw = s;
            }
        }
        if (raw == null || raw.isBlank()) {
            return "low";
        }
        String norm = raw.trim().toLowerCase(Locale.ROOT);
        return switch (norm) {
            case "low", "minimal" -> "low";
            case "medium", "med" -> "medium";
            case "high", "xhigh", "max" -> "xhigh";
            default -> "low";
        };
    }

    private Map<String, Object> buildChatTemplateKwargs(ChatCompletionRequest request, boolean enableThinking) {
        Map<String, Object> kwargs = new LinkedHashMap<>();
        if (request.chatTemplateKwargs() != null) {
            kwargs.putAll(request.chatTemplateKwargs());
        }
        kwargs.put("enable_thinking", enableThinking);
        if (enableThinking) {
            kwargs.put("reasoning_effort", resolveReasoningEffort(request));
        }
        return kwargs;
    }

    private int resolveEffectiveMaxTokens(ChatCompletionRequest request, boolean enableThinking) {
        int baseTokens = (request.maxTokens() != null && request.maxTokens() > 0)
            ? request.maxTokens()
            : 4096;
        if (enableThinking) {
            String effort = resolveReasoningEffort(request);
            int thinkingBudget = switch (effort) {
                case "xhigh" -> 8192;
                case "medium" -> 4096;
                default -> 2048;
            };
            return Math.max(baseTokens + thinkingBudget, 4096);
        }
        return Math.max(baseTokens, 256);
    }

    private String resolveBaseEndpoint(ManagedRig rig, CatalogModel modelMeta) {
        String baseEndpoint = rig.getInferenceEndpointUrl();
        if (baseEndpoint != null && !baseEndpoint.isBlank()) {
            return baseEndpoint;
        }
        return "http://127.0.0.1:" + LLAMA_SERVER_PORT;
    }

    private List<ChatMessageDto> resolveEffectiveMessagesWithMemory(ChatCompletionRequest request) {
        List<ChatMessageDto> incoming = request.messages() != null ? request.messages() : List.of();
        String convId = request.conversationId();
        if (convId == null || convId.isBlank()) {
            return incoming;
        }
        boolean hasPriorAssistantTurn = incoming.stream().anyMatch(m -> "assistant".equalsIgnoreCase(m.role()));
        if (hasPriorAssistantTurn) {
            return incoming;
        }
        List<ChatMessageDto> stored = conversationHistoryById.get(convId.trim());
        if (stored == null || stored.isEmpty()) {
            return incoming;
        }
        java.util.ArrayList<ChatMessageDto> merged = new java.util.ArrayList<>();
        Optional<ChatMessageDto> sysMsg = incoming.stream().filter(m -> "system".equalsIgnoreCase(m.role())).findFirst();
        if (sysMsg.isPresent()) {
            merged.add(sysMsg.get());
        } else {
            stored.stream().filter(m -> "system".equalsIgnoreCase(m.role())).findFirst().ifPresent(merged::add);
        }
        for (ChatMessageDto m : stored) {
            if (!"system".equalsIgnoreCase(m.role())) {
                merged.add(m);
            }
        }
        for (ChatMessageDto m : incoming) {
            if (!"system".equalsIgnoreCase(m.role())) {
                merged.add(m);
            }
        }
        return merged;
    }

    private void recordConversationTurnMemory(String convId, List<ChatMessageDto> effectiveMessages, String assistantReply) {
        if (convId == null || convId.isBlank() || assistantReply == null || assistantReply.isBlank()) {
            return;
        }
        String cleaned = assistantReply.replaceAll("(?s)<think>.*?</think>", "").trim();
        if (cleaned.isEmpty()) {
            cleaned = assistantReply.trim();
        }
        java.util.ArrayList<ChatMessageDto> updated = new java.util.ArrayList<>(effectiveMessages);
        updated.add(new ChatMessageDto("assistant", cleaned, null));
        if (updated.size() > 40) {
            List<ChatMessageDto> tail = updated.subList(updated.size() - 38, updated.size());
            java.util.ArrayList<ChatMessageDto> capped = new java.util.ArrayList<>();
            updated.stream().filter(m -> "system".equalsIgnoreCase(m.role())).findFirst().ifPresent(capped::add);
            capped.addAll(tail);
            updated = capped;
        }
        conversationHistoryById.put(convId.trim(), updated);
    }

    public SseEmitter handleChatCompletionStream(String authorizationHeader, ChatCompletionRequest request) {
        Optional<OpenAiApiKey> apiKeyOpt = authenticateBearerOrJwt(authorizationHeader);
        SelectedRigContext ctx = resolveTargetRig(request);
        ManagedRig targetRig = ctx.rig();
        boolean stickyHit = ctx.stickyHit();
        CatalogModel modelMeta = ctx.modelMeta();
        boolean enableThinking = resolveEnableThinking(request);

        List<String> gpuSummaries = targetRig.getAllocations().stream()
            .map(RigMemberAllocation::getGpuName)
            .toList();
        String securityProfileName = modelMeta != null
            ? modelMeta.getSecurityProfile().name()
            : targetRig.getSecurityProfile().name();
        var latestCovariant = auditService.listRecentAuditEvents("COVARIANT_MATHEMATICAL_AUDIT").stream()
            .filter(ev -> ev.getCanaryRelativeErr() != null)
            .findFirst();
        final double canaryDriftL2 = latestCovariant.map(ev -> ev.getCanaryRelativeErr()).orElse(0.0);
        final String canaryStatus = latestCovariant
            .map(ev -> ev.isPassed()
                ? "VERIFIED_COVARIANT_INVARIANT (drift_L2=" + canaryDriftL2 + ")"
                : "COVARIANT_AUDIT_FAILED (drift_L2=" + canaryDriftL2 + ")")
            .orElse("NOT_AUDITED_YET");
        String baseEndpoint = resolveBaseEndpoint(targetRig, modelMeta);

        SseEmitter emitter = new SseEmitter(900_000L);

        CompletableFuture.runAsync(() -> {
            long startNs = System.nanoTime();
            incrementInFlight(targetRig.getId());
            try {
                CutmeshSwarmTelemetry initialTelemetry = new CutmeshSwarmTelemetry(
                    targetRig.getId(),
                    targetRig.getRigName(),
                    stickyHit,
                    securityProfileName,
                    targetRig.getTensorSplitCsv(),
                    targetRig.getAutoMaximizedContextSize(),
                    targetRig.isAntiCollusionVerified(),
                    canaryDriftL2,
                    canaryStatus,
                    0L,
                    gpuSummaries,
                    null,
                    null,
                    null,
                    null,
                    0
                );
                Map<String, Object> initEvent = new LinkedHashMap<>();
                initEvent.put("object", "chat.completion.chunk");
                initEvent.put("choices", List.of());
                initEvent.put("cutmesh_telemetry", initialTelemetry);
                emitter.send(SseEmitter.event().data(objectMapper.writeValueAsString(initEvent), MediaType.APPLICATION_JSON));

                List<ChatMessageDto> effectiveMessages = resolveEffectiveMessagesWithMemory(request);
                int effectiveMaxTokens = resolveEffectiveMaxTokens(request, enableThinking);
                Map<String, Object> payload = new LinkedHashMap<>();
                payload.put("model", request.model());
                payload.put("messages", effectiveMessages);
                payload.put("temperature", request.temperature() != null ? request.temperature() : 0.6);
                payload.put("max_tokens", effectiveMaxTokens);
                payload.put("n_predict", effectiveMaxTokens);
                payload.put("stream", true);
                payload.put("cache_prompt", true);
                payload.put("timings_per_token", true);
                payload.put("chat_template_kwargs", buildChatTemplateKwargs(request, enableThinking));
                if (!enableThinking) {
                    payload.put("reasoning_budget", 0);
                    payload.put("reasoning_format", "none");
                }

                String bodyJson = objectMapper.writeValueAsString(payload);
                HttpRequest httpReq = HttpRequest.newBuilder()
                    .uri(URI.create(baseEndpoint + "/v1/chat/completions"))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(900))
                    .POST(HttpRequest.BodyPublishers.ofString(bodyJson))
                    .build();

                HttpResponse<Stream<String>> resp = httpClient.send(httpReq, HttpResponse.BodyHandlers.ofLines());
                if (resp.statusCode() != 200) {
                    throw new IllegalStateException("Motore llama-server ha risposto con HTTP " + resp.statusCode());
                }

                final int[] promptTokensHolder = {0};
                final int[] predictedTokensHolder = {0};
                final int[] thinkingTokensHolder = {0};
                final double[] promptMsHolder = {0.0};
                final double[] predictedMsHolder = {0.0};
                final double[] promptPerSecHolder = {0.0};
                final double[] predictedPerSecHolder = {0.0};
                StringBuilder assistantContentBuilder = new StringBuilder();

                try (Stream<String> lines = resp.body()) {
                    for (String line : (Iterable<String>) lines::iterator) {
                        if (line == null || !line.startsWith("data:")) {
                            continue;
                        }
                        String dataJson = line.substring(5).trim();
                        if (dataJson.isEmpty()) {
                            continue;
                        }
                        if ("[DONE]".equals(dataJson)) {
                            break;
                        }
                        try {
                            JsonNode chunk = objectMapper.readTree(dataJson);
                            JsonNode delta = chunk.path("choices").path(0).path("delta");
                            String cPiece = delta.path("content").asString("");
                            if (!cPiece.isEmpty()) {
                                assistantContentBuilder.append(cPiece);
                            }
                            String rPiece = delta.path("reasoning_content").asString("");
                            if (!rPiece.isEmpty()) {
                                thinkingTokensHolder[0]++;
                            }
                            JsonNode timings = chunk.path("timings");
                            if (!timings.isMissingNode()) {
                                promptTokensHolder[0] = timings.path("prompt_n").asInt(promptTokensHolder[0]);
                                predictedTokensHolder[0] = timings.path("predicted_n").asInt(predictedTokensHolder[0]);
                                promptMsHolder[0] = timings.path("prompt_ms").asDouble(promptMsHolder[0]);
                                predictedMsHolder[0] = timings.path("predicted_ms").asDouble(predictedMsHolder[0]);
                                promptPerSecHolder[0] = timings.path("prompt_per_second").asDouble(promptPerSecHolder[0]);
                                predictedPerSecHolder[0] = timings.path("predicted_per_second").asDouble(predictedPerSecHolder[0]);
                            }
                        } catch (Exception ignored) {
                        }
                        emitter.send(SseEmitter.event().data(dataJson, MediaType.APPLICATION_JSON));
                    }
                }

                recordConversationTurnMemory(request.conversationId(), effectiveMessages, assistantContentBuilder.toString());

                long totalLatencyMs = Math.max(12L, (System.nanoTime() - startNs) / 1_000_000L);
                recordRigLatencyEwma(targetRig.getId(), totalLatencyMs);
                recordRigSpeedMetrics(targetRig.getId(), promptPerSecHolder[0], predictedPerSecHolder[0]);

                int pTok = Math.max(1, promptTokensHolder[0]);
                int cTok = Math.max(1, predictedTokensHolder[0]);
                int totalTokens = pTok + cTok;

                CutmeshSwarmTelemetry finalTelemetry = new CutmeshSwarmTelemetry(
                    targetRig.getId(),
                    targetRig.getRigName(),
                    stickyHit,
                    securityProfileName,
                    targetRig.getTensorSplitCsv(),
                    targetRig.getAutoMaximizedContextSize(),
                    targetRig.isAntiCollusionVerified(),
                    canaryDriftL2,
                    canaryStatus,
                    totalLatencyMs,
                    gpuSummaries,
                    promptMsHolder[0],
                    predictedMsHolder[0],
                    promptPerSecHolder[0],
                    predictedPerSecHolder[0],
                    thinkingTokensHolder[0]
                );

                Map<String, Object> finalChunk = new LinkedHashMap<>();
                finalChunk.put("object", "chat.completion.chunk");
                finalChunk.put("choices", List.of());
                finalChunk.put("usage", new ChatCompletionUsage(pTok, cTok, totalTokens));
                finalChunk.put("cutmesh_telemetry", finalTelemetry);
                emitter.send(SseEmitter.event().data(objectMapper.writeValueAsString(finalChunk), MediaType.APPLICATION_JSON));
                emitter.send(SseEmitter.event().data("[DONE]"));
                emitter.complete();

                recordCompletionMetricsAsync(targetRig.getId(), apiKeyOpt.map(OpenAiApiKey::getId).orElse(null), totalTokens);
            } catch (Exception e) {
                log.warn("SSE stream error on rig {}: {}", targetRig.getRigName(), e.getMessage());
                try {
                    Map<String, Object> errChunk = Map.of(
                        "error", Map.of("message", "Errore streaming inferenza sul Rig " + targetRig.getRigName() + ": " + e.getMessage())
                    );
                    emitter.send(SseEmitter.event().data(objectMapper.writeValueAsString(errChunk), MediaType.APPLICATION_JSON));
                    emitter.complete();
                } catch (Exception ignored) {
                    emitter.completeWithError(e);
                }
            } finally {
                decrementInFlight(targetRig.getId());
            }
        });

        return emitter;
    }

    private void recordCompletionMetricsAsync(String rigId, String apiKeyId, int totalTokens) {
        try {
            rigRepository.findById(rigId).ifPresent(r -> {
                r.setServedRequestsCount(r.getServedRequestsCount() + 1L);
                rigRepository.save(r);
            });
            if (apiKeyId != null) {
                apiKeyRepository.findById(apiKeyId).ifPresent(k -> {
                    k.setTotalRequests(k.getTotalRequests() + 1L);
                    k.setTotalTokensGenerated(k.getTotalTokensGenerated() + totalTokens);
                    k.setLastUsedAt(Instant.now());
                    apiKeyRepository.save(k);
                });
            }
        } catch (Exception e) {
            log.debug("Could not persist async completion metrics: {}", e.getMessage());
        }
    }

    public ChatCompletionResponse handleChatCompletion(String authorizationHeader, ChatCompletionRequest request) {
        long startNs = System.nanoTime();
        Optional<OpenAiApiKey> apiKeyOpt = authenticateBearerOrJwt(authorizationHeader);
        SelectedRigContext ctx = resolveTargetRig(request);
        ManagedRig targetRig = ctx.rig();
        boolean stickyHit = ctx.stickyHit();
        CatalogModel modelMeta = ctx.modelMeta();

        // Execute with In-Flight tracking & Automatic Multi-Rig Failover
        LlamaInferenceResult realResult = null;
        for (ManagedRig candidateRig : ctx.failoverPool()) {
            incrementInFlight(candidateRig.getId());
            long attemptStartNs = System.nanoTime();
            try {
                LlamaInferenceResult attempt = executeRealLlamaInferenceOnRig(request, modelMeta, candidateRig);
                if (attempt != null) {
                    double elapsedMs = (System.nanoTime() - attemptStartNs) / 1_000_000.0;
                    recordRigLatencyEwma(candidateRig.getId(), elapsedMs);
                    targetRig = candidateRig;
                    realResult = attempt;
                    break;
                }
            } finally {
                decrementInFlight(candidateRig.getId());
            }
        }

        if (realResult == null) {
            throw new ResponseStatusException(
                HttpStatus.GATEWAY_TIMEOUT,
                "Il motore di inferenza llama-server sul Rig '" + targetRig.getRigName() + "' non ha risposto entro il timeout o non è raggiungibile."
            );
        }

        recordRigSpeedMetrics(targetRig.getId(), realResult.promptPerSecond(), realResult.predictedPerSecond());

        List<String> gpuSummaries = targetRig.getAllocations().stream()
            .map(RigMemberAllocation::getGpuName)
            .toList();

        double canaryDriftL2 = 0.0;
        String canaryStatus = "NOT_AUDITED_YET";
        Double measuredF32Err = null;
        Double measuredQ8Q5Err = null;
        Double measuredCanaryErr = null;
        Double measuredRank1Corr = null;
        var latestCovariant = auditService.listRecentAuditEvents("COVARIANT_MATHEMATICAL_AUDIT").stream().findFirst();
        if (latestCovariant.isPresent()) {
            var ev = latestCovariant.get();
            measuredF32Err = ev.getF32RelativeErr();
            measuredQ8Q5Err = ev.getQ8ToQ5RelativeErr();
            measuredCanaryErr = ev.getCanaryRelativeErr();
            measuredRank1Corr = ev.getRank1CorrMax();
            if (measuredCanaryErr != null) {
                canaryDriftL2 = measuredCanaryErr;
                canaryStatus = Boolean.TRUE.equals(ev.isPassed())
                    ? "VERIFIED_COVARIANT_INVARIANT (drift_L2=" + canaryDriftL2 + ")"
                    : "COVARIANT_AUDIT_FAILED (drift_L2=" + canaryDriftL2 + ")";
            }
        }

        String assistantReply = realResult.content();
        String reasoningContent = realResult.reasoningContent();
        int promptTokens = realResult.promptTokens();
        int completionTokens = realResult.completionTokens();
        int totalTokens = promptTokens + completionTokens;

        recordCompletionMetricsAsync(targetRig.getId(), apiKeyOpt.map(OpenAiApiKey::getId).orElse(null), totalTokens);

        long latencyMs = Math.max(12L, (System.nanoTime() - startNs) / 1_000_000L);

        auditService.recordEvent(
            "INFERENCE_COMPLETION_TELEMETRY",
            "INFO",
            String.format(Locale.ROOT, "Served Chat Completion on Rig '%s' (%s, %d ms)",
                targetRig.getRigName(), canaryStatus, latencyMs),
            String.format(Locale.ROOT, "Completamento Chat Servito sul Rig '%s' (%s, %d ms)",
                targetRig.getRigName(), canaryStatus, latencyMs),
            String.format(Locale.ROOT, "Served /v1/chat/completions on model '%s' via Rig '%s' (StickySession=%s, GPUs=%s, Tokens=%d, %.2f tok/s).",
                targetRig.getModelId(), targetRig.getRigName(), stickyHit, String.join(" + ", gpuSummaries), totalTokens, realResult.predictedPerSecond()),
            String.format(Locale.ROOT, "Servita /v1/chat/completions sul modello '%s' tramite Rig '%s' (SessioneSticky=%s, GPU=%s, Token=%d, %.2f tok/s).",
                targetRig.getModelId(), targetRig.getRigName(), stickyHit, String.join(" + ", gpuSummaries), totalTokens, realResult.predictedPerSecond()),
            targetRig.getId(),
            true,
            measuredF32Err,
            measuredQ8Q5Err,
            measuredCanaryErr,
            measuredRank1Corr,
            targetRig.isAntiCollusionVerified(),
            null
        );

        String securityProfileName = modelMeta != null
            ? modelMeta.getSecurityProfile().name()
            : targetRig.getSecurityProfile().name();

        CutmeshSwarmTelemetry telemetry = new CutmeshSwarmTelemetry(
            targetRig.getId(),
            targetRig.getRigName(),
            stickyHit,
            securityProfileName,
            targetRig.getTensorSplitCsv(),
            targetRig.getAutoMaximizedContextSize(),
            targetRig.isAntiCollusionVerified(),
            canaryDriftL2,
            canaryStatus,
            latencyMs,
            gpuSummaries,
            realResult.promptMs(),
            realResult.predictedMs(),
            realResult.promptPerSecond(),
            realResult.predictedPerSecond(),
            reasoningContent != null && !reasoningContent.isBlank() ? Math.max(1, reasoningContent.length() / 4) : 0
        );

        return new ChatCompletionResponse(
            "chatcmpl-cutmesh-" + UUID.randomUUID().toString().substring(0, 12),
            "chat.completion",
            Instant.now().getEpochSecond(),
            targetRig.getModelId(),
            List.of(new ChatCompletionChoice(0, new ChatMessageDto("assistant", assistantReply, reasoningContent), "stop")),
            new ChatCompletionUsage(promptTokens, completionTokens, totalTokens),
            telemetry
        );
    }

    private record LlamaInferenceResult(
        String content,
        String reasoningContent,
        int promptTokens,
        int completionTokens,
        double promptMs,
        double predictedMs,
        double promptPerSecond,
        double predictedPerSecond
    ) {
    }

    private ManagedRig selectLeastLoadedRig(List<ManagedRig> candidates) {
        return stickyRigRouter.selectLeastLoadedRig(candidates, this::isRigEndpointHealthy);
    }

    private List<ManagedRig> sortRigsByLoadScore(List<ManagedRig> candidates, String preferredRigId) {
        return stickyRigRouter.sortRigsByLoadScore(candidates, preferredRigId, this::isRigEndpointHealthy);
    }

    public double computeRigLoadScore(ManagedRig rig) {
        return stickyRigRouter.computeRigLoadScore(rig, isRigEndpointHealthy(rig));
    }

    public int getInFlightCount(String rigId) {
        return stickyRigRouter.getInFlightCount(rigId);
    }

    public double getRigAvgLatencyMs(String rigId) {
        return stickyRigRouter.getRigAvgLatencyMs(rigId);
    }

    public double getRigLastPrefillTokPerSec(String rigId) {
        return stickyRigRouter.getRigLastPrefillTokPerSec(rigId);
    }

    public double getRigLastDecodeTokPerSec(String rigId) {
        return stickyRigRouter.getRigLastDecodeTokPerSec(rigId);
    }

    private void recordRigSpeedMetrics(String rigId, double prefillTokPerSec, double decodeTokPerSec) {
        stickyRigRouter.recordRigSpeedMetrics(rigId, prefillTokPerSec, decodeTokPerSec);
    }

    private void incrementInFlight(String rigId) {
        stickyRigRouter.incrementInFlight(rigId);
    }

    private void decrementInFlight(String rigId) {
        stickyRigRouter.decrementInFlight(rigId);
    }

    private void recordRigLatencyEwma(String rigId, double sampleMs) {
        stickyRigRouter.recordRigLatencyEwma(rigId, sampleMs);
    }

    public Map<String, Object> getRouterLoadBalancerStatus() {
        List<ManagedRig> servingRigs = rigRepository.findByLifecycleStateOrderByCreatedAtDesc(RigLifecycleState.SERVING);
        List<Map<String, Object>> upstreams = servingRigs.stream().map(rig -> {
            boolean healthy = isRigEndpointHealthy(rig);
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("rigId", rig.getId());
            m.put("rigName", rig.getRigName());
            m.put("modelId", rig.getModelId());
            m.put("upstreamEndpoint", rig.getInferenceEndpointUrl());
            m.put("inferenceEngineReady", healthy);
            m.put("activeInFlightRequests", getInFlightCount(rig.getId()));
            m.put("ewmaLatencyMs", Math.round(getRigAvgLatencyMs(rig.getId()) * 10.0) / 10.0);
            m.put("loadScore", Math.round(computeRigLoadScore(rig) * 100.0) / 100.0);
            m.put("servedRequestsCount", rig.getServedRequestsCount());
            return m;
        }).toList();

        return Map.of(
            "publicConsumerEndpoint", "/v1/chat/completions",
            "routingPolicy", "STICKY_KV_CACHE_AFFINITY + LEAST_IN_FLIGHT_WEIGHTED_EWMA + AUTO_FAILOVER",
            "activeStickyConversations", stickyRigRouter.getActiveStickyConversationCount(),
            "activeUpstreamsCount", upstreams.size(),
            "upstreams", upstreams
        );
    }

    /**
     * Routes the OpenAI chat completion request to the specific Rig's upstream endpoint
     * (or fallback default port 8089) without the end user ever needing to know internal ports.
     */
    private LlamaInferenceResult executeRealLlamaInferenceOnRig(
        ChatCompletionRequest request,
        CatalogModel modelMeta,
        ManagedRig targetRig
    ) {
        try {
            String baseEndpoint = resolveBaseEndpoint(targetRig, modelMeta);
            boolean enableThinking = resolveEnableThinking(request);
            List<ChatMessageDto> effectiveMessages = resolveEffectiveMessagesWithMemory(request);
            int effectiveMaxTokens = resolveEffectiveMaxTokens(request, enableThinking);

            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("model", request.model());
            payload.put("messages", effectiveMessages);
            payload.put("temperature", request.temperature() != null ? request.temperature() : 0.6);
            payload.put("max_tokens", effectiveMaxTokens);
            payload.put("n_predict", effectiveMaxTokens);
            payload.put("stream", false);
            payload.put("cache_prompt", true);
            payload.put("timings_per_token", true);
            payload.put("chat_template_kwargs", buildChatTemplateKwargs(request, enableThinking));
            if (!enableThinking) {
                payload.put("reasoning_budget", 0);
                payload.put("reasoning_format", "none");
            }

            String bodyJson = objectMapper.writeValueAsString(payload);
            HttpRequest httpReq = HttpRequest.newBuilder()
                .uri(URI.create(baseEndpoint + "/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(900))
                .POST(HttpRequest.BodyPublishers.ofString(bodyJson))
                .build();

            HttpResponse<String> resp = httpClient.send(httpReq, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() == 200) {
                JsonNode root = objectMapper.readTree(resp.body());
                JsonNode msgNode = root.path("choices").path(0).path("message");
                String content = msgNode.path("content").asString("");
                String reasoning = msgNode.path("reasoning_content").asString("");
                int promptTok = root.path("usage").path("prompt_tokens").asInt(16);
                int compTok = root.path("usage").path("completion_tokens").asInt(32);

                JsonNode timings = root.path("timings");
                double promptMs = timings.path("prompt_ms").asDouble(0.0);
                double predictedMs = timings.path("predicted_ms").asDouble(0.0);
                double promptPerSec = timings.path("prompt_per_second").asDouble(0.0);
                double predictedPerSec = timings.path("predicted_per_second").asDouble(0.0);

                if (!content.isBlank() || !reasoning.isBlank()) {
                    recordConversationTurnMemory(request.conversationId(), effectiveMessages, !content.isBlank() ? content : reasoning);
                    return new LlamaInferenceResult(
                        content,
                        reasoning.isBlank() ? null : reasoning,
                        promptTok,
                        compTok,
                        promptMs,
                        predictedMs,
                        promptPerSec,
                        predictedPerSec
                    );
                }
            }
        } catch (Exception e) {
            log.warn("Upstream rig {} inference error: {}", targetRig.getRigName(), e.getMessage());
        }
        return null;
    }

    @org.springframework.scheduling.annotation.Scheduled(fixedDelayString = "3000", initialDelayString = "2000")
    public void refreshRigHealthCache() {
        List<ManagedRig> servingRigs = rigRepository.findByLifecycleStateOrderByCreatedAtDesc(RigLifecycleState.SERVING);
        for (ManagedRig rig : servingRigs) {
            int code = probeRigEndpointHttpCode(rig);
            cachedRigHttpCodeById.put(rig.getId(), code);
            if (code == 200) {
                orchestratorService.transitionRigStateMachine(
                    rig.getId(),
                    RigRuntimePhaseState.STEP_4_OPERATIONAL_READY,
                    "llama-server /health == 200 OK (" + rig.getInferenceEndpointUrl() + ")"
                );
            } else if (orchestratorService.getRigFsmState(rig.getId()) == RigRuntimePhaseState.STEP_4_OPERATIONAL_READY
                && !orchestratorService.isRigProcessAlive(rig.getId())) {
                orchestratorService.transitionRigStateMachine(
                    rig.getId(),
                    RigRuntimePhaseState.ERROR_DEGRADED,
                    "Motore inferenza non raggiungibile su " + rig.getInferenceEndpointUrl()
                );
            }
        }
    }

    public int getCachedRigEndpointHttpCode(String rigId) {
        if (rigId == null) {
            return 0;
        }
        return cachedRigHttpCodeById.getOrDefault(rigId, 0);
    }

    public boolean isRigEndpointHealthy(ManagedRig rig) {
        if (rig == null || rig.getId() == null) {
            return false;
        }
        return cachedRigHttpCodeById.getOrDefault(rig.getId(), 0) == 200;
    }

    public int probeRigEndpointHttpCode(ManagedRig rig) {
        if (rig == null) {
            return 0;
        }
        String baseUrl = rig.getInferenceEndpointUrl();
        if (baseUrl == null || baseUrl.isBlank()) {
            baseUrl = "http://127.0.0.1:" + LLAMA_SERVER_PORT;
        }
        try {
            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/health"))
                .timeout(Duration.ofMillis(700))
                .GET()
                .build();
            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode();
        } catch (Exception e) {
            return 0;
        }
    }

    private boolean isEndpointUrlHealthy(String baseUrl) {
        if (baseUrl == null || baseUrl.isBlank()) {
            return false;
        }
        try {
            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/health"))
                .timeout(Duration.ofMillis(700))
                .GET()
                .build();
            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    public boolean isLlamaServerHealthy() {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create("http://127.0.0.1:" + LLAMA_SERVER_PORT + "/health"))
                .timeout(Duration.ofMillis(800))
                .GET()
                .build();
            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    private void startLlamaServerIfAvailable(CatalogModel modelMeta) {
        try {
            File llamaExe = resolveLlamaServerBinary();
            if (llamaExe == null || !llamaExe.exists()) {
                return;
            }
            if (modelMeta == null || modelMeta.getBaseGgufPath() == null) {
                return;
            }
            File ggufFile = auditService.resolveModelFile(modelMeta.getBaseGgufPath());
            if (ggufFile == null || !ggufFile.exists()) {
                return;
            }

            ProcessBuilder pb = new ProcessBuilder(
                llamaExe.getAbsolutePath(),
                "-m", ggufFile.getAbsolutePath(),
                "--host", "127.0.0.1",
                "--port", String.valueOf(LLAMA_SERVER_PORT),
                "-ngl", "99",
                "-c", "4096"
            );
            pb.redirectErrorStream(true);
            llamaServerProcess = pb.start();

            for (int i = 0; i < 20; i++) {
                Thread.sleep(400);
                if (isLlamaServerHealthy()) {
                    log.info("Started real llama-server on port {} with model {}", LLAMA_SERVER_PORT, ggufFile.getName());
                    return;
                }
            }
        } catch (Exception e) {
            log.warn("Could not auto-start llama-server: {}", e.getMessage());
        }
    }

    private File resolveLlamaServerBinary() {
        File directWin = new File("bin/llamacpp/llama-server.exe");
        if (directWin.exists()) {
            return directWin;
        }
        File parentWin = new File("../bin/llamacpp/llama-server.exe");
        if (parentWin.exists()) {
            return parentWin;
        }
        File unixBin = new File("/usr/local/bin/llama-server");
        if (unixBin.exists()) {
            return unixBin;
        }
        File optBin = new File("/opt/cutmesh/llamacpp/llama-server");
        if (optBin.exists()) {
            return optBin;
        }
        return null;
    }

    private static String buildFallbackReply(
        String userPrompt,
        ManagedRig rig,
        List<String> gpus,
        boolean stickyHit,
        double canaryDrift
    ) {
        return String.format(
            Locale.ROOT,
            "⚠️ [MOTORE DI INFERENZA NON OPERATIVO — PESI NON CARICATI IN VRAM]\n\n"
                + "Il Rig `%s` (%s) ha calcolato la partizione dei layer (`[%s]`, `%d MB` VRAM allocati, `-c %d`), "
                + "ma il motore CUDA `llama-server` (`127.0.0.1:8089`) NON ha ancora i pesi `.gguf` caricati nella VRAM delle GPU remote.\n"
                + "• Stato Operativo Modello (%s): `NON PRONTO (SOLO TOPOLOGIA PIANIFICATA)`\n"
                + "• Prompt ricevuto: \"%s\"\n"
                + "• Azione richiesta: Avviare `llama-server` sul nodo Master del Rig e verificare `/health == 200 OK`.",
            rig.getRigName(),
            String.join(" + ", gpus),
            rig.getTensorSplitCsv(),
            rig.getTotalAllocatedVramMb(),
            rig.getAutoMaximizedContextSize(),
            rig.getModelDisplayName(),
            userPrompt
        );
    }
}
