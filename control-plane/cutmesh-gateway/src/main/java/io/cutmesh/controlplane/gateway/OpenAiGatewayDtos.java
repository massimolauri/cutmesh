package io.cutmesh.controlplane.gateway;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import java.time.Instant;
import java.util.List;
import java.util.Map;

public final class OpenAiGatewayDtos {

    private OpenAiGatewayDtos() {
    }

    public record CreateOpenAiKeyRequest(
        @NotBlank String name,
        @Min(1) @Max(10000) int rateLimitRpm
    ) {
    }

    public record CreatedOpenAiKeyResponse(
        String id,
        String name,
        String rawApiKeySecret,
        String keyPrefix,
        int rateLimitRpm,
        Instant createdAt
    ) {
    }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record ChatMessageDto(
        @NotBlank String role,
        String content,
        @JsonProperty("reasoning_content") String reasoningContent
    ) {
        public ChatMessageDto(String role, String content) {
            this(role, content, null);
        }
    }

    public record ChatCompletionRequest(
        @NotBlank String model,
        @NotEmpty List<ChatMessageDto> messages,
        Double temperature,
        @JsonProperty("max_tokens") Integer maxTokens,
        @JsonProperty("conversation_id") String conversationId,
        Boolean stream,
        @JsonProperty("enable_thinking") Boolean enableThinking,
        @JsonProperty("reasoning_effort") String reasoningEffort,
        @JsonProperty("chat_template_kwargs") Map<String, Object> chatTemplateKwargs
    ) {
    }

    public record ChatCompletionChoice(
        int index,
        ChatMessageDto message,
        @JsonProperty("finish_reason") String finishReason
    ) {
    }

    public record ChatCompletionUsage(
        @JsonProperty("prompt_tokens") int promptTokens,
        @JsonProperty("completion_tokens") int completionTokens,
        @JsonProperty("total_tokens") int totalTokens
    ) {
    }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record CutmeshSwarmTelemetry(
        @JsonProperty("rig_id") String rigId,
        @JsonProperty("rig_name") String rigName,
        @JsonProperty("sticky_session_hit") boolean stickySessionHit,
        @JsonProperty("security_profile") String securityProfile,
        @JsonProperty("tensor_split_csv") String tensorSplitCsv,
        @JsonProperty("context_window") int contextWindow,
        @JsonProperty("anti_collusion_verified") boolean antiCollusionVerified,
        @JsonProperty("canary_drift_l2") double canaryDriftL2,
        @JsonProperty("canary_status") String canaryStatus,
        @JsonProperty("latency_ms") long latencyMs,
        @JsonProperty("participating_gpus") List<String> participatingGpus,
        @JsonProperty("prompt_ms") Double promptMs,
        @JsonProperty("predicted_ms") Double predictedMs,
        @JsonProperty("prompt_per_second") Double promptPerSecond,
        @JsonProperty("predicted_per_second") Double predictedPerSecond,
        @JsonProperty("thinking_tokens") Integer thinkingTokens
    ) {
    }

    public record ChatCompletionResponse(
        String id,
        String object,
        long created,
        String model,
        List<ChatCompletionChoice> choices,
        ChatCompletionUsage usage,
        @JsonProperty("cutmesh_telemetry") CutmeshSwarmTelemetry cutmeshTelemetry
    ) {
    }
}

