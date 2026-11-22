package io.cutmesh.controlplane;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

@SpringBootTest
@AutoConfigureMockMvc
class ControlPlaneIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void fullEndToEndControlPlaneSecurityRbacWorkerAndAudit() throws Exception {
        // 1. Public health status (unauthenticated)
        mockMvc.perform(get("/api/v1/public/status"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("UP"))
            .andExpect(jsonPath("$.googleOAuthEnabled").value(false));

        // 2. Unauthenticated access to protected endpoints must return 401
        mockMvc.perform(get("/api/v1/users"))
            .andExpect(status().isUnauthorized());

        // 3. Google OAuth must fail closed (503) when CUTMESH_GOOGLE_CLIENT_ID is unset
        mockMvc.perform(post("/api/v1/auth/google")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"email\":\"user@cutmesh.test\",\"displayName\":\"Test User\"}"))
            .andExpect(status().isServiceUnavailable());

        // 4. Bootstrapped SUPER_ADMIN PBKDF2 login succeeds and issues HS256 JWT
        MvcResult adminLoginRes = mockMvc.perform(post("/api/v1/auth/login")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"email\":\"admin@cutmesh.test\",\"password\":\"AdminTestPass!2026\"}"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.role").value("SUPER_ADMIN"))
            .andExpect(jsonPath("$.jwtToken").isNotEmpty())
            .andReturn();
        String adminJwt = objectMapper.readTree(adminLoginRes.getResponse().getContentAsString())
            .get("jwtToken").asString();

        // Wrong password must fail with 401
        mockMvc.perform(post("/api/v1/auth/login")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"email\":\"admin@cutmesh.test\",\"password\":\"wrong-pass\"}"))
            .andExpect(status().isUnauthorized());

        // 5. Public registration: SUPER_ADMIN blocked (403), API_USER allowed (200)
        mockMvc.perform(post("/api/v1/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"email\":\"escalate@cutmesh.test\",\"password\":\"Pass!1234\",\"displayName\":\"Escalate\",\"role\":\"SUPER_ADMIN\"}"))
            .andExpect(status().isForbidden());

        MvcResult apiUserRegRes = mockMvc.perform(post("/api/v1/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"email\":\"dev@cutmesh.test\",\"password\":\"DevPass!2026\",\"displayName\":\"API Dev\",\"role\":\"API_USER\",\"preferredLocale\":\"en\"}"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.role").value("API_USER"))
            .andReturn();
        String apiUserJwt = objectMapper.readTree(apiUserRegRes.getResponse().getContentAsString())
            .get("jwtToken").asString();

        // RBAC enforcement: API_USER cannot list users or issue GPU enrollment tokens (403)
        mockMvc.perform(get("/api/v1/users")
                .header("Authorization", "Bearer " + apiUserJwt))
            .andExpect(status().isForbidden());
        mockMvc.perform(post("/api/v1/enrollment-tokens")
                .header("Authorization", "Bearer " + apiUserJwt)
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"label\":\"unauth-gpu\",\"defaultPoolMode\":\"SHARED_POOL\"}"))
            .andExpect(status().isForbidden());

        // Locale update with JWT succeeds
        mockMvc.perform(put("/api/v1/auth/locale")
                .header("Authorization", "Bearer " + adminJwt)
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"preferredLocale\":\"en\"}"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.preferredLocale").value("en"));

        // 6. Authenticated SUPER_ADMIN issues GPU enrollment token (`syn_node_live_...`)
        MvcResult tokenRes = mockMvc.perform(post("/api/v1/enrollment-tokens")
                .header("Authorization", "Bearer " + adminJwt)
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"label\":\"RTX-4060Ti-Test-Node\",\"defaultPoolMode\":\"SHARED_POOL\"}"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.rawTokenSecret").isNotEmpty())
            .andReturn();
        String rawNodeToken = objectMapper.readTree(tokenRes.getResponse().getContentAsString())
            .get("rawTokenSecret").asString();
        assertThat(rawNodeToken).startsWith("syn_node_live_");

        // 7. Worker heartbeat & disconnect require valid `syn_node_live_...` token (401 without or wrong token)
        String heartbeatNode1 = """
            {
              "iroh_node_id": "iroh_node_pk_alpha_gpu_0001",
              "iroh_ticket": "iroh://iroh_node_pk_alpha_gpu_0001@127.0.0.1:50052",
              "gpu_name": "NVIDIA RTX 5090",
              "free_vram_mb": 31800,
              "total_vram_mb": 32768,
              "max_allocatable_vram_mb": 30000,
              "pool_mode": "SHARED_POOL",
              "rpc_healthy": true,
              "watchdog_restarts": 0,
              "version": "0.2.0"
            }
            """;

        mockMvc.perform(post("/api/v1/internal/workers/heartbeat")
                .contentType(MediaType.APPLICATION_JSON)
                .content(heartbeatNode1))
            .andExpect(status().isUnauthorized());

        mockMvc.perform(post("/api/v1/internal/workers/heartbeat")
                .header("Authorization", "Bearer " + rawNodeToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(heartbeatNode1))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.accepted").value(true))
            .andExpect(jsonPath("$.assignment_state").value("LOCKED_IN_RIG"))
            .andExpect(jsonPath("$.assigned_model_id").value("qwen3.8-27b"));

        mockMvc.perform(get("/api/v1/catalog/models/qwen3.8-27b/policy")
                .header("Authorization", "Bearer " + apiUserJwt))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.model_id").value("qwen3.8-27b"))
            .andExpect(jsonPath("$.engine.default_ctx_size").value(262144));

        // 8. API_USER creates OpenAI key, logs in via /api/v1/auth/api-key, and queries /v1/models
        MvcResult keyRes = mockMvc.perform(post("/api/v1/api-keys")
                .header("Authorization", "Bearer " + apiUserJwt)
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"name\":\"Test Consumer Key\",\"rateLimitRpm\":120}"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.rawApiKeySecret").isNotEmpty())
            .andReturn();
        String rawOpenAiKey = objectMapper.readTree(keyRes.getResponse().getContentAsString())
            .get("rawApiKeySecret").asString();
        assertThat(rawOpenAiKey).startsWith("sk-cutmesh-");

        mockMvc.perform(post("/api/v1/auth/api-key")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"apiKey\":\"" + rawOpenAiKey + "\"}"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.role").value("API_USER"))
            .andExpect(jsonPath("$.jwtToken").isNotEmpty());

        mockMvc.perform(get("/v1/models")
                .header("Authorization", "Bearer " + rawOpenAiKey))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data").isArray());

        // 9. Dissolve rig & test authenticated worker disconnect (reject unauthenticated, succeed with token)
        MvcResult rigsRes = mockMvc.perform(get("/api/v1/rigs")
                .header("Authorization", "Bearer " + adminJwt))
            .andExpect(status().isOk())
            .andReturn();
        JsonNode rigsArray = objectMapper.readTree(rigsRes.getResponse().getContentAsString());
        String activeRigId = rigsArray.get(0).get("id").asString();

        mockMvc.perform(delete("/api/v1/rigs/" + activeRigId)
                .header("Authorization", "Bearer " + adminJwt))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.lifecycleState").value("DISSOLVED"));

        mockMvc.perform(post("/api/v1/internal/workers/disconnect")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"iroh_node_id\":\"iroh_node_pk_alpha_gpu_0001\"}"))
            .andExpect(status().isUnauthorized());

        mockMvc.perform(post("/api/v1/internal/workers/disconnect")
                .header("Authorization", "Bearer " + rawNodeToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"iroh_node_id\":\"iroh_node_pk_alpha_gpu_0001\"}"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.disconnected").value(true));
    }
}
