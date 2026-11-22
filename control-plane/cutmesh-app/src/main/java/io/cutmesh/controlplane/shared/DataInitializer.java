package io.cutmesh.controlplane.shared;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.json.JsonMapper;
import io.cutmesh.controlplane.iam.AppUser;
import io.cutmesh.controlplane.orchestration.CatalogModel;
import io.cutmesh.controlplane.orchestration.OrchestrationMode;
import io.cutmesh.controlplane.orchestration.SecurityProfile;
import io.cutmesh.controlplane.iam.AppUserRepository;
import io.cutmesh.controlplane.orchestration.CatalogModelRepository;

import io.cutmesh.controlplane.audit.SecurityAuditService;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ClassPathResource;

/**
 * Seeds clean production accounts for the 3 RBAC roles (SUPER_ADMIN, HARDWARE_PROVIDER, API_USER)
 * and loads verified catalog model policies from JSON configuration files (`models/<id>/policy.json`
 * on disk with classpath fallback `policies/<id>.json`).
 */
@Configuration
public class DataInitializer {

    private static final Logger log = LoggerFactory.getLogger(DataInitializer.class);
    private static final ObjectMapper JSON_MAPPER = JsonMapper.builder().build();
    private static final List<String> VERIFIED_MODEL_IDS = List.of("qwen3.8-27b");

    static long computeMinimumRunnableVramMb(String policyJson, long fallbackMb) {
        if (policyJson == null || policyJson.isBlank()) {
            return fallbackMb;
        }
        try {
            JsonNode root = JSON_MAPPER.readTree(policyJson);
            JsonNode sh = root.path("sharding");
            if (sh.isMissingNode()) {
                return fallbackMb;
            }
            double totalMib = sh.path("non_block_master_weights_mib").asDouble(0.0);
            for (JsonNode w : sh.path("layer_weights_mib")) {
                totalMib += w.asDouble(0.0);
            }
            JsonNode spec = root.path("speculative");
            if (!spec.isMissingNode() && spec.path("enabled").asBoolean(false)) {
                totalMib += spec.path("draft_weights_mib").asDouble(0.0);
            }
            long kvPer1k = Math.max(1L, sh.path("kv_vram_per_1k_tokens_mb").asLong(0L));
            long maxCtx = Math.max(4096L, root.path("engine").path("default_ctx_size").asLong(32768L));
            if (totalMib <= 0.0 || kvPer1k <= 0L) {
                return fallbackMb;
            }
            double stage0ExtraMib = sh.path("stage0_extra_vram_reserve_mb").asDouble(1024.0);
            // Admit any rig that fits weights + Stage-0 headroom + >=32k KV cache;
            // RigPlacementPlanner automatically scales `-c` up to max_context_length when surplus VRAM exists.
            long minBaselineCtx = Math.min(maxCtx, 32768L);
            double minKvMib = (minBaselineCtx / 1024.0) * kvPer1k;
            return Math.max(fallbackMb, Math.round(totalMib + stage0ExtraMib + minKvMib));
        } catch (Exception e) {
            return fallbackMb;
        }
    }

    static String loadPolicyJson(String modelId) {
        String[] externalRoots = {
            "models",
            "../models",
            "/app/models",
            "/opt/cutmesh/models",
            "/data/cutmesh/models",
            "/data/cutmesh/models"
        };
        for (String root : externalRoots) {
            File f = new File(new File(root, modelId), "policy.json");
            if (f.isFile()) {
                try {
                    log.debug("Loaded model policy for '{}' from disk: {}", modelId, f.getAbsolutePath());
                    return Files.readString(f.toPath(), StandardCharsets.UTF_8);
                } catch (IOException e) {
                    log.warn("Failed reading external policy {}: {}", f.getAbsolutePath(), e.getMessage());
                }
            }
        }

        ClassPathResource resource = new ClassPathResource("policies/" + modelId + ".json");
        try {
            return resource.getContentAsString(StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException(
                "Missing model policy JSON for '" + modelId + "' (checked external models/ dirs and classpath:policies/" + modelId + ".json)",
                e
            );
        }
    }

    @Bean
    public CommandLineRunner initializeCutmeshControlPlane(
        AppUserRepository userRepository,
        CatalogModelRepository catalogRepository,
        SecurityAuditService auditService,
        @Value("${cutmesh.bootstrap.demo-accounts:false}") boolean bootstrapDemoAccounts,
        @Value("${cutmesh.bootstrap.admin-email:admin@cutmesh.local}") String bootstrapAdminEmail,
        @Value("${cutmesh.bootstrap.admin-password:}") String bootstrapAdminPassword
    ) {
        return args -> {
            if (userRepository.count() == 0 && bootstrapAdminPassword != null && !bootstrapAdminPassword.isBlank()) {
                String cleanEmail = bootstrapAdminEmail.trim().toLowerCase(java.util.Locale.ROOT);
                AppUser admin = new AppUser(
                    cleanEmail,
                    "Cluster Administrator",
                    "LOCAL",
                    "local|" + cleanEmail,
                    "https://api.dicebear.com/7.x/shapes/svg?seed=" + cleanEmail,
                    "en",
                    "SUPER_ADMIN",
                    CryptoHashUtils.hashPassword(bootstrapAdminPassword)
                );
                userRepository.save(admin);
                log.info("Bootstrapped SUPER_ADMIN account: {}", admin.getEmail());
            }

            if (bootstrapDemoAccounts) {
                if (userRepository.findByEmailIgnoreCase("provider@cutmesh.local").isEmpty()) {
                    AppUser provider = new AppUser(
                        "provider@cutmesh.local",
                        "GPU Node Provider",
                        "LOCAL",
                        "local|provider@cutmesh.local",
                        "https://api.dicebear.com/7.x/shapes/svg?seed=GpuProvider",
                        "it",
                        "HARDWARE_PROVIDER",
                        CryptoHashUtils.hashPassword("provider-dev-local")
                    );
                    userRepository.save(provider);
                }
                if (userRepository.findByEmailIgnoreCase("developer@cutmesh.local").isEmpty()) {
                    AppUser apiUser = new AppUser(
                        "developer@cutmesh.local",
                        "API Developer",
                        "LOCAL",
                        "local|developer@cutmesh.local",
                        "https://api.dicebear.com/7.x/shapes/svg?seed=ApiDeveloper",
                        "it",
                        "API_USER",
                        CryptoHashUtils.hashPassword("developer-dev-local")
                    );
                    userRepository.save(apiUser);
                }
            }

            // Keep ONLY verified production catalog models in the database
            Set<String> allowedIds = Set.copyOf(VERIFIED_MODEL_IDS);
            for (CatalogModel existing : catalogRepository.findAll()) {
                if (!allowedIds.contains(existing.getModelId())) {
                    catalogRepository.deleteById(existing.getModelId());
                    log.info("Removed obsolete catalog model from DB: {}", existing.getModelId());
                }
            }

            int priority = 1;
            for (String modelId : VERIFIED_MODEL_IDS) {
                String policyJson = loadPolicyJson(modelId);
                JsonNode root = JSON_MAPPER.readTree(policyJson);
                JsonNode sharding = root.path("sharding");
                JsonNode engine = root.path("engine");

                String ggufFilename = root.path("gguf_filename").asText(modelId + ".gguf");
                String covariantFilename = ggufFilename.endsWith(".gguf")
                    ? ggufFilename.substring(0, ggufFilename.length() - 5) + "-covariant.gguf"
                    : ggufFilename + "-covariant.gguf";

                int trunkLayers = sharding.path("trunk_layers").asInt(64);
                int mtpCount = sharding.path("mtp_layers").isArray() ? sharding.path("mtp_layers").size() : 0;

                CatalogModel model = catalogRepository.findById(modelId).orElseGet(CatalogModel::new);
                model.setModelId(modelId);
                model.setDisplayName(root.path("display_name").asText(modelId));
                model.setHuggingfaceRepo(root.path("canonical_hf_url").asText(""));
                model.setBaseGgufPath("models/" + ggufFilename);
                model.setCovariantGgufPath("models/" + covariantFilename);
                model.setRequiredVramMb(computeMinimumRunnableVramMb(policyJson, 19000L));
                model.setTotalLayers(trunkLayers + mtpCount);
                model.setKvVramPer1kTokensMb(sharding.path("kv_vram_per_1k_tokens_mb").asLong(36L));
                model.setMaxContextLength(engine.path("max_context_length").asInt(262144));
                model.setPolicyJson(policyJson);
                model.setSecurityProfile(SecurityProfile.FULL_COVARIANT_SHIELD);
                if (model.getOrchestrationMode() == null) {
                    model.setOrchestrationMode(OrchestrationMode.AUTO_PILOT);
                }
                if (model.getPriorityOrder() <= 0 || model.getPriorityOrder() == 100) {
                    model.setPriorityOrder(priority);
                }
                model.setMinActiveRigs(1);
                model.setMaxActiveRigs(4);
                if (!model.isCovariantCompiled()) {
                    model.setLastCompileMs(null);
                    model.setLastCompileThroughputMbPerSec(null);
                    model.setTransformedQ50Rows(null);
                    model.setCalibratedRmsnormTensors(null);
                }
                catalogRepository.save(model);
                priority++;
            }
        };
    }
}
