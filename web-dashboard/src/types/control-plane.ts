import { Locale } from "@/i18n/translations";

export type ActiveSection =
  | "overview"
  | "orchestrator"
  | "provider"
  | "playground"
  | "apikeys"
  | "users"
  | "audit";

export type AppRole = "SUPER_ADMIN" | "HARDWARE_PROVIDER" | "API_USER";

export interface UserProfile {
  id: string;
  email: string;
  displayName: string;
  oauthProvider: string;
  avatarUrl: string;
  preferredLocale: Locale;
  role: AppRole | string;
  createdAt?: string;
  lastLoginAt?: string;
  jwtToken?: string;
}

export interface EnrollmentTokenItem {
  id: string;
  label: string;
  tokenPrefix: string;
  defaultPoolMode: "SHARED_POOL" | "PRIVATE_SOLO";
  revoked: boolean;
  createdAt: string;
  lastUsedAt?: string;
}

export interface EnrollmentTokenIssued {
  id: string;
  label: string;
  rawTokenSecret: string;
  tokenPrefix: string;
  defaultPoolMode: "SHARED_POOL" | "PRIVATE_SOLO";
  dockerRunLinuxCommand: string;
  dockerRunWsl2Command: string;
  nativeRustCliCommand: string;
  createdAt: string;
}

export interface GpuWorkerNode {
  id: string;
  nodeName: string;
  irohNodeId: string;
  irohTicket: string;
  localRpcPort: number;
  device: string;
  gpuName: string;
  freeVramMb: number;
  totalVramMb: number;
  maxAllocatableVramMb: number;
  effectiveTflops: number;
  effectiveBandwidthGbS: number;
  allocatedPowerWatts: number;
  powerBudgetPct: number;
  temperatureC: number;
  gpuUtilizationPct: number;
  poolMode: "SHARED_POOL" | "PRIVATE_SOLO";
  assignmentState: "IDLE_STANDBY_POOL" | "LOCKED_IN_RIG" | "DRAINING" | "OFFLINE_STALE";
  assignedRigId?: string;
  assignedModelId?: string;
  assignedRole?: string;
  rpcHealthy: boolean;
  watchdogRestarts: number;
  version: string;
  lastHeartbeatAt: string;
}

export interface CatalogModel {
  modelId: string;
  displayName: string;
  huggingfaceRepo: string;
  baseGgufPath: string;
  covariantGgufPath: string;
  requiredVramMb: number;
  totalLayers: number;
  kvVramPer1kTokensMb: number;
  maxContextLength?: number;
  securityProfile:
    | "FULL_COVARIANT_SHIELD"
    | "SHARD_ONLY_NO_COVARIANT"
    | "STANDARD_PLAINTEXT";
  orchestrationMode: "AUTO_PILOT" | "MANUAL_ONLY";
  minActiveRigs: number;
  maxActiveRigs: number;
  priorityOrder: number;
  covariantCompiled: boolean;
  lastCompileMs?: number;
  lastCompileThroughputMbPerSec?: number;
  transformedQ50Rows?: number;
  calibratedRmsnormTensors?: number;
}

export interface RigMemberAllocation {
  id: string;
  workerNodeId: string;
  irohNodeId: string;
  gpuName: string;
  pipelineStageIndex: number;
  assignedRole: string;
  allocatedVramMb: number;
  tensorSplitFraction: number;
  layerStart: number;
  layerEnd: number;
  byteRangeStart: number;
  byteRangeEnd: number;
}

export interface ManagedRig {
  id: string;
  rigName: string;
  modelId: string;
  modelDisplayName: string;
  lifecycleState: "SERVING" | "DISSOLVED" | "DEGRADED";
  orchestrationMode: "AUTO_PILOT" | "MANUAL_ONLY";
  securityProfile: string;
  totalAllocatedVramMb: number;
  requiredModelVramMb: number;
  autoMaximizedContextSize: number;
  tensorSplitCsv: string;
  antiCollusionVerified: boolean;
  servedRequestsCount: number;
  createdAt: string;
  allocations: RigMemberAllocation[];
  inferenceEngineReady?: boolean;
  llamaProcessAlive?: boolean;
  runtimePhase?: "LAYERS_ALLOCATED" | "PREWARMING_SHARDS" | "LOADING_VRAM" | "OPERATIONAL" | "ERROR_DEGRADED";
  runtimeStep?: number;
  runtimeLastLog?: string;
  operationalStatusMessage?: string;
  inferenceEndpointUrl?: string;
  activeInFlightRequests?: number;
  avgLatencyMs?: number;
  lastPrefillTokPerSec?: number;
  lastDecodeTokPerSec?: number;
}

export interface SecurityAuditEvent {
  id: string;
  category: string;
  severity: string;
  titleEn: string;
  titleIt: string;
  summaryEn: string;
  summaryIt: string;
  subjectId?: string;
  passed: boolean;
  f32RelativeErr?: number;
  q8ToQ5RelativeErr?: number;
  canaryRelativeErr?: number;
  rank1CorrMax?: number;
  antiCollusionEnforced?: boolean;
  metricsJson?: string;
  createdAt: string;
}

export interface CovariantAuditReport {
  auditEventId: string;
  dimension: number;
  intermediateDimension: number;
  f32RelativeErr: number;
  q80ToQ50RelativeErr: number;
  q80CanaryRelativeErr: number;
  rank1EmpiricalCorrMax: number;
  antiCollusionEnforced: boolean;
  canaryVerified: boolean;
  auditPassed: boolean;
  executionSource: string;
  elapsedMs: number;
  rawProtocolAudit?: Record<string, any>;
}

export interface OpenAiApiKeyItem {
  id: string;
  name: string;
  keyPrefix: string;
  rateLimitRpm: number;
  totalRequests: number;
  totalTokensGenerated: number;
  revoked: boolean;
  createdAt: string;
}

export interface CutmeshTelemetry {
  rig_id: string;
  rig_name: string;
  sticky_session_hit: boolean;
  security_profile: string;
  tensor_split_csv: string;
  context_window: number;
  anti_collusion_verified: boolean;
  canary_drift_l2: number;
  canary_status: string;
  latency_ms: number;
  participating_gpus: string[];
  prompt_ms?: number;
  predicted_ms?: number;
  prompt_per_second?: number;
  predicted_per_second?: number;
  thinking_tokens?: number;
}

export interface ChatMessageItem {
  role: "user" | "assistant";
  content: string;
  reasoningContent?: string;
  isThinking?: boolean;
  isStreaming?: boolean;
  latencyMs?: number;
  promptMs?: number;
  predictedMs?: number;
  promptTokens?: number;
  completionTokens?: number;
  thinkingTokens?: number;
  tokens?: number;
  tokPerSec?: number;
  promptTokPerSec?: number;
  finishReason?: string;
}
