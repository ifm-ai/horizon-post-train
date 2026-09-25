//! Scheduler config: on-disk YAML shape + runtime form.

use std::{collections::HashMap, time::Duration};

use serde::{Deserialize, Serialize};

use super::Class;
use crate::tenant::TenantKey;

/// Per-class configuration as it appears in the optional YAML file.
///
/// Lives separately from [`ClassRuntimeConfig`] because the YAML form
/// uses primitive types friendly to serde and human editing, while the
/// runtime form pre-converts seconds into [`Duration`] so hot paths
/// don't repeat the conversion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClassConfig {
    /// Absolute minimum slots reserved for this class — the value the
    /// effective reservation never drops below. Higher-class reservations
    /// are honored by lower-class admissions via the packed-CAS slot
    /// accounting in [`super::slots`].
    pub reserved_floor: u16,
    /// Share of backend capacity reserved for this class, on top of the
    /// floor: `effective = max(reserved_floor, ceil(reserved_per_slot ×
    /// capacity))`, recomputed as capacity changes. `0.0` (the default) is
    /// purely absolute — identical to a fixed `reserved_floor`. Must be
    /// finite and non-negative.
    #[serde(default)]
    pub reserved_per_slot: f64,
    /// Per-class soft share of the global work-conserving queue budget.
    /// Dispatch remains class-aware, but an active class may borrow unused
    /// queue capacity from another class.
    pub queue_size: u32,
    /// How long a queued waiter waits before the admission middleware
    /// returns 408. Seconds at rest; converted to [`Duration`] in
    /// [`ClassRuntimeConfig`].
    pub queue_timeout_secs: u64,
    /// Head-of-queue age past which the dispatcher promotes a waiter
    /// out of normal priority order to avoid starvation. Seconds at
    /// rest; converted to [`Duration`] in [`ClassRuntimeConfig`].
    pub starvation_threshold_secs: u64,
    /// Whether admissions in this class are allowed to preempt a
    /// lower-class inflight request that has not yet emitted its first
    /// byte. Higher classes default to `true`; lower classes default
    /// to `false`.
    pub can_preempt: bool,
}

impl ClassConfig {
    /// Built-in defaults. These are what every class gets when no YAML
    /// file supplies an override.
    pub fn default_for(class: Class) -> Self {
        match class {
            Class::System => Self {
                reserved_floor: 32,
                reserved_per_slot: 0.0,
                queue_size: 64,
                queue_timeout_secs: 30,
                starvation_threshold_secs: 5,
                can_preempt: true,
            },
            Class::Interactive => Self {
                reserved_floor: 128,
                reserved_per_slot: 0.25,
                queue_size: 256,
                queue_timeout_secs: 30,
                starvation_threshold_secs: 5,
                can_preempt: true,
            },
            Class::Default => Self {
                reserved_floor: 0,
                reserved_per_slot: 0.10,
                queue_size: 512,
                queue_timeout_secs: 60,
                starvation_threshold_secs: 30,
                can_preempt: false,
            },
            Class::Bulk => Self {
                reserved_floor: 0,
                reserved_per_slot: 0.0,
                queue_size: 1024,
                queue_timeout_secs: 300,
                starvation_threshold_secs: 120,
                can_preempt: false,
            },
        }
    }
}

/// Runtime view of [`ClassConfig`] — only the fields the dispatcher
/// reads on its hot path, with seconds pre-converted to [`Duration`].
/// `reserved_floor`/`reserved_per_slot` and `queue_size` live elsewhere
/// (the packed-CAS array and the per-class queue impl), so they don't
/// appear here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassRuntimeConfig {
    pub queue_timeout: Duration,
    pub starvation_threshold: Duration,
    pub can_preempt: bool,
}

impl ClassRuntimeConfig {
    pub fn from_class_config(cfg: &ClassConfig) -> Self {
        Self {
            queue_timeout: Duration::from_secs(cfg.queue_timeout_secs),
            starvation_threshold: Duration::from_secs(cfg.starvation_threshold_secs),
            can_preempt: cfg.can_preempt,
        }
    }
}

/// Per-tenant policy entry in the YAML file.
///
/// Future fields (`weight`, `slot_quota`, `rps_cap`) are additive: adding
/// them is non-breaking because the trait
/// [`super::policy::TenantPolicyResolver`] returns the whole struct.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TenantPolicyConfig {
    pub max_class: Class,
}

fn default_fair_share_weight() -> f64 {
    1.0
}

fn default_fair_share_output_tokens() -> u32 {
    256
}

/// One model's hierarchical weighted-sharing policy.
///
/// Explicit tenants and the aggregate `other` bucket contend at the outer
/// level. Real tenant identities are retained inside `other` and share that
/// bucket equally, so settlement and accounting never collapse to a synthetic
/// tenant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelFairShareConfig {
    /// Per-tenant outer weights keyed by canonical `TenantKey` string.
    #[serde(default)]
    pub tenant_weights: HashMap<String, f64>,
    /// Aggregate outer weight for every tenant absent from `tenant_weights`.
    #[serde(default = "default_fair_share_weight")]
    pub other_weight: f64,
}

/// Flat process-wide sharing plus optional hierarchical per-model profiles.
///
/// Weights are relative and need not sum to 100. For example, weights 10 and
/// 5 give two continuously contending tenants a 2:1 output-token share.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FairShareConfig {
    /// Weight for a resolved tenant absent from `tenant_weights`.
    #[serde(default = "default_fair_share_weight")]
    pub default_weight: f64,
    /// Provisional charge when the trusted request estimate is absent or
    /// invalid. Terminal response usage replaces this estimate.
    #[serde(default = "default_fair_share_output_tokens")]
    pub default_output_tokens: u32,
    /// Honor `x-smg-output-token-estimate`. Keep false unless a trusted proxy
    /// strips client copies and injects a validated value.
    #[serde(default)]
    pub trust_output_token_estimate_header: bool,
    /// Honor `x-smg-request-model` when selecting a per-model profile. Keep
    /// false unless a trusted proxy strips client copies and injects the
    /// authenticated request body's model value.
    #[serde(default)]
    pub trust_request_model_header: bool,
    /// Per-tenant relative weights keyed by canonical `TenantKey` string.
    #[serde(default)]
    pub tenant_weights: HashMap<String, f64>,
    /// Optional hierarchical policies keyed by canonical model id. Absence
    /// preserves the original flat process-wide ledger exactly.
    #[serde(default)]
    pub model_profiles: HashMap<String, ModelFairShareConfig>,
}

/// Admission budget for one trusted upstream partition selector.
///
/// A deployment chooses one capacity mode for every configured partition:
///
/// - static: set `max_concurrent_requests` (the original behavior), or
/// - replica-aware: set `capacity_from_healthy_replicas: true`, omit the static
///   maximum, and either derive each worker's ceiling from its reported
///   `max_running_requests` or explicitly configure a per-replica override.
///
/// Replica-aware partitions divide the live global scheduler capacity in
/// proportion to the number of healthy workers assigned to each partition.
/// A worker's `admission_partition` metadata label takes precedence over its
/// primary model id, which lets an upstream controller move reserved replicas
/// into a private lane without restarting the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionPartitionConfig {
    /// Static maximum requests admitted concurrently in this partition.
    /// Required in static mode and omitted in replica-aware mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrent_requests: Option<u16>,
    /// Derive this partition's share from its healthy replica count.
    #[serde(default)]
    pub capacity_from_healthy_replicas: bool,
    /// Optional admission slots contributed by each healthy replica in
    /// replica-aware mode. When omitted, use worker-reported
    /// `max_running_requests`. Omitted in static mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrent_requests_per_healthy_replica: Option<u16>,
    /// Work-conserving queue budget shared by the priority classes inside
    /// this partition.
    pub queue_size: u32,
}

fn default_admission_partition() -> String {
    "default".to_string()
}

/// Optional YAML config loaded via `--priority-scheduler-config <path>`.
///
/// Both maps are absent-as-empty: an empty document parses to
/// `PrioritySchedulerYaml::default()`, and downstream
/// [`SchedulerSettings::from_cli_and_yaml`] fills in built-in defaults
/// for any class that wasn't overridden.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritySchedulerYaml {
    #[serde(default)]
    pub classes: HashMap<Class, ClassConfig>,
    #[serde(default)]
    pub tenant_policies: HashMap<String, TenantPolicyConfig>,
    /// Optional global weighted output-token ledger. Absence preserves FIFO
    /// queueing and all existing scheduler behavior.
    #[serde(default)]
    pub fair_share: Option<FairShareConfig>,
    /// Optional admission partitions keyed by the exact value of the trusted
    /// `x-smg-admission-partition` header. An empty map preserves the original
    /// single global scheduler.
    #[serde(default)]
    pub admission_partitions: HashMap<String, AdmissionPartitionConfig>,
    /// Partition used when the trusted header is absent, invalid, or does not
    /// match a configured key.
    #[serde(default = "default_admission_partition")]
    pub default_admission_partition: String,
}

impl Default for PrioritySchedulerYaml {
    fn default() -> Self {
        Self {
            classes: HashMap::new(),
            tenant_policies: HashMap::new(),
            fair_share: None,
            admission_partitions: HashMap::new(),
            default_admission_partition: default_admission_partition(),
        }
    }
}

/// Per-field validation failures discovered while assembling
/// [`SchedulerSettings`]. Capacity-vs-reserved is not checked here: the
/// scheduler derives effective reservations from the floors + shares and
/// clamps them to the live capacity (priority-ordered), so reservations
/// can never exceed capacity and there is nothing to reject.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SettingsValidationError {
    #[error("class {class:?}: queue_timeout_secs must be > 0")]
    ZeroQueueTimeout { class: Class },
    #[error("class {class:?}: starvation_threshold_secs must be > 0")]
    ZeroStarvationThreshold { class: Class },
    #[error("class {class:?}: reserved_per_slot must be finite and >= 0")]
    InvalidReservedPerSlot { class: Class },
    #[error("fair_share.default_weight must be finite and > 0")]
    InvalidFairShareDefaultWeight,
    #[error("fair_share.default_output_tokens must be > 0")]
    ZeroFairShareDefaultOutputTokens,
    #[error("fair_share.tenant_weights[{tenant:?}] must be finite and > 0")]
    InvalidFairShareTenantWeight { tenant: String },
    #[error("fair_share.model_profiles require trust_request_model_header=true")]
    ModelProfilesRequireTrustedModelHeader,
    #[error("fair_share.model_profiles contains an empty or untrimmed model key {model:?}")]
    InvalidFairShareModelKey { model: String },
    #[error(
        "fair_share.model_profiles[{model:?}].tenant_weights[{tenant:?}] must be finite and > 0"
    )]
    InvalidModelFairShareTenantWeight { model: String, tenant: String },
    #[error("fair_share.model_profiles[{model:?}].other_weight must be finite and > 0")]
    InvalidModelFairShareOtherWeight { model: String },
}

/// Runtime scheduler configuration assembled from CLI flags + the
/// optional YAML file + built-in defaults. Built once at startup and
/// then read-only.
///
/// Capacity is not stored here; the scheduler reads it live from
/// [`crate::worker::WorkerCapacity`] and reacts to changes via its
/// watch channel. Settings stay fixed.
#[derive(Debug, Clone)]
pub struct SchedulerSettings {
    /// Master switch. When `false`, the priority scheduler is not
    /// constructed and the gateway falls back to its legacy
    /// concurrency-limit admission path.
    pub enabled: bool,
    /// Tenant policy applied when a tenant is not in `tenant_policies`.
    pub default_max_class: Class,
    /// Per-class config, indexed by `class as usize`. Always populated
    /// for all four classes — YAML overrides land on top of
    /// [`ClassConfig::default_for`].
    classes: [ClassConfig; 4],
    /// Per-tenant clamp lookup. Keys come from
    /// [`crate::tenant::RouteRequestMeta::tenant_key`].
    pub tenant_policies: HashMap<TenantKey, TenantPolicyConfig>,
    fair_share: Option<FairShareConfig>,
    /// Cap on the number of tenants emitted as labels for
    /// tenant-labeled scheduler and fair-share metrics. Everything past the
    /// cap is bucketed under `tenant="other"`.
    pub tenant_metric_top_n: u32,
}

impl SchedulerSettings {
    /// Indexed accessor for the per-class config.
    pub fn class_config(&self, class: Class) -> &ClassConfig {
        &self.classes[class as usize]
    }

    #[must_use]
    pub fn fair_share_config(&self) -> Option<&FairShareConfig> {
        self.fair_share.as_ref()
    }

    /// Scale the built-in per-class queue weights to one exact global budget.
    /// This is used when no scheduler YAML overrides the class limits, so the
    /// legacy `--queue-size` flag remains the total queue contract after the
    /// priority scheduler is enabled.
    pub fn with_global_queue_budget(mut self, total: usize) -> Self {
        let total = u32::try_from(total).unwrap_or(u32::MAX);
        let current_total: u64 = Class::ALL
            .iter()
            .map(|class| u64::from(self.classes[*class as usize].queue_size))
            .sum();
        if current_total == 0 {
            return self;
        }

        let mut allocated = 0_u32;
        for (index, class) in Class::ALL.iter().enumerate() {
            let queue_size = if index + 1 == Class::ALL.len() {
                total.saturating_sub(allocated)
            } else {
                let weight = u64::from(self.classes[*class as usize].queue_size);
                let scaled = (u64::from(total) * weight) / current_total;
                u32::try_from(scaled).unwrap_or(u32::MAX)
            };
            self.classes[*class as usize].queue_size = queue_size;
            allocated = allocated.saturating_add(queue_size);
        }
        self
    }

    /// Scale absolute class floors to a partition's share of the configured
    /// global capacity, then apply that partition's queue budget. Percentage
    /// reservations remain unchanged and therefore continue to scale against
    /// the partition's live capacity.
    pub fn for_admission_partition(
        mut self,
        partition_capacity: u16,
        global_capacity: u16,
        queue_size: usize,
    ) -> Self {
        if global_capacity > 0 {
            for class in Class::ALL {
                let floor = u64::from(self.classes[class as usize].reserved_floor);
                let numerator = floor.saturating_mul(u64::from(partition_capacity));
                let scaled = numerator.div_ceil(u64::from(global_capacity));
                self.classes[class as usize].reserved_floor =
                    u16::try_from(scaled).unwrap_or(u16::MAX);
            }
        }
        self.with_global_queue_budget(queue_size)
    }

    /// Assemble settings from CLI flags + optional YAML, validating
    /// per-field invariants. Reservations are clamped to the live capacity
    /// by the scheduler, so there is no capacity-vs-reserved check here.
    ///
    /// Merge order: built-in defaults → YAML overrides per class. CLI
    /// flags supply the top-level fields (`enabled`, `default_max_class`,
    /// `tenant_metric_top_n`) since they aren't representable in the YAML.
    pub fn from_cli_and_yaml(
        enabled: bool,
        default_max_class: Class,
        tenant_metric_top_n: u32,
        yaml: Option<&PrioritySchedulerYaml>,
    ) -> Result<Self, SettingsValidationError> {
        let mut classes: [ClassConfig; 4] = [
            ClassConfig::default_for(Class::Bulk),
            ClassConfig::default_for(Class::Default),
            ClassConfig::default_for(Class::Interactive),
            ClassConfig::default_for(Class::System),
        ];

        if let Some(yaml) = yaml {
            for (class, override_cfg) in &yaml.classes {
                classes[*class as usize] = *override_cfg;
            }
        }

        for class in Class::ALL {
            let cfg = &classes[class as usize];
            if cfg.queue_timeout_secs == 0 {
                return Err(SettingsValidationError::ZeroQueueTimeout { class });
            }
            if cfg.starvation_threshold_secs == 0 {
                return Err(SettingsValidationError::ZeroStarvationThreshold { class });
            }
            if !cfg.reserved_per_slot.is_finite() || cfg.reserved_per_slot < 0.0 {
                return Err(SettingsValidationError::InvalidReservedPerSlot { class });
            }
        }

        let fair_share = yaml.and_then(|value| value.fair_share.clone());
        if let Some(config) = &fair_share {
            if !config.default_weight.is_finite() || config.default_weight <= 0.0 {
                return Err(SettingsValidationError::InvalidFairShareDefaultWeight);
            }
            if config.default_output_tokens == 0 {
                return Err(SettingsValidationError::ZeroFairShareDefaultOutputTokens);
            }
            for (tenant, weight) in &config.tenant_weights {
                if !weight.is_finite() || *weight <= 0.0 {
                    return Err(SettingsValidationError::InvalidFairShareTenantWeight {
                        tenant: tenant.clone(),
                    });
                }
            }
            if !config.model_profiles.is_empty() && !config.trust_request_model_header {
                return Err(SettingsValidationError::ModelProfilesRequireTrustedModelHeader);
            }
            for (model, profile) in &config.model_profiles {
                if model.trim().is_empty() || model.trim() != model {
                    return Err(SettingsValidationError::InvalidFairShareModelKey {
                        model: model.clone(),
                    });
                }
                if !profile.other_weight.is_finite() || profile.other_weight <= 0.0 {
                    return Err(SettingsValidationError::InvalidModelFairShareOtherWeight {
                        model: model.clone(),
                    });
                }
                for (tenant, weight) in &profile.tenant_weights {
                    if !weight.is_finite() || *weight <= 0.0 {
                        return Err(SettingsValidationError::InvalidModelFairShareTenantWeight {
                            model: model.clone(),
                            tenant: tenant.clone(),
                        });
                    }
                }
            }
        }

        let tenant_policies = yaml
            .map(|y| {
                y.tenant_policies
                    .iter()
                    .map(|(k, v)| (TenantKey::new(k), *v))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            enabled,
            default_max_class,
            classes,
            tenant_policies,
            fair_share,
            tenant_metric_top_n,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::middleware::scheduler::Class;

    #[test]
    fn test_default_for_system() {
        let cfg = ClassConfig::default_for(Class::System);
        assert_eq!(cfg.reserved_floor, 32);
        assert_eq!(cfg.queue_size, 64);
        assert_eq!(cfg.queue_timeout_secs, 30);
        assert_eq!(cfg.starvation_threshold_secs, 5);
        assert!(cfg.can_preempt);
    }

    #[test]
    fn test_default_for_interactive() {
        let cfg = ClassConfig::default_for(Class::Interactive);
        assert_eq!(cfg.reserved_floor, 128);
        assert_eq!(cfg.queue_size, 256);
        assert_eq!(cfg.queue_timeout_secs, 30);
        assert_eq!(cfg.starvation_threshold_secs, 5);
        assert!(cfg.can_preempt);
    }

    #[test]
    fn test_default_for_default() {
        let cfg = ClassConfig::default_for(Class::Default);
        assert_eq!(cfg.reserved_floor, 0);
        assert_eq!(cfg.queue_size, 512);
        assert_eq!(cfg.queue_timeout_secs, 60);
        assert_eq!(cfg.starvation_threshold_secs, 30);
        assert!(!cfg.can_preempt);
    }

    #[test]
    fn test_default_for_bulk() {
        let cfg = ClassConfig::default_for(Class::Bulk);
        assert_eq!(cfg.reserved_floor, 0);
        assert_eq!(cfg.queue_size, 1024);
        assert_eq!(cfg.queue_timeout_secs, 300);
        assert_eq!(cfg.starvation_threshold_secs, 120);
        assert!(!cfg.can_preempt);
    }

    #[test]
    fn test_runtime_config_converts_seconds_to_duration() {
        let cfg = ClassConfig::default_for(Class::Default);
        let runtime = ClassRuntimeConfig::from_class_config(&cfg);
        assert_eq!(runtime.queue_timeout, Duration::from_secs(60));
        assert_eq!(runtime.starvation_threshold, Duration::from_secs(30));
        assert!(!runtime.can_preempt);
    }

    #[test]
    fn test_runtime_config_preserves_can_preempt_flag() {
        let interactive =
            ClassRuntimeConfig::from_class_config(&ClassConfig::default_for(Class::Interactive));
        assert!(interactive.can_preempt);
        let bulk = ClassRuntimeConfig::from_class_config(&ClassConfig::default_for(Class::Bulk));
        assert!(!bulk.can_preempt);
    }

    // ── PrioritySchedulerYaml serde ───────────────────────────────────

    #[test]
    fn test_yaml_empty_document_yields_default() {
        let parsed: PrioritySchedulerYaml = serde_yaml::from_str("").unwrap();
        assert!(parsed.classes.is_empty());
        assert!(parsed.tenant_policies.is_empty());
    }

    #[test]
    fn test_yaml_partial_class_override_round_trips() {
        let yaml = r"
classes:
  interactive:
    reserved_floor: 200
    queue_size: 256
    queue_timeout_secs: 30
    starvation_threshold_secs: 5
    can_preempt: true
";
        let parsed: PrioritySchedulerYaml = serde_yaml::from_str(yaml).unwrap();
        let interactive = parsed
            .classes
            .get(&Class::Interactive)
            .expect("interactive present");
        assert_eq!(interactive.reserved_floor, 200);
        // Only one class entry — others are absent (settings layer fills defaults).
        assert_eq!(parsed.classes.len(), 1);
        assert!(parsed.tenant_policies.is_empty());
    }

    #[test]
    fn test_yaml_tenant_policy_round_trips() {
        let yaml = r#"
tenant_policies:
  "auth:acme":
    max_class: interactive
  "auth:internal-cron":
    max_class: system
"#;
        let parsed: PrioritySchedulerYaml = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(parsed.tenant_policies.len(), 2);
        assert_eq!(
            parsed.tenant_policies["auth:acme"].max_class,
            Class::Interactive
        );
        assert_eq!(
            parsed.tenant_policies["auth:internal-cron"].max_class,
            Class::System
        );
    }

    #[test]
    fn test_yaml_fair_share_weights_round_trip() {
        let yaml = r#"
fair_share:
  default_weight: 0.5
  default_output_tokens: 128
  trust_output_token_estimate_header: true
  tenant_weights:
    "header:alice": 10
    "header:bob": 5
"#;
        let parsed: PrioritySchedulerYaml = serde_yaml::from_str(yaml).unwrap();
        let fair_share = parsed.fair_share.as_ref().unwrap();
        assert_eq!(fair_share.default_weight, 0.5);
        assert_eq!(fair_share.default_output_tokens, 128);
        assert!(fair_share.trust_output_token_estimate_header);
        assert_eq!(fair_share.tenant_weights["header:alice"], 10.0);
        assert_eq!(fair_share.tenant_weights["header:bob"], 5.0);

        let settings =
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&parsed)).unwrap();
        assert_eq!(settings.fair_share_config(), Some(fair_share));
    }

    #[test]
    fn test_invalid_fair_share_weights_are_rejected() {
        for weight in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let yaml = PrioritySchedulerYaml {
                fair_share: Some(FairShareConfig {
                    default_weight: 1.0,
                    default_output_tokens: 128,
                    trust_output_token_estimate_header: false,
                    trust_request_model_header: false,
                    tenant_weights: HashMap::from([("header:alice".to_string(), weight)]),
                    model_profiles: HashMap::new(),
                }),
                ..Default::default()
            };
            assert!(matches!(
                SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)),
                Err(SettingsValidationError::InvalidFairShareTenantWeight { .. })
            ));
        }
    }

    #[test]
    fn test_model_profiles_require_the_trusted_request_model_header() {
        let yaml = PrioritySchedulerYaml {
            fair_share: Some(FairShareConfig {
                default_weight: 1.0,
                default_output_tokens: 128,
                trust_output_token_estimate_header: false,
                trust_request_model_header: false,
                tenant_weights: HashMap::new(),
                model_profiles: HashMap::from([(
                    "kimi-k3".to_string(),
                    ModelFairShareConfig {
                        tenant_weights: HashMap::new(),
                        other_weight: 20.0,
                    },
                )]),
            }),
            ..Default::default()
        };

        assert!(matches!(
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)),
            Err(SettingsValidationError::ModelProfilesRequireTrustedModelHeader)
        ));
    }

    #[test]
    fn test_per_model_hierarchical_weights_round_trip() {
        let parsed: PrioritySchedulerYaml = serde_yaml::from_str(
            r#"
fair_share:
  trust_request_model_header: true
  model_profiles:
    deepseek-v4-flash:
      tenant_weights:
        "header:junu": 30
        "header:xuezhou": 40
        "header:zhenting": 10
      other_weight: 20
    kimi-k3:
      tenant_weights:
        "header:mukhesh": 80
      other_weight: 20
"#,
        )
        .unwrap();
        let fair_share = parsed.fair_share.as_ref().unwrap();
        assert!(fair_share.trust_request_model_header);
        assert_eq!(
            fair_share.model_profiles["deepseek-v4-flash"].tenant_weights["header:junu"],
            30.0
        );
        assert_eq!(
            fair_share.model_profiles["deepseek-v4-flash"].other_weight,
            20.0
        );
        assert_eq!(
            fair_share.model_profiles["kimi-k3"].tenant_weights["header:mukhesh"],
            80.0
        );
        SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&parsed)).unwrap();
    }

    #[test]
    fn test_yaml_admission_partitions_round_trip() {
        let yaml = r#"
admission_partitions:
  kimi-k3:
    max_concurrent_requests: 7168
    queue_size: 1600
  private:
    max_concurrent_requests: 64
    queue_size: 40
  default:
    max_concurrent_requests: 128
    queue_size: 80
default_admission_partition: default
"#;
        let parsed: PrioritySchedulerYaml = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(parsed.admission_partitions.len(), 3);
        assert_eq!(
            parsed.admission_partitions["kimi-k3"],
            AdmissionPartitionConfig {
                max_concurrent_requests: Some(7168),
                capacity_from_healthy_replicas: false,
                max_concurrent_requests_per_healthy_replica: None,
                queue_size: 1600,
            }
        );
        assert_eq!(parsed.default_admission_partition, "default");
    }

    #[test]
    fn test_yaml_replica_aware_admission_partition_round_trip() {
        let yaml = r#"
admission_partitions:
  kimi-k3:
    capacity_from_healthy_replicas: true
    max_concurrent_requests_per_healthy_replica: 34
    queue_size: 1600
  default:
    capacity_from_healthy_replicas: true
    max_concurrent_requests_per_healthy_replica: 34
    queue_size: 80
default_admission_partition: default
"#;
        let parsed: PrioritySchedulerYaml = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            parsed.admission_partitions["kimi-k3"],
            AdmissionPartitionConfig {
                max_concurrent_requests: None,
                capacity_from_healthy_replicas: true,
                max_concurrent_requests_per_healthy_replica: Some(34),
                queue_size: 1600,
            }
        );
    }

    #[test]
    fn test_yaml_unknown_class_value_is_serde_error() {
        let yaml = r#"
tenant_policies:
  "auth:acme":
    max_class: garbage
"#;
        let result: Result<PrioritySchedulerYaml, _> = serde_yaml::from_str(yaml);
        assert!(result.is_err(), "expected serde error for unknown class");
    }

    #[test]
    fn test_yaml_class_name_serializes_as_lowercase() {
        let mut classes = HashMap::new();
        classes.insert(Class::Bulk, ClassConfig::default_for(Class::Bulk));
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: Default::default(),
            ..Default::default()
        };
        let rendered = serde_yaml::to_string(&yaml).unwrap();
        assert!(
            rendered.contains("bulk:"),
            "class key should serialize as lowercase: {rendered}"
        );
    }

    // ── SchedulerSettings::from_cli_and_yaml ─────────────────────────

    use crate::tenant::TenantKey;

    #[test]
    fn test_settings_no_yaml_uses_builtin_defaults() {
        let s = SchedulerSettings::from_cli_and_yaml(false, Class::Default, 32, None).unwrap();
        assert!(!s.enabled);
        assert_eq!(s.default_max_class, Class::Default);
        assert_eq!(s.tenant_metric_top_n, 32);
        for class in Class::ALL {
            assert_eq!(s.class_config(class), &ClassConfig::default_for(class));
        }
        assert!(s.tenant_policies.is_empty());
    }

    #[test]
    fn test_global_queue_budget_scales_defaults_to_exact_total() {
        let s = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, None)
            .unwrap()
            .with_global_queue_budget(2_000);
        let total: u32 = Class::ALL
            .iter()
            .map(|class| s.class_config(*class).queue_size)
            .sum();
        assert_eq!(total, 2_000);
        assert!(
            s.class_config(Class::Bulk).queue_size > s.class_config(Class::Interactive).queue_size
        );
    }

    #[test]
    fn test_global_queue_budget_can_disable_queues() {
        let s = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, None)
            .unwrap()
            .with_global_queue_budget(0);
        assert!(Class::ALL
            .iter()
            .all(|class| s.class_config(*class).queue_size == 0));
    }

    #[test]
    fn test_admission_partition_scales_absolute_floors_but_keeps_shares() {
        let settings = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, None)
            .unwrap()
            .for_admission_partition(64, 8000, 40);
        assert_eq!(settings.class_config(Class::System).reserved_floor, 1);
        assert_eq!(settings.class_config(Class::Interactive).reserved_floor, 2);
        assert_eq!(
            settings.class_config(Class::Interactive).reserved_per_slot,
            0.25
        );
        assert_eq!(
            settings.class_config(Class::Default).reserved_per_slot,
            0.10
        );
        assert_eq!(
            Class::ALL
                .iter()
                .map(|class| settings.class_config(*class).queue_size)
                .sum::<u32>(),
            40
        );
    }

    #[test]
    fn test_settings_yaml_partial_override_merges_with_defaults() {
        let mut classes = HashMap::new();
        let mut interactive = ClassConfig::default_for(Class::Interactive);
        interactive.reserved_floor = 64;
        classes.insert(Class::Interactive, interactive);
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: Default::default(),
            ..Default::default()
        };
        let s =
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();
        assert_eq!(s.class_config(Class::Interactive).reserved_floor, 64);
        // Bulk untouched — still equal to built-in default.
        assert_eq!(
            s.class_config(Class::Bulk),
            &ClassConfig::default_for(Class::Bulk)
        );
    }

    #[test]
    fn test_settings_yaml_tenant_policy_propagates() {
        let mut tenant_policies = HashMap::new();
        tenant_policies.insert(
            "auth:acme".to_string(),
            TenantPolicyConfig {
                max_class: Class::Interactive,
            },
        );
        let yaml = PrioritySchedulerYaml {
            classes: Default::default(),
            tenant_policies,
            ..Default::default()
        };
        let s =
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();
        assert_eq!(s.tenant_policies.len(), 1);
        let key = TenantKey::new("auth:acme");
        assert_eq!(s.tenant_policies[&key].max_class, Class::Interactive);
    }

    #[test]
    fn test_settings_rejects_zero_queue_timeout() {
        let mut classes = HashMap::new();
        let mut bulk = ClassConfig::default_for(Class::Bulk);
        bulk.queue_timeout_secs = 0;
        classes.insert(Class::Bulk, bulk);
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: Default::default(),
            ..Default::default()
        };
        let err = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml))
            .unwrap_err();
        assert!(matches!(
            err,
            SettingsValidationError::ZeroQueueTimeout { class: Class::Bulk }
        ));
    }

    #[test]
    fn test_settings_rejects_zero_starvation_threshold() {
        let mut classes = HashMap::new();
        let mut bulk = ClassConfig::default_for(Class::Bulk);
        bulk.starvation_threshold_secs = 0;
        classes.insert(Class::Bulk, bulk);
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: Default::default(),
            ..Default::default()
        };
        let err = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml))
            .unwrap_err();
        assert!(matches!(
            err,
            SettingsValidationError::ZeroStarvationThreshold { class: Class::Bulk }
        ));
    }
}
