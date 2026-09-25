//! Private, one-shot proof that a request owns an externally authorized
//! fair-share scheduler reservation.

use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

use thiserror::Error;
use uuid::Uuid;

use super::capacity_credit::CapacityCreditBinding;
use crate::tenant::TenantKey;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SchedulerAdmissionSource {
    #[expect(
        dead_code,
        reason = "reserved for a future scheduler-owned proof; only redeemed credits mint proofs today"
    )]
    SchedulerReservation,
    RedeemedCapacityCredit,
}

struct SchedulerAdmissionProofInner {
    source: SchedulerAdmissionSource,
    tenant: TenantKey,
    model: Arc<str>,
    partition: Arc<str>,
    request_id: Arc<str>,
    credit_generation: Arc<str>,
    policy_epoch: u64,
    route_request_id: Uuid,
    state: AtomicU8,
}

const PROOF_FRESH: u8 = 0;
const PROOF_CLAIMED: u8 = 1;
const PROOF_REVOKED: u8 = 2;

/// Opaque witness attached only after a capacity credit is successfully
/// redeemed. Clones share one Fresh/Claimed/Revoked state machine, so retries,
/// cancellation, and metadata cloning cannot reuse stale authorization.
#[derive(Clone, Debug)]
pub(crate) struct SchedulerAdmissionProof {
    inner: Arc<SchedulerAdmissionProofInner>,
}

/// One successful claim of a scheduler admission proof.
///
/// This is deliberately opaque and non-cloneable so downstream code must move
/// it into exactly one dispatch guard. It authorizes no settlement and owns no
/// scheduler capacity; the original `SchedulerPermit` remains authoritative.
#[derive(Debug)]
pub(crate) struct ClaimedSchedulerAdmissionProof {
    _inner: Arc<SchedulerAdmissionProofInner>,
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub(crate) enum ProofError {
    #[error("scheduler admission proof tenant mismatch")]
    TenantMismatch,
    #[error("scheduler admission proof model mismatch")]
    ModelMismatch,
    #[error("scheduler admission proof partition mismatch")]
    PartitionMismatch,
    #[error("scheduler admission proof was already claimed")]
    AlreadyClaimed,
    #[error("scheduler admission proof request mismatch")]
    RequestMismatch,
    #[error("scheduler admission proof was revoked")]
    Revoked,
}

impl SchedulerAdmissionProof {
    pub(super) fn from_redeemed_capacity_credit(
        binding: &CapacityCreditBinding,
        route_request_id: Uuid,
    ) -> Self {
        Self {
            inner: Arc::new(SchedulerAdmissionProofInner {
                source: SchedulerAdmissionSource::RedeemedCapacityCredit,
                tenant: binding.tenant().clone(),
                model: Arc::from(binding.model()),
                partition: Arc::from(binding.partition()),
                request_id: Arc::from(binding.request_id()),
                credit_generation: Arc::from(binding.generation()),
                policy_epoch: binding.policy_epoch(),
                route_request_id,
                state: AtomicU8::new(PROOF_FRESH),
            }),
        }
    }

    /// Claim this proof once for the exact canonical request scope.
    pub(crate) fn try_claim(
        &self,
        tenant: &TenantKey,
        route_request_id: Uuid,
        model: &str,
        partition: &str,
    ) -> Result<ClaimedSchedulerAdmissionProof, ProofError> {
        if &self.inner.tenant != tenant {
            return Err(ProofError::TenantMismatch);
        }
        if self.inner.route_request_id != route_request_id {
            return Err(ProofError::RequestMismatch);
        }
        if self.inner.model.as_ref() != model {
            return Err(ProofError::ModelMismatch);
        }
        if self.inner.partition.as_ref() != partition {
            return Err(ProofError::PartitionMismatch);
        }
        match self.inner.state.compare_exchange(
            PROOF_FRESH,
            PROOF_CLAIMED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(PROOF_CLAIMED) => return Err(ProofError::AlreadyClaimed),
            Err(PROOF_REVOKED) => return Err(ProofError::Revoked),
            Err(_) => return Err(ProofError::Revoked),
        }
        Ok(ClaimedSchedulerAdmissionProof {
            _inner: Arc::clone(&self.inner),
        })
    }

    pub(super) fn revoke(&self) {
        let _ = self.inner.state.compare_exchange(
            PROOF_FRESH,
            PROOF_REVOKED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

impl std::fmt::Debug for SchedulerAdmissionProofInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerAdmissionProofInner")
            .field("source", &self.source)
            .field("tenant", &self.tenant)
            .field("model", &self.model)
            .field("partition", &self.partition)
            .field("request_id", &self.request_id)
            .field("credit_generation", &self.credit_generation)
            .field("policy_epoch", &self.policy_epoch)
            .field("route_request_id", &self.route_request_id)
            .field("state", &self.state.load(Ordering::Acquire))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proof() -> (SchedulerAdmissionProof, Uuid) {
        let route_request_id = Uuid::now_v7();
        (
            SchedulerAdmissionProof::from_redeemed_capacity_credit(
                &CapacityCreditBinding::new(
                    "green-1",
                    7,
                    "kimi-k3",
                    "kimi-k3",
                    TenantKey::new("header:besher.hassan"),
                    "request-1",
                    256,
                )
                .unwrap(),
                route_request_id,
            ),
            route_request_id,
        )
    }

    #[test]
    fn exact_scope_claims_once_across_clones() {
        let (proof, request_id) = proof();
        let clone = proof.clone();
        assert!(proof
            .try_claim(
                &TenantKey::new("header:besher.hassan"),
                request_id,
                "kimi-k3",
                "kimi-k3"
            )
            .is_ok());
        assert!(matches!(
            clone.try_claim(
                &TenantKey::new("header:besher.hassan"),
                request_id,
                "kimi-k3",
                "kimi-k3"
            ),
            Err(ProofError::AlreadyClaimed)
        ));
    }

    #[test]
    fn scope_mismatch_fails_without_consuming_proof() {
        let (proof, request_id) = proof();
        assert!(matches!(
            proof.try_claim(
                &TenantKey::new("header:mallikarjuna"),
                request_id,
                "kimi-k3",
                "kimi-k3"
            ),
            Err(ProofError::TenantMismatch)
        ));
        assert!(matches!(
            proof.try_claim(
                &TenantKey::new("header:besher.hassan"),
                request_id,
                "deepseek-v4-flash-0731",
                "kimi-k3"
            ),
            Err(ProofError::ModelMismatch)
        ));
        assert!(matches!(
            proof.try_claim(
                &TenantKey::new("header:besher.hassan"),
                request_id,
                "kimi-k3",
                "private"
            ),
            Err(ProofError::PartitionMismatch)
        ));
        assert!(proof
            .try_claim(
                &TenantKey::new("header:besher.hassan"),
                request_id,
                "kimi-k3",
                "kimi-k3"
            )
            .is_ok());
    }

    #[test]
    fn different_request_metadata_cannot_claim() {
        let (proof, request_id) = proof();
        assert!(matches!(
            proof.try_claim(
                &TenantKey::new("header:besher.hassan"),
                Uuid::now_v7(),
                "kimi-k3",
                "kimi-k3"
            ),
            Err(ProofError::RequestMismatch)
        ));
        assert!(proof
            .try_claim(
                &TenantKey::new("header:besher.hassan"),
                request_id,
                "kimi-k3",
                "kimi-k3"
            )
            .is_ok());
    }
}
