from pydantic import BaseModel, Field


class SessionRecord(BaseModel):
    timestamp: float
    method: str
    path: str
    request: dict
    response: dict
    status_code: int
    rollout_sampling_mask: dict[str, str] | None = None
    rollout_sampling_log_probs: list[float] | None = None


class GetSessionResponse(BaseModel):
    session_id: str
    records: list[SessionRecord]
    metadata: dict = Field(default_factory=dict)


class MergedSessionSample(BaseModel):
    tokens: list[int]
    response: str = ""
    response_length: int
    loss_mask: list[int]
    rollout_log_probs: list[float]
    rollout_sampling_mask: dict[str, str] | None = None
    status: str
    metadata: dict = Field(default_factory=lambda: {"response_decoded": False})
    weight_versions: list[str] = Field(default_factory=list)
    prefix_cache_meta_infos: list[dict] = Field(default_factory=list)
    rollout_routed_experts: str | None = None


class GetMergedSessionResponse(BaseModel):
    session_id: str
    sample: MergedSessionSample | None = None
    metadata: dict = Field(default_factory=dict)
