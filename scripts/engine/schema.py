from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class Weights(BaseModel):
    w_trend: float
    w_momo: float
    w_mom_ret: float
    w_liq: float
    w_vol_guard: float
    w_beta: float
    w_sector: float
    w_sector_sent: float
    w_ticker_sent: float
    w_roe: float
    w_earnings_yield: float
    w_rs: float
    # Optional global scale for fundamentals weights to align horizon
    fund_scale: float = 1.0


class Thresholds(BaseModel):
    base_add: float
    base_new: float
    trim_th: float
    q_add: float
    q_new: float
    min_liq_norm: float
    near_ceiling_pct: float
    tp_pct: float
    sl_pct: float
    tp_atr_mult: Optional[float] = None
    sl_atr_mult: Optional[float] = None
    tp_floor_pct: Optional[float] = None
    tp_cap_pct: Optional[float] = None
    sl_floor_pct: Optional[float] = None
    sl_cap_pct: Optional[float] = None
    tp_trim_frac: float
    exit_on_ma_break: Union[int, bool]
    cooldown_days: int
    tp_sl_mode: Literal['legacy', 'atr_per_ticker'] = 'atr_per_ticker'
    # Multi-target ATR scaling parameters (optional)
    tp1_atr_mult: Optional[float] = Field(default=None, ge=0.8, le=1.3)
    tp2_atr_mult: Optional[float] = Field(default=None, ge=1.5, le=2.5)
    trailing_atr_mult: Optional[float] = Field(default=None, ge=1.1, le=2.0)
    trim_frac_tp1: Optional[float] = Field(default=None, ge=0.2, le=0.7)
    trim_frac_tp2: Optional[float] = Field(default=None, ge=0.2, le=0.6)
    breakeven_after_tp1: Optional[Union[int, bool]] = None
    time_stop_days: Optional[int] = Field(default=None, ge=3, le=7)
    trim_rsi_gate: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    cooldown_days_after_exit: Optional[int] = Field(default=None, ge=1, le=5)
    partial_entry_enabled: Union[int, bool] = 0
    partial_entry_frac: Optional[float] = Field(default=None, ge=0.1, le=0.5)
    partial_entry_floor_lot: int = Field(default=1, ge=1)
    new_partial_buffer: float = Field(default=0.05, ge=0.02, le=0.08)
    # New: parameterize RSI thresholds used in trim/exit heuristics
    exit_ma_break_rsi: float = Field(ge=0.0, le=100.0, default=45.0)
    trim_rsi_below_ma20: float = Field(ge=0.0, le=100.0, default=45.0)
    trim_rsi_macdh_neg: float = Field(ge=0.0, le=100.0, default=40.0)
    # New: soften MA-break exits when conviction score is high (0..1)
    exit_ma_break_score_gate: float = Field(ge=0.0, le=1.0, default=0.0)
    # New: minimum positive ticker_bias required to downgrade MA-break EXIT -> TRIM
    # Range 0..1. Recommended ~0.05 risk-on, higher in risk-off.
    tilt_exit_downgrade_min: float = Field(ge=0.0, le=1.0, default=0.05)
    # New: minimum R multiple required to allow MA-break downgrade EXIT -> TRIM
    # Computed as tp_pct_eff / sl_pct_eff (>=0). Default 1.0 (at least 1R reward/risk).
    exit_downgrade_min_r: float = Field(ge=0.0, default=0.0)
    # Session-aware deferral for MA-break exits: require phase >= this value
    # Accepted: 'morning', 'lunch', 'afternoon', 'atc', 'post'
    # Default keeps legacy behavior (no deferral) by allowing from 'morning'.
    exit_ma_break_min_phase: str = Field(default='morning')
    # Transaction cost integration: scale for gating thresholds.
    # Effective base_add/base_new = base + tc_gate_scale * pricing.tc_roundtrip_frac
    tc_gate_scale: float = Field(ge=0.0, le=2.0, default=0.0)
    # Quantile pool selection for add/new gating
    quantile_pool: Literal['subset', 'full'] = 'subset'
    # Combination rules for TP/SL when both static and ATR-based are provided
    tp_rule: Literal['min', 'max', 'dynamic_only', 'static_only'] = 'min'
    sl_rule: Literal['min', 'max', 'dynamic_only', 'static_only'] = 'min'
    # Stateless stop/TP parameters
    tp1_frac: float = Field(default=0.50, ge=0.0, le=1.0)
    tp1_hh_lookback: int = Field(default=10, ge=1)
    trail_hh_lookback: int = Field(default=22, ge=1)
    trail_atr_mult: float = Field(default=2.50, ge=0.0)
    be_buffer_pct: float = Field(default=0.0, ge=0.0)
    sl_trim_step_1_trigger: float = Field(default=0.50, ge=0.0, le=1.0)
    sl_trim_step_1_frac: float = Field(default=0.25, ge=0.0, le=1.0)
    sl_trim_step_2_trigger: float = Field(default=0.80, ge=0.0, le=1.0)
    sl_trim_step_2_frac: float = Field(default=0.35, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ranges(self):
        if not (0.0 <= float(self.base_add) <= 1.0):
            raise ValueError("base_add must be in 0..1")
        if not (0.0 <= float(self.base_new) <= 1.0):
            raise ValueError("base_new must be in 0..1")
        if not (-1.0 <= float(self.trim_th) <= 1.0):
            raise ValueError("trim_th must be in -1..1")
        if not (0.0 <= self.q_add <= 1.0):
            raise ValueError("q_add must be in 0..1")
        if not (0.0 <= self.q_new <= 1.0):
            raise ValueError("q_new must be in 0..1")
        if not (0.0 <= self.min_liq_norm <= 1.0):
            raise ValueError("min_liq_norm must be in 0..1")
        if not (0.0 < self.near_ceiling_pct <= 1.0):
            raise ValueError("near_ceiling_pct must be in (0..1]")
        for name in ("tp_pct", "sl_pct"):
            v = getattr(self, name)
            if float(v) < 0.0 or float(v) > 1.0:
                raise ValueError(f"thresholds.{name} must be in 0..1")
        for name in ("tp_floor_pct", "sl_floor_pct", "tp_cap_pct", "sl_cap_pct"):
            v = getattr(self, name)
            if v is not None and float(v) < 0.0:
                raise ValueError(f"thresholds.{name} must be >= 0")
        for name in ("tp_atr_mult", "sl_atr_mult"):
            v = getattr(self, name)
            if v is not None and float(v) < 0.0:
                raise ValueError(f"thresholds.{name} must be >= 0")
        if not (0.0 <= float(self.tp_trim_frac) <= 1.0):
            raise ValueError("thresholds.tp_trim_frac must be in 0..1")
        if int(self.cooldown_days) < 0:
            raise ValueError("thresholds.cooldown_days must be >= 0")
        # Validate session phase value if provided
        allowed_phases = {"morning", "lunch", "afternoon", "atc", "post"}
        if str(self.exit_ma_break_min_phase).strip().lower() not in allowed_phases:
            raise ValueError("thresholds.exit_ma_break_min_phase must be one of: morning, lunch, afternoon, atc, post")
        # Validate quantile_pool
        if str(self.quantile_pool).strip().lower() not in {"subset", "full"}:
            raise ValueError("thresholds.quantile_pool must be 'subset' or 'full'")
        # Additional RSI thresholds already range-checked by Field
        if not (0.0 <= float(self.tp1_frac) <= 1.0):
            raise ValueError("thresholds.tp1_frac must be in 0..1")
        if int(self.tp1_hh_lookback) < 1:
            raise ValueError("thresholds.tp1_hh_lookback must be >= 1")
        if int(self.trail_hh_lookback) < 1:
            raise ValueError("thresholds.trail_hh_lookback must be >= 1")
        if float(self.trail_atr_mult) < 0.0:
            raise ValueError("thresholds.trail_atr_mult must be >= 0")
        if float(self.be_buffer_pct) < 0.0:
            raise ValueError("thresholds.be_buffer_pct must be >= 0")
        if not (0.0 <= float(self.sl_trim_step_1_trigger) <= 1.0):
            raise ValueError("thresholds.sl_trim_step_1_trigger must be in 0..1")
        if not (0.0 <= float(self.sl_trim_step_2_trigger) <= 1.0):
            raise ValueError("thresholds.sl_trim_step_2_trigger must be in 0..1")
        if float(self.sl_trim_step_1_trigger) > float(self.sl_trim_step_2_trigger):
            raise ValueError("thresholds.sl_trim_step_1_trigger must be <= sl_trim_step_2_trigger")
        for name in ("sl_trim_step_1_frac", "sl_trim_step_2_frac"):
            if not (0.0 <= float(getattr(self, name)) <= 1.0):
                raise ValueError(f"thresholds.{name} must be in 0..1")
        if str(self.tp_sl_mode).strip().lower() not in {"legacy", "atr_per_ticker"}:
            raise ValueError("thresholds.tp_sl_mode must be 'legacy' or 'atr_per_ticker'")
        # Additional optional multi-target controls
        if self.tp1_atr_mult is not None and not (0.8 <= float(self.tp1_atr_mult) <= 1.3):
            raise ValueError("thresholds.tp1_atr_mult must be in 0.8..1.3")
        if self.tp2_atr_mult is not None and not (1.5 <= float(self.tp2_atr_mult) <= 2.5):
            raise ValueError("thresholds.tp2_atr_mult must be in 1.5..2.5")
        if self.trailing_atr_mult is not None and not (1.1 <= float(self.trailing_atr_mult) <= 2.0):
            raise ValueError("thresholds.trailing_atr_mult must be in 1.1..2.0")
        if self.trim_frac_tp1 is not None and not (0.2 <= float(self.trim_frac_tp1) <= 0.7):
            raise ValueError("thresholds.trim_frac_tp1 must be in 0.2..0.7")
        if self.trim_frac_tp2 is not None and not (0.2 <= float(self.trim_frac_tp2) <= 0.6):
            raise ValueError("thresholds.trim_frac_tp2 must be in 0.2..0.6")
        if self.time_stop_days is not None and not (3 <= int(self.time_stop_days) <= 7):
            raise ValueError("thresholds.time_stop_days must be between 3 and 7")
        if self.trim_rsi_gate is not None and not (0.0 <= float(self.trim_rsi_gate) <= 100.0):
            raise ValueError("thresholds.trim_rsi_gate must be in 0..100")
        if self.cooldown_days_after_exit is not None and not (1 <= int(self.cooldown_days_after_exit) <= 5):
            raise ValueError("thresholds.cooldown_days_after_exit must be between 1 and 5")
        if self.partial_entry_frac is not None and not (0.1 <= float(self.partial_entry_frac) <= 0.5):
            raise ValueError("thresholds.partial_entry_frac must be in 0.1..0.5")
        if not (0.02 <= float(self.new_partial_buffer) <= 0.08):
            raise ValueError("thresholds.new_partial_buffer must be in 0.02..0.08")
        return self


class ThresholdProfiles(BaseModel):
    risk_on: Thresholds
    risk_off: Thresholds


class NeutralAdaptive(BaseModel):
    # Master enable for neutral adaptations
    neutral_enable: Union[int, bool] = 1
    # Risk-on probability band to qualify as neutral (inclusive)
    neutral_risk_on_prob_low: float = Field(ge=0.0, le=1.0, default=0.35)
    neutral_risk_on_prob_high: float = Field(ge=0.0, le=1.0, default=0.65)
    # Maximum acceptable index ATR percentile/level to remain neutral
    neutral_index_atr_soft_cap: Optional[float] = Field(default=None, ge=0.0)
    # Breadth tolerance (half-width) around neutral centre (default 0.5)
    neutral_breadth_band: float = Field(ge=0.0, le=1.0, default=0.05)
    # Optional explicit breadth centre (defaults to 0.5 when None)
    neutral_breadth_center: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Scaling applied to thresholds (bounded by per-field floors)
    neutral_base_new_scale: float = Field(gt=0.0, le=1.0, default=0.90)
    neutral_base_new_floor: float = Field(ge=0.0, le=1.0, default=0.80)
    neutral_base_add_scale: float = Field(gt=0.0, le=1.0, default=0.90)
    neutral_base_add_floor: float = Field(ge=0.0, le=1.0, default=0.80)
    # Partial entry configuration
    partial_threshold_ratio: float = Field(ge=0.0, le=1.0, default=0.75)
    partial_entry_frac: float = Field(gt=0.0, le=1.0, default=0.30)
    partial_allow_leftover: Union[int, bool] = 0
    # Minimum number of NEW orders ensured via override per neutral day
    min_new_per_day: int = Field(ge=0, default=1)
    max_new_overrides_per_day: int = Field(ge=0, default=1)
    # Additional cap for ADD orders when neutral
    add_max_neutral_cap: int = Field(ge=0, default=1)

    @model_validator(mode="after")
    def _validate_ranges(self):
        if float(self.neutral_risk_on_prob_low) > float(self.neutral_risk_on_prob_high):
            raise ValueError("neutral_risk_on_prob_low must be <= neutral_risk_on_prob_high")
        if self.neutral_index_atr_soft_cap is not None and float(self.neutral_index_atr_soft_cap) < 0.0:
            raise ValueError("neutral_index_atr_soft_cap must be >= 0 when provided")
        if self.min_new_per_day > 0 and self.max_new_overrides_per_day == 0:
            raise ValueError("max_new_overrides_per_day must be >0 when min_new_per_day >0")
        return self


class Pricing(BaseModel):
    risk_on_buy: List[str]
    risk_on_sell: List[str]
    risk_off_buy: List[str]
    risk_off_sell: List[str]
    atr_fallback_buy_mult: float
    atr_fallback_sell_mult: float
    # Optional: transaction cost (round-trip) fraction for ExpR calculations
    tc_roundtrip_frac: float = 0.0
    # Optional: sell transaction tax fraction (applied to SELL proceeds)
    tc_sell_tax_frac: float = Field(ge=0.0, le=0.05, default=0.001)
    # Optional: fill probability heuristics (execution UX)
    class FillProb(BaseModel):
        base: float = Field(ge=0.0, le=1.0, default=0.30)
        cross: float = Field(ge=0.0, le=1.0, default=0.90)
        near_ceiling: float = Field(ge=0.0, le=1.0, default=0.05)
        min: float = Field(ge=0.0, le=1.0, default=0.05)
        decay_scale_min_ticks: float = Field(gt=0.0, default=5.0)
        partial_fill_kappa: float = Field(ge=0.0, le=1.0, default=0.65)
        min_fill_notional_vnd: float = Field(ge=0.0, default=5_000_000.0)

    fill_prob: Optional[FillProb] = None

    class SlippageModel(BaseModel):
        alpha_bps: float = Field(default=5.0)
        beta_dist_per_tick: float = Field(default=1.0)
        beta_size: float = Field(default=50.0)
        beta_vol: float = Field(default=10.0)
        mae_bps: float = Field(default=10.0, ge=5.0, le=40.0)
        last_fit_date: Optional[str] = None

    slippage_model: Optional[SlippageModel] = None


class Execution(BaseModel):
    class FillConfig(BaseModel):
        horizon_s: int = Field(default=60, ge=10, le=600)
        window_sigma_s: int = Field(default=45, ge=15, le=600)
        window_vol_s: int = Field(default=90, ge=30, le=600)
        target_prob: float = Field(default=0.60, ge=0.0, le=0.99)
        max_chase_ticks: int = Field(default=1, ge=0, le=5)
        cancel_ratio_per_min: float = Field(default=0.30, ge=0.0, le=0.95)
        joiner_factor: float = Field(default=0.05, ge=0.0, le=1.0)
        no_cross: bool = Field(default=True)

    stop_ttl_min: int = Field(default=3, ge=1)
    slip_pct_min: float = Field(default=0.002, ge=0.0)
    slip_atr_mult: float = Field(default=0.50, ge=0.0)
    slip_ticks_min: int = Field(default=1, ge=0)
    flash_k_atr: float = Field(default=1.50, ge=0.0)
    fill: Optional[FillConfig] = None

    # New: controls for order price crossing behavior at engine stage
    # Keep legacy behavior by default (filter BUY when limit>market in engine).
    # When set to 0/False, engine will clamp BUY limit down to market instead of filtering,
    # letting the order pass to IO where session-aware clamping is already applied.
    filter_buy_limit_gt_market: Union[int, bool] = 1

    class Ladder(BaseModel):
        enabled: bool = True
        max_levels: int = Field(default=3, ge=1, le=5)
        lot_size: int = Field(default=100, ge=1)
        method: Literal['atr_pct', 'ticks'] = 'atr_pct'
        buy_offsets_atr: Dict[str, List[float]] = Field(default_factory=lambda: {
            'risk_on': [0.00, 0.25, 0.50],
            'neutral': [0.00, 0.35, 0.70],
            'risk_off': [0.10, 0.50, 1.00],
        })
        sell_offsets_atr: Dict[str, List[float]] = Field(default_factory=lambda: {
            'risk_on': [0.30, 0.60, 0.90],
            'neutral': [0.50, 1.00, 1.50],
            'risk_off': [0.70, 1.20, 1.80],
        })
        weights: Dict[str, List[float]] = Field(default_factory=lambda: {
            'risk_on': [0.60, 0.30, 0.10],
            'neutral': [0.45, 0.35, 0.20],
            'risk_off': [0.25, 0.35, 0.40],
        })
        ttl_min: List[int] = Field(default_factory=lambda: [5, 15, 30])
        reprice_ticks: Dict[str, int] = Field(default_factory=lambda: {
            'risk_on': 2, 'neutral': 1, 'risk_off': 0,
        })
        min_fill_prob: float = Field(default=0.20, ge=0.0, le=1.0)
        adtv_cap_frac: float = Field(default=0.03, ge=0.0, le=0.20)
        atc_guard_minutes: int = Field(default=5, ge=0)
        skip_if_limit_lock: bool = True
        no_cross: bool = True

        @model_validator(mode="after")
        def _validate_shapes(self):
            for key in ('risk_on','neutral','risk_off'):
                if key not in self.buy_offsets_atr or key not in self.sell_offsets_atr or key not in self.weights:
                    raise ValueError(f"ladder requires offsets/weights for profile '{key}'")
            return self

    ladder: Optional[Ladder] = None

    class TimeOfDay(BaseModel):
        class PhaseRule(BaseModel):
            allow_cross: Optional[bool] = None
            max_tranches: Optional[int] = None
            exit_ma_break_min_phase: Optional[str] = None
            allow_cross_if_target_prob_gte: Optional[float] = None

        phase_rules: Dict[str, PhaseRule] = Field(default_factory=dict)

    time_of_day: Optional[TimeOfDay] = None


class MarketFilter(BaseModel):
    # Keys calibrated by engine can be omitted in config (set at runtime):
    risk_off_index_drop_pct: Optional[float] = Field(default=None, gt=0.0)
    risk_off_trend_floor: float
    risk_off_breadth_floor: float = Field(ge=0.0, le=1.0)
    breadth_relax_margin: Optional[float] = Field(default=None, ge=0.0, le=0.2)
    # Guard behaviour when market weak:
    # - 'pause' (default): pause NEW and defer ADD, allow limited leader-bypass
    # - 'scale_only': do not filter candidates; only scale BUY budget via caps
    guard_behavior: Literal['pause', 'scale_only'] = 'pause'
    market_score_soft_floor: float = Field(ge=0.0, le=1.0)
    market_score_hard_floor: float = Field(ge=0.0, le=1.0)
    leader_min_rsi: float
    leader_min_mom_norm: float = Field(ge=0.0, le=1.0)
    leader_require_ma20: Union[int, bool]
    leader_require_ma50: Union[int, bool]
    leader_max: int = Field(ge=0, default=0)
    risk_off_drawdown_floor: float = Field(ge=0.0, le=1.0, default=0.0)
    # ATR percentile thresholds (calibrated): optional in config
    index_atr_soft_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    index_atr_hard_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Optional: foreign/global guards (disabled if null)
    # When US EPU percentile >= soft, cap buy budget; when >= hard, freeze new buys.
    us_epu_soft_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    us_epu_hard_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # When DXY percentile >= thresholds, cap or freeze buys
    dxy_soft_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dxy_hard_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # When S&P 500 max drawdown (rolling) >= this fraction (e.g. 0.25), freeze new buys.
    spx_drawdown_hard_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Budget scaling caps under guards (were hard-coded in engine)
    # When guard_new active but not severe, cap budget by this fraction
    guard_new_scale_cap: float = Field(ge=0.0, le=1.0, default=0.40)
    # When ATR percentile >= soft threshold (but < hard), cap budget by this fraction
    atr_soft_scale_cap: float = Field(ge=0.0, le=1.0, default=0.50)
    # Multiplier for severe index drop relative to risk_off_index_drop_pct
    severe_drop_mult: float = Field(gt=0.0, default=1.50)
    # Optional: when leader bypass finds no leaders but guard_new is active and
    # bypass is allowed (not severe), keep top-K NEW by score anyway (K=0 disables).
    leader_fallback_topk_if_empty: int = Field(ge=0, default=0)
    # Hard guards (calibrated): optional in config
    idx_chg_smoothed_hard_drop: Optional[float] = Field(default=None, ge=0.0)  # percent magnitude
    trend_norm_hard_floor: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    vol_ann_hard_ceiling: Optional[float] = Field(default=None, gt=0.0)

    class Events(BaseModel):
        enable: Union[int, bool] = 0
        no_new_on_event: Union[int, bool] = 1
        t_minus: int = Field(default=1, ge=0)
        t_plus: int = Field(default=1, ge=0)

    events: Optional[Events] = None

    @model_validator(mode="after")
    def _check_values(self):
        if self.leader_min_rsi < 0.0 or self.leader_min_rsi > 100.0:
            raise ValueError("market_filter.leader_min_rsi must be in 0..100")
        if self.market_score_soft_floor < self.market_score_hard_floor:
            raise ValueError("market_filter.market_score_soft_floor must be >= market_score_hard_floor")
        if self.index_atr_soft_pct is not None and self.index_atr_hard_pct is not None:
            if self.index_atr_soft_pct > self.index_atr_hard_pct:
                raise ValueError("market_filter.index_atr_soft_pct must be <= index_atr_hard_pct")
        return self


class GlobalTilts(BaseModel):
    # Optional dynamic sector tilts from global factors (disabled by default)
    brent_energy_enable: Union[int, bool] = 0
    # Apply when Brent 63d momentum >= this value (e.g., 0.05 = +5%)
    brent_mom_soft: Optional[float] = Field(default=None)
    # Minimum sector rank boost applied to the configured sector when condition holds
    brent_boost_min: float = Field(default=0.20, ge=0.0, le=1.0)
    # Sector label in your data (Vietnamese), e.g., 'Năng lượng'
    brent_energy_sector_label: str = Field(default='Năng lượng')


class MarketMicrostructure(BaseModel):
    # Daily band (±) as a fraction (e.g., 0.07 for ±7%)
    daily_band_pct: float = Field(ge=0.0, le=1.0, default=0.07)


class MarketConfig(BaseModel):
    microstructure: MarketMicrostructure = Field(default_factory=MarketMicrostructure)


class RegimeComponent(BaseModel):
    mean: float
    std: float = Field(gt=0.0)
    weight: float


class RegimeModel(BaseModel):
    components: Dict[str, RegimeComponent]
    intercept: float
    threshold: float = Field(ge=0.0, le=1.0)


class DynamicCaps(BaseModel):
    enable: Union[int, bool]
    pos_min: float
    pos_max: float
    sector_min: float
    sector_max: float
    blend: float
    override_static: Union[int, bool]

    @model_validator(mode="after")
    def _check_ranges(self):
        for name in ("pos_min", "pos_max", "sector_min", "sector_max", "blend"):
            v = getattr(self, name)
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"dynamic_caps.{name} must be in 0..1")
        return self


class MeanVarianceCalibration(BaseModel):
    enable: Union[int, bool] = 0
    risk_alpha: List[float]
    cov_reg: List[float]
    bl_alpha_scale: List[float]
    lookback_days: int = 250
    test_horizon_days: int = 60
    min_history_days: int = 320

    @model_validator(mode="after")
    def _check_lists(self):
        if not self.risk_alpha:
            raise ValueError("mean_variance_calibration.risk_alpha must be non-empty")
        if not self.cov_reg:
            raise ValueError("mean_variance_calibration.cov_reg must be non-empty")
        if not self.bl_alpha_scale:
            raise ValueError("mean_variance_calibration.bl_alpha_scale must be non-empty")
        if any(float(x) <= 0 for x in self.risk_alpha):
            raise ValueError("mean_variance_calibration.risk_alpha must contain positives")
        if any(float(x) < 0 for x in self.cov_reg):
            raise ValueError("mean_variance_calibration.cov_reg must be >= 0")
        if any(float(x) < 0 for x in self.bl_alpha_scale):
            raise ValueError("mean_variance_calibration.bl_alpha_scale must be >= 0")
        if self.lookback_days < 60:
            raise ValueError("mean_variance_calibration.lookback_days must be >= 60")
        if self.test_horizon_days < 10:
            raise ValueError("mean_variance_calibration.test_horizon_days must be >= 10")
        if self.min_history_days < self.lookback_days + self.test_horizon_days:
            raise ValueError("mean_variance_calibration.min_history_days too small for requested windows")
        return self


class Sizing(BaseModel):
    softmax_tau: float
    add_share: float
    new_share: float
    min_lot: int
    risk_weighting: Literal["score_softmax", "inverse_atr", "hybrid", "risk_parity", "inverse_sigma"]
    risk_alpha: float = Field(ge=0.0)
    max_pos_frac: float
    max_sector_frac: float
    reuse_sell_proceeds_frac: float
    risk_blend: float
    leftover_redistribute: Union[int, bool]
    min_ticket_k: float
    cov_lookback_days: int
    cov_reg: float
    risk_parity_floor: float
    dynamic_caps: DynamicCaps
    allocation_model: Literal["softmax", "risk_budget", "mean_variance"] = "softmax"
    bl_rf_annual: float = Field(default=0.03)
    bl_mkt_prem_annual: float = Field(default=0.07)
    bl_alpha_scale: float = Field(default=0.02)
    risk_blend_eta: float = Field(default=0.0)
    min_names_target: int = Field(default=0, ge=0)
    market_index_symbol: str = Field(default='VNINDEX')
    # Risk-per-trade sizing (optional; 0 disables)
    risk_per_trade_frac: float = 0.0
    default_stop_atr_mult: float = 0.0
    tranche_frac: float = Field(default=0.25, ge=0.0, le=1.0)
    qty_min_lot: int = Field(default=100, ge=100)
    min_notional_per_order: float = Field(default=2_000_000.0, ge=1_000_000.0)
    # Max lots for the very first entry of a NEW position (non-partial). Default 1.
    new_first_tranche_lots: int = Field(default=1, ge=1)
    mean_variance_calibration: Optional[MeanVarianceCalibration] = None

    @model_validator(mode="after")
    def _check_ranges(self):
        for name in ("add_share", "new_share", "max_pos_frac", "max_sector_frac", "reuse_sell_proceeds_frac", "risk_blend", "risk_parity_floor"):
            v = getattr(self, name)
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"sizing.{name} must be in 0..1")
        if self.cov_lookback_days < 20:
            raise ValueError("sizing.cov_lookback_days must be >= 20")
        if self.cov_reg < 0.0:
            raise ValueError("sizing.cov_reg must be >= 0")
        if not (-0.10 <= float(self.bl_rf_annual) <= 0.25):
            raise ValueError("sizing.bl_rf_annual must be between -0.10 and 0.25")
        if not (-0.10 <= float(self.bl_mkt_prem_annual) <= 0.40):
            raise ValueError("sizing.bl_mkt_prem_annual must be between -0.10 and 0.40")
        if not (0.0 <= float(self.bl_alpha_scale) <= 0.10):
            raise ValueError("sizing.bl_alpha_scale must be in 0..0.10")
        if not (0.0 <= float(self.risk_blend_eta) <= 1.0):
            raise ValueError("sizing.risk_blend_eta must be in 0..1")
        if not (0.0 <= float(self.risk_per_trade_frac) <= 1.0):
            raise ValueError("sizing.risk_per_trade_frac must be in 0..1")
        if float(self.default_stop_atr_mult) < 0.0:
            raise ValueError("sizing.default_stop_atr_mult must be >= 0")
        if not (0.0 <= float(self.tranche_frac) <= 1.0):
            raise ValueError("sizing.tranche_frac must be in 0..1")
        if int(self.qty_min_lot) < 100:
            raise ValueError("sizing.qty_min_lot must be >= 100")
        if float(self.min_notional_per_order) < 1_000_000.0:
            raise ValueError("sizing.min_notional_per_order must be >= 1_000_000 VND")
        if not str(self.market_index_symbol).strip():
            raise ValueError("sizing.market_index_symbol must be non-empty")
        return self


class TickerOverride(BaseModel):
    # Optional per-ticker overrides for action thresholds
    base_add: Optional[float] = None
    base_new: Optional[float] = None
    trim_th: Optional[float] = None
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    tp_atr_mult: Optional[float] = None
    sl_atr_mult: Optional[float] = None
    tp_floor_pct: Optional[float] = None
    tp_cap_pct: Optional[float] = None
    sl_floor_pct: Optional[float] = None
    sl_cap_pct: Optional[float] = None
    tp_trim_frac: Optional[float] = None
    exit_on_ma_break: Optional[Union[int, bool]] = None
    exit_ma_break_rsi: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    trim_rsi_below_ma20: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    trim_rsi_macdh_neg: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    tp_sl_mode: Optional[Literal['legacy', 'atr_per_ticker']] = None
    tp1_atr_mult: Optional[float] = Field(default=None, ge=0.8, le=1.3)
    tp2_atr_mult: Optional[float] = Field(default=None, ge=1.5, le=2.5)
    trailing_atr_mult: Optional[float] = Field(default=None, ge=1.1, le=2.0)
    trim_frac_tp1: Optional[float] = Field(default=None, ge=0.2, le=0.7)
    trim_frac_tp2: Optional[float] = Field(default=None, ge=0.2, le=0.6)
    breakeven_after_tp1: Optional[Union[int, bool]] = None
    time_stop_days: Optional[int] = Field(default=None, ge=0)
    trim_rsi_gate: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    cooldown_days_after_exit: Optional[int] = Field(default=None, ge=0)
    partial_entry_enabled: Optional[Union[int, bool]] = None
    partial_entry_frac: Optional[float] = Field(default=None, ge=0.1, le=0.5)
    partial_entry_floor_lot: Optional[int] = Field(default=None, ge=1)
    new_partial_buffer: Optional[float] = Field(default=None, ge=0.02, le=0.08)
    mae_bps: Optional[float] = Field(default=None, ge=5.0, le=40.0)
    # Optional per-ticker sizing overrides
    risk_per_trade_frac: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    default_stop_atr_mult: Optional[float] = Field(default=None, ge=0.0)
    max_pos_frac: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Optional per-ticker override for R multiple threshold
    exit_downgrade_min_r: Optional[float] = Field(default=None, ge=0.0)
    sl_trim_step_1_trigger: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sl_trim_step_1_frac: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sl_trim_step_2_trigger: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sl_trim_step_2_frac: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class OrdersUI(BaseModel):
    class TTLMinutes(BaseModel):
        base: int = Field(ge=1, default=12)
        soft: int = Field(ge=1, default=7)
        hard: int = Field(ge=1, default=5)

    class Watchlist(BaseModel):
        enable: Union[int, bool] = 1
        min_priority: float = Field(ge=0.0, le=1.0, default=0.25)
        micro_window: int = Field(ge=1, default=3)

    ttl_minutes: TTLMinutes = Field(default_factory=TTLMinutes)
    watchlist: Watchlist = Field(default_factory=Watchlist)
    # Number of tickers to show for suggestions (UI only)
    suggestions_top_n: int = Field(ge=1, default=3)
    # When enabled, write pre-sized BUY candidates (before safety/market/budget filters)
    # into orders_filtered.csv with Reason='candidate_pre'. Default disabled to keep legacy outputs stable.
    write_pre_candidates: Union[int, bool] = 0
    # Pre-size mode for candidate_pre sizing:
    # - 'budget_softmax': split daily budget by score (legacy behavior)
    # - 'score_min_lot': map min score to 1 lot, others proportionally in lots
    pre_size_mode: Literal['budget_softmax','score_min_lot'] = 'budget_softmax'


class Evaluation(BaseModel):
    # Fraction of ADTV used per day when estimating DaysToLiq
    days_to_liq_frac: float = Field(gt=0.0, le=1.0, default=0.20)


class FeatureFlags(BaseModel):
    # Enable robust normalization (MAD-based with tighter z clipping) for feature z-scores
    # 0 disables (use mean/std with z_clip=8.0). Default disabled for backward compatibility.
    normalization_robust: Union[int, bool] = 0


class CalibrationFlags(BaseModel):
    # Enable one-shot calibrations during each engine run (config-driven, no env).
    # Set on_run=1 and toggle specific modules below to 1 to execute.
    on_run: Union[int, bool] = 0
    # Specific calibrators (0=off, 1=on)
    regime_components: Union[int, bool] = 0
    regime: Union[int, bool] = 0
    market_filter: Union[int, bool] = 0
    breadth_floor: Union[int, bool] = 0
    leader_gates: Union[int, bool] = 0
    liquidity: Union[int, bool] = 0
    sizing: Union[int, bool] = 0
    softmax_tau: Union[int, bool] = 0
    thresholds_topk: Union[int, bool] = 0
    risk_limits: Union[int, bool] = 0
    dynamic_caps: Union[int, bool] = 0


class PolicyOverrides(BaseModel):
    buy_budget_frac: float
    buy_budget_by_regime: Optional[Dict[str, float]] = None
    add_max: int
    new_max: int
    weights: Weights
    thresholds: Thresholds
    thresholds_profiles: Optional[ThresholdProfiles] = None
    neutral_adaptive: NeutralAdaptive = Field(default_factory=NeutralAdaptive)
    # Optional global bias (leaning buy/observe) in [-0.2..0.2].
    market_bias: Optional[float] = 0.0
    sector_bias: Dict[str, float]
    ticker_bias: Dict[str, float]
    pricing: Pricing
    execution: Execution = Field(default_factory=Execution)
    sizing: Sizing
    market_filter: MarketFilter
    # Optional UI and market config
    orders_ui: Optional[OrdersUI] = None
    # Optional risk overlays (vol targeting / kill-switch)
    class Risk(BaseModel):
        class VolTarget(BaseModel):
            enable: Union[int, bool] = 0
            target_ann: float = Field(default=0.15, gt=0.0)
            lookback_days: int = Field(default=20, ge=1)
            scale_min: float = Field(default=0.6, ge=0.0)
            scale_max: float = Field(default=1.4, ge=0.0)

        class KillSwitch(BaseModel):
            enable: Union[int, bool] = 0
            dd_hard_pct: float = Field(default=0.12, ge=0.0)
            sl_streak_n: int = Field(default=5, ge=0)
            window_days: int = Field(default=3, ge=1)
            cooldown_days: int = Field(default=3, ge=0)
            actions: Dict[str, float] = Field(default_factory=lambda: {"buy_budget_frac": 0.0, "add_budget_frac_scale": 0.33})

        vol_target: VolTarget = Field(default_factory=VolTarget)
        kill_switch: KillSwitch = Field(default_factory=KillSwitch)

    risk: Optional[Risk] = None
    market: Optional[MarketConfig] = None
    regime_model: RegimeModel
    # Optional: scales used to normalize regime components (avoid hidden constants)
    class RegimeScales(BaseModel):
        vol_ann_unit: float = Field(gt=0.0, default=0.45)
        trend_unit: float = Field(gt=0.0, default=0.05)
        idx_smoothed_unit: float = Field(gt=0.0, default=1.5)
        drawdown_unit: float = Field(gt=0.0, default=0.20)
        momentum_unit: float = Field(gt=0.0, default=0.12)

    regime_scales: Optional[RegimeScales] = None
    # Optional per-ticker overrides (empty map allowed)
    ticker_overrides: Dict[str, TickerOverride] = Field(default_factory=dict)
    # Optional calibration targets to guide formula-based calibration steps
    calibration_targets: Optional[Dict[str, Dict[str, float]]] = None
    # Optional global tilts (e.g., Brent -> Energy sector boost)
    global_tilts: Optional[GlobalTilts] = None
    # Optional evaluation knobs (non-trading policy)
    evaluation: Optional[Evaluation] = None
    # Optional feature engineering flags (non-structural, affects normalization only)
    features: Optional[FeatureFlags] = None
    # Optional auto-calibration flags (config-driven; prefer over env toggles)
    calibration: Optional[CalibrationFlags] = None

    @model_validator(mode="after")
    def _check_top(self):
        if not (0.0 <= float(self.buy_budget_frac) <= 0.30):
            raise ValueError("buy_budget_frac out of range 0..0.30")
        if self.buy_budget_by_regime is not None:
            for key, value in self.buy_budget_by_regime.items():
                if key not in {"risk_on", "neutral", "risk_off"}:
                    raise ValueError("buy_budget_by_regime keys must be 'risk_on', 'neutral', or 'risk_off'")
                if not (0.0 <= float(value) <= 0.30):
                    raise ValueError("buy_budget_by_regime values must be in 0..0.30")
        for name in ("add_max", "new_max"):
            v = getattr(self, name)
            if not (0 <= int(v) <= 50):
                raise ValueError(f"{name} out of range 0..50")
        return self
