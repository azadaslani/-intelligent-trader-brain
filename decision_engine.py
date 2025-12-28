from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional


class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MarketRegime(str, Enum):
    TREND = "trend"
    RANGE = "range"
    UNKNOWN = "unknown"


class TrendDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class ModelPrediction:
    source: str
    direction: float
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not -1.0 <= self.direction <= 1.0:
            raise ValueError(f"direction must be in [-1, 1], got {self.direction}")


@dataclass(frozen=True)
class TimeframeContext:
    timeframe: str
    trend: TrendDirection
    regime: MarketRegime
    volatility: float


@dataclass(frozen=True)
class TradeSignal:
    action: TradeAction
    confidence: float
    regime: MarketRegime
    direction_score: float
    reasons: List[str]
    rationale: str
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeAssessment:
    regime: MarketRegime
    trend: TrendDirection
    alignment: float
    avg_volatility: float
    notes: str


@dataclass(frozen=True)
class FusionResult:
    direction: float
    confidence: float
    agreement: float
    weights: Dict[str, float]
    conflict: bool
    confidence_gap: float


@dataclass(frozen=True)
class SignatureHit:
    name: str
    bias: float
    strength: float
    confidence: float
    notes: str = ""

    def __post_init__(self) -> None:
        if not -1.0 <= self.bias <= 1.0:
            raise ValueError(f"bias must be in [-1, 1], got {self.bias}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class SignatureContext:
    hits: List[SignatureHit]
    net_bias: float
    net_strength: float
    veto: bool
    boost: float
    reasons: List[str]


@dataclass(frozen=True)
class MemoryStats:
    n: int
    wins: int
    losses: int
    avg_r: float
    last_update: int


@dataclass(frozen=True)
class MemoryDecision:
    key: str
    stats: Optional[MemoryStats]
    adjustment: float
    veto: bool
    reasons: List[str]


class MemoryLayer:
    def __init__(self, bank: Dict[str, Dict[str, float]]) -> None:
        self._bank = bank

    def build_key(
        self,
        symbol: str,
        timeframe: str,
        reasons: List[str],
        regime: MarketRegime,
        vol_bin: int,
        align_bin: int,
    ) -> str:
        reasons_token = "-".join(sorted(reasons))
        return f"{symbol}:{timeframe}:{regime.value}:{vol_bin}:{align_bin}:{reasons_token}"

    def evaluate(self, key: str) -> MemoryDecision:
        stats_payload = self._bank.get("keys", {}).get(key)
        stats = self._parse_stats(stats_payload) if stats_payload else None
        if not stats or stats.n < 15:
            return MemoryDecision(
                key=key,
                stats=stats,
                adjustment=0.0,
                veto=False,
                reasons=["MEM_INSUFFICIENT_DATA"],
            )

        win_rate = stats.wins / stats.n
        adjustment = self._clamp((win_rate - 0.5) * 0.20, -0.08, 0.08)
        veto = stats.n >= 25 and win_rate <= 0.35 and stats.avg_r < 0
        reasons = [
            f"MEM_N_{stats.n}",
            f"MEM_WR_{win_rate:.2f}",
            f"MEM_ADJ_{adjustment:.2f}",
        ]
        if veto:
            reasons.append("MEM_VETO")

        return MemoryDecision(
            key=key,
            stats=stats,
            adjustment=adjustment,
            veto=veto,
            reasons=reasons,
        )

    @staticmethod
    def _parse_stats(payload: Dict[str, float]) -> MemoryStats:
        return MemoryStats(
            n=int(payload.get("n", 0)),
            wins=int(payload.get("wins", 0)),
            losses=int(payload.get("losses", 0)),
            avg_r=float(payload.get("avg_r", 0.0)),
            last_update=int(payload.get("last_update", 0)),
        )

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(value, high))


class SignatureLayer:
    def __init__(self, bank: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        self._bank = bank or {}

    def evaluate(
        self,
        hits: List[SignatureHit],
        regime: RegimeAssessment,
        fusion: FusionResult,
    ) -> SignatureContext:
        del regime
        weighted_hits: List[tuple[SignatureHit, float]] = []
        veto_enabled = False
        for hit in hits:
            rule = self._bank.get(hit.name.upper())
            if rule:
                weight = float(rule.get("weight", 0.5))
                min_strength = float(rule.get("min_strength", 0.0))
                if hit.strength < min_strength:
                    continue
                veto_enabled = veto_enabled or bool(rule.get("veto_conflict", False))
            else:
                weight = 0.5
            effective_weight = weight * hit.strength * hit.confidence
            weighted_hits.append((hit, effective_weight))

        weights = [entry[1] for entry in weighted_hits]
        total_weight = sum(weights)
        if total_weight > 0:
            net_bias = sum(
                hit.bias * weight for hit, weight in weighted_hits
            ) / (
                total_weight
            )
        else:
            net_bias = 0.0
        net_strength = max(0.0, min(sum(weights) / max(1, len(weighted_hits)), 1.0))
        boost = min(0.20, 0.10 * net_strength)
        veto = (
            veto_enabled
            and net_strength >= 0.65
            and self._sign(net_bias) != 0
            and (
                self._sign(net_bias) != self._sign(fusion.direction)
            )
        )
        reasons = [f"SIG_HIT_{hit.name.upper()}" for hit, _ in weighted_hits]
        reasons.append(f"SIG_NET_{self._bias_label(net_bias)}")
        reasons.append(f"SIG_STRENGTH_{net_strength:.2f}")

        return SignatureContext(
            hits=[hit for hit, _ in weighted_hits],
            net_bias=net_bias,
            net_strength=net_strength,
            veto=veto,
            boost=boost,
            reasons=reasons,
        )

    @staticmethod
    def _sign(value: float) -> int:
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    @staticmethod
    def _bias_label(net_bias: float) -> str:
        if net_bias > 0.1:
            return "bull"
        if net_bias < -0.1:
            return "bear"
        return "neutral"


def load_signature_bank(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Signature bank must be a dictionary of signature rules")
    return {str(key).upper(): dict(value) for key, value in data.items()}


class DecisionEngine:
    def __init__(
        self,
        min_confidence: float = 0.6,
        direction_threshold: float = 0.15,
        regime_alignment_floor: float = 0.45,
        signature_layer: Optional[SignatureLayer] = None,
    ) -> None:
        if not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must be within (0, 1]")
        if direction_threshold <= 0.0:
            raise ValueError("direction_threshold must be positive")
        if not 0.0 <= regime_alignment_floor <= 1.0:
            raise ValueError("regime_alignment_floor must be within [0, 1]")

        self._min_confidence = min_confidence
        self._direction_threshold = direction_threshold
        self._regime_alignment_floor = regime_alignment_floor
        self._signature_layer = signature_layer or SignatureLayer()

    def generate_signal(
        self,
        lstm_prediction: ModelPrediction,
        xgb_prediction: ModelPrediction,
        context: Dict[str, TimeframeContext],
        symbol: str,
        primary_timeframe: str,
        signature_hits: Optional[List[SignatureHit]] = None,
        memory_layer: Optional[MemoryLayer] = None,
    ) -> TradeSignal:
        predictions = [lstm_prediction, xgb_prediction]
        regime = self._assess_regime(list(context.values()))
        fusion = self._fuse_predictions(predictions)
        adjusted_confidence = fusion.confidence * regime.alignment
        min_confidence, direction_threshold = self._regime_thresholds(
            regime, self._min_confidence, self._direction_threshold
        )

        if fusion.conflict and fusion.confidence_gap < 0.25:
            reasons = self._build_reasons(
                action=TradeAction.HOLD,
                regime=regime,
                alignment=regime.alignment,
                fusion=fusion,
                min_confidence=min_confidence,
                direction_threshold=direction_threshold,
            )
            return TradeSignal(
                action=TradeAction.HOLD,
                confidence=adjusted_confidence,
                regime=regime.regime,
                direction_score=fusion.direction,
                reasons=reasons,
                rationale=f"Conflict between models. Regime: {regime.notes}.",
                metadata={
                    "alignment": regime.alignment,
                    "raw_confidence": fusion.confidence,
                    "conflict": float(fusion.conflict),
                    **self._empty_memory_metadata(),
                    **self._flatten_weights(fusion.weights),
                },
            )

        base_floor = self._regime_alignment_floor
        if fusion.agreement >= 0.75 and fusion.confidence >= 0.75:
            base_floor = max(0.35, base_floor - 0.10)
        if regime.trend is TrendDirection.NEUTRAL:
            base_floor = max(0.35, base_floor - 0.05)

        if regime.alignment < base_floor:
            return TradeSignal(
                action=TradeAction.HOLD,
                confidence=adjusted_confidence,
                regime=regime.regime,
                direction_score=fusion.direction,
                reasons=self._build_reasons(
                    action=TradeAction.HOLD,
                    regime=regime,
                    alignment=regime.alignment,
                    fusion=fusion,
                    min_confidence=min_confidence,
                )
                + ["REGIME_MISALIGN"],
                rationale=f"Regime misalignment: {regime.notes}",
                metadata={
                    "alignment": regime.alignment,
                    "raw_confidence": fusion.confidence,
                    "conflict": float(fusion.conflict),
                    **self._empty_memory_metadata(),
                    **self._flatten_weights(fusion.weights),
                },
            )

        signature = self._signature_layer.evaluate(signature_hits or [], regime, fusion)
        adjusted_direction = (
            fusion.direction + 0.30 * signature.net_bias * signature.net_strength
        )
        adjusted_confidence = min(1.0, adjusted_confidence * (1 + signature.boost))

        if signature.veto:
            return TradeSignal(
                action=TradeAction.HOLD,
                confidence=adjusted_confidence,
                regime=regime.regime,
                direction_score=adjusted_direction,
                reasons=self._build_reasons(
                    action=TradeAction.HOLD,
                    regime=regime,
                    alignment=regime.alignment,
                    fusion=fusion,
                    min_confidence=min_confidence,
                    direction_threshold=direction_threshold,
                )
                + signature.reasons
                + ["SIG_VETO"],
                rationale="Signature vetoed trade due to structural conflict.",
                metadata={
                    "alignment": regime.alignment,
                    "raw_confidence": fusion.confidence,
                    "conflict": float(fusion.conflict),
                    "sig_net_bias": signature.net_bias,
                    "sig_net_strength": signature.net_strength,
                    "sig_boost": signature.boost,
                    "sig_veto": float(signature.veto),
                    **self._empty_memory_metadata(),
                    **self._flatten_weights(fusion.weights),
                },
            )

        mem_layer = memory_layer or MemoryLayer({"keys": {}})
        memory_key = mem_layer.build_key(
            symbol=symbol,
            timeframe=primary_timeframe,
            reasons=signature.reasons,
            regime=regime.regime,
            vol_bin=self._bucket(regime.avg_volatility, bins=5),
            align_bin=self._bucket(regime.alignment, bins=5),
        )
        memory = mem_layer.evaluate(memory_key)
        adjusted_confidence = min(1.0, adjusted_confidence * (1 + memory.adjustment))

        if memory.veto:
            return TradeSignal(
                action=TradeAction.HOLD,
                confidence=adjusted_confidence,
                regime=regime.regime,
                direction_score=adjusted_direction,
                reasons=self._build_reasons(
                    action=TradeAction.HOLD,
                    regime=regime,
                    alignment=regime.alignment,
                    fusion=fusion,
                    min_confidence=min_confidence,
                    direction_threshold=direction_threshold,
                )
                + signature.reasons
                + memory.reasons,
                rationale="Memory vetoed trade due to poor historical outcomes.",
                metadata={
                    "alignment": regime.alignment,
                    "raw_confidence": fusion.confidence,
                    "conflict": float(fusion.conflict),
                    "sig_net_bias": signature.net_bias,
                    "sig_net_strength": signature.net_strength,
                    "sig_boost": signature.boost,
                    "sig_veto": float(signature.veto),
                    "mem_key": memory.key,
                    "mem_n": float(memory.stats.n) if memory.stats else 0.0,
                    "mem_win_rate": (
                        memory.stats.wins / memory.stats.n
                        if memory.stats and memory.stats.n
                        else 0.0
                    ),
                    "mem_adj": memory.adjustment,
                    "mem_veto": float(memory.veto),
                    **self._flatten_weights(fusion.weights),
                },
            )

        action = self._direction_to_action(
            adjusted_direction,
            adjusted_confidence,
            min_confidence=min_confidence,
            direction_threshold=direction_threshold,
        )
        reasons = self._build_reasons(
            action=action,
            regime=regime,
            alignment=regime.alignment,
            fusion=fusion,
            min_confidence=min_confidence,
            direction_threshold=direction_threshold,
        ) + signature.reasons + memory.reasons
        rationale = self._build_rationale(action, regime, predictions)

        return TradeSignal(
            action=action,
            confidence=adjusted_confidence,
            regime=regime.regime,
            direction_score=adjusted_direction,
            reasons=reasons,
            rationale=rationale,
            metadata={
                "alignment": regime.alignment,
                "raw_confidence": fusion.confidence,
                "conflict": float(fusion.conflict),
                "sig_net_bias": signature.net_bias,
                "sig_net_strength": signature.net_strength,
                "sig_boost": signature.boost,
                "sig_veto": float(signature.veto),
                "mem_key": memory.key,
                "mem_n": float(memory.stats.n) if memory.stats else 0.0,
                "mem_win_rate": (
                    memory.stats.wins / memory.stats.n
                    if memory.stats and memory.stats.n
                    else 0.0
                ),
                "mem_adj": memory.adjustment,
                "mem_veto": float(memory.veto),
                **self._flatten_weights(fusion.weights),
            },
        )

    def _fuse_predictions(self, predictions: Iterable[ModelPrediction]) -> FusionResult:
        weighted_sum = 0.0
        total_weight = 0.0
        confidences: List[float] = []
        directions: List[float] = []
        weights: Dict[str, float] = {}

        for prediction in predictions:
            weighted_sum += prediction.direction * prediction.confidence
            total_weight += prediction.confidence
            confidences.append(prediction.confidence)
            directions.append(prediction.direction)

        if total_weight == 0.0:
            return FusionResult(
                direction=0.0,
                confidence=0.0,
                agreement=0.0,
                weights={},
                conflict=False,
                confidence_gap=0.0,
            )

        fused_direction = weighted_sum / total_weight
        weights = {
            prediction.source: prediction.confidence / total_weight
            for prediction in predictions
        }
        average_confidence = sum(confidences) / len(confidences)
        agreement = self._direction_agreement(directions)
        fused_confidence = max(
            0.0, min(1.0, average_confidence * (0.6 + 0.4 * agreement))
        )
        conflict = self._is_conflict(predictions)
        confidence_gap = self._confidence_gap_raw(confidences)
        if conflict:
            fused_confidence *= 0.6

        return FusionResult(
            direction=fused_direction,
            confidence=fused_confidence,
            agreement=agreement,
            weights=weights,
            conflict=conflict,
            confidence_gap=confidence_gap,
        )

    def _direction_agreement(self, directions: List[float]) -> float:
        if not directions:
            return 0.0

        positive = sum(1 for d in directions if d > 0.0)
        negative = sum(1 for d in directions if d < 0.0)
        neutral = len(directions) - positive - negative

        if positive and negative:
            return 0.25 + (neutral / len(directions)) * 0.25
        if positive or negative:
            return 0.75 + (neutral / len(directions)) * 0.25
        return 0.5

    def _assess_regime(self, contexts: List[TimeframeContext]) -> RegimeAssessment:
        if not contexts:
            return RegimeAssessment(
                regime=MarketRegime.UNKNOWN,
                trend=TrendDirection.NEUTRAL,
                alignment=0.0,
                avg_volatility=0.0,
                notes="No timeframe context provided",
            )

        trend_votes = {TrendDirection.BULLISH: 0, TrendDirection.BEARISH: 0}
        regime_votes = {MarketRegime.TREND: 0, MarketRegime.RANGE: 0}
        volatility_levels: List[float] = []

        for context in contexts:
            if context.trend in trend_votes:
                trend_votes[context.trend] += 1
            if context.regime in regime_votes:
                regime_votes[context.regime] += 1
            if context.volatility is None:
                continue
            if isinstance(context.volatility, float) and math.isnan(context.volatility):
                continue
            volatility_levels.append(float(context.volatility))

        dominant_trend = (
            TrendDirection.BULLISH
            if trend_votes[TrendDirection.BULLISH] > trend_votes[TrendDirection.BEARISH]
            else TrendDirection.BEARISH
            if trend_votes[TrendDirection.BEARISH] > trend_votes[TrendDirection.BULLISH]
            else TrendDirection.NEUTRAL
        )

        dominant_regime = (
            MarketRegime.TREND
            if regime_votes[MarketRegime.TREND] > regime_votes[MarketRegime.RANGE]
            else MarketRegime.RANGE
            if regime_votes[MarketRegime.RANGE] > regime_votes[MarketRegime.TREND]
            else MarketRegime.UNKNOWN
        )

        alignment = self._calculate_alignment(contexts, dominant_trend, dominant_regime)
        notes = f"Trend={dominant_trend.value}, Regime={dominant_regime.value}"
        avg_volatility = 0.0
        if volatility_levels:
            avg_volatility = sum(volatility_levels) / len(volatility_levels)
            avg_volatility = max(0.0, min(avg_volatility, 1.0))
            alignment *= 1.0 - avg_volatility * 0.1

        return RegimeAssessment(
            regime=dominant_regime,
            trend=dominant_trend,
            alignment=max(0.0, min(1.0, alignment)),
            avg_volatility=avg_volatility,
            notes=notes,
        )

    def _calculate_alignment(
        self,
        contexts: List[TimeframeContext],
        dominant_trend: TrendDirection,
        dominant_regime: MarketRegime,
    ) -> float:
        if dominant_trend is TrendDirection.NEUTRAL:
            return 0.5

        trend_alignment = sum(
            1 for context in contexts if context.trend == dominant_trend
        ) / len(contexts)
        regime_alignment = sum(
            1 for context in contexts if context.regime == dominant_regime
        ) / len(contexts)

        return 0.6 * trend_alignment + 0.4 * regime_alignment

    def _direction_to_action(
        self,
        direction_score: float,
        confidence: float,
        min_confidence: float,
        direction_threshold: float,
    ) -> TradeAction:
        if confidence < min_confidence:
            return TradeAction.HOLD
        if direction_score >= direction_threshold:
            return TradeAction.BUY
        if direction_score <= -direction_threshold:
            return TradeAction.SELL
        return TradeAction.HOLD

    def _build_rationale(
        self,
        action: TradeAction,
        regime: RegimeAssessment,
        predictions: Iterable[ModelPrediction],
    ) -> str:
        sources = ", ".join(pred.source for pred in predictions)
        if action is TradeAction.HOLD:
            return f"Hold due to low conviction. Regime: {regime.notes}. Models: {sources}."
        return f"{action.value} aligned with {regime.notes}. Models: {sources}."

    def _is_conflict(self, predictions: Iterable[ModelPrediction]) -> bool:
        preds = list(predictions)
        if len(preds) < 2:
            return False
        first, second = preds[0], preds[1]
        if first.confidence < 0.65 or second.confidence < 0.65:
            return False
        if first.direction == 0.0 or second.direction == 0.0:
            return False
        return (first.direction > 0) != (second.direction > 0)

    def _regime_thresholds(
        self,
        regime: RegimeAssessment,
        min_confidence: float,
        direction_threshold: float,
    ) -> tuple[float, float]:
        adjusted_confidence = min_confidence
        adjusted_threshold = direction_threshold

        if regime.regime is MarketRegime.RANGE:
            adjusted_threshold *= 1.5
            adjusted_confidence = min(1.0, adjusted_confidence + 0.05)

        if regime.avg_volatility >= 0.7:
            adjusted_threshold *= 1.25
            adjusted_confidence = min(1.0, adjusted_confidence + 0.05)

        return adjusted_confidence, adjusted_threshold

    def _build_reasons(
        self,
        action: TradeAction,
        regime: RegimeAssessment,
        alignment: float,
        fusion: FusionResult,
        min_confidence: float,
        direction_threshold: Optional[float] = None,
    ) -> List[str]:
        reasons: List[str] = []
        reasons.append(f"REGIME_{regime.regime.name}")
        reasons.append(f"TF_ALIGN_{alignment:.2f}")
        if fusion.conflict:
            reasons.append("MODEL_CONFLICT")
        else:
            reasons.append("MODEL_AGREE" if fusion.agreement >= 0.75 else "MODEL_MIXED")
        reasons.append(f"CONF_{fusion.confidence:.2f}")

        if action is TradeAction.HOLD:
            if fusion.conflict:
                if fusion.confidence_gap < 0.25:
                    reasons.append("HOLD_LOW_EDGE")
            elif fusion.confidence < min_confidence:
                reasons.append("HOLD_LOW_EDGE")
            elif direction_threshold is not None and abs(fusion.direction) < direction_threshold:
                reasons.append("HOLD_NO_EDGE")

        return reasons

    def _confidence_gap_raw(self, confidences: List[float]) -> float:
        if len(confidences) < 2:
            return 0.0
        values = sorted(confidences, reverse=True)
        return values[0] - values[1]

    def _flatten_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        return {f"weight_{key}": value for key, value in weights.items()}

    @staticmethod
    def _empty_memory_metadata() -> Dict[str, float]:
        return {
            "mem_key": "",
            "mem_n": 0.0,
            "mem_win_rate": 0.0,
            "mem_adj": 0.0,
            "mem_veto": 0.0,
        }

    @staticmethod
    def _bucket(value: float, bins: int) -> int:
        if bins <= 0:
            return 0
        clamped = max(0.0, min(value, 1.0))
        return min(int(clamped * bins), bins - 1)
