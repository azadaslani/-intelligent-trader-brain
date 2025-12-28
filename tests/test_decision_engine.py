import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from decision_engine import (
    DecisionEngine,
    load_signature_bank,
    MemoryLayer,
    MarketRegime,
    ModelPrediction,
    SignatureHit,
    SignatureLayer,
    TimeframeContext,
    TradeAction,
    TrendDirection,
)


class DecisionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        bank_path = Path(__file__).resolve().parents[1] / "signature_bank.json"
        self._bank = load_signature_bank(str(bank_path))
        self._memory_layer = MemoryLayer({"keys": {}})

    def _context(self, trend: TrendDirection) -> dict[str, TimeframeContext]:
        return {
            "M30": TimeframeContext(
                timeframe="M30",
                trend=trend,
                regime=MarketRegime.TREND,
                volatility=0.2,
            ),
            "H4": TimeframeContext(
                timeframe="H4",
                trend=trend,
                regime=MarketRegime.TREND,
                volatility=0.25,
            ),
        }

    def test_buy_when_models_agree_bullish_trend(self) -> None:
        engine = DecisionEngine()
        signal = engine.generate_signal(
            ModelPrediction(source="LSTM", direction=0.7, confidence=0.8),
            ModelPrediction(source="XGB", direction=0.6, confidence=0.75),
            self._context(TrendDirection.BULLISH),
            symbol="EURUSD",
            primary_timeframe="M30",
        )
        self.assertIs(signal.action, TradeAction.BUY)

    def test_sell_when_models_agree_bearish_trend(self) -> None:
        engine = DecisionEngine()
        signal = engine.generate_signal(
            ModelPrediction(source="LSTM", direction=-0.7, confidence=0.8),
            ModelPrediction(source="XGB", direction=-0.6, confidence=0.75),
            self._context(TrendDirection.BEARISH),
            symbol="EURUSD",
            primary_timeframe="M30",
        )
        self.assertIs(signal.action, TradeAction.SELL)

    def test_hold_on_conflict_low_gap(self) -> None:
        engine = DecisionEngine()
        signal = engine.generate_signal(
            ModelPrediction(source="LSTM", direction=0.8, confidence=0.7),
            ModelPrediction(source="XGB", direction=-0.7, confidence=0.7),
            self._context(TrendDirection.BULLISH),
            symbol="EURUSD",
            primary_timeframe="M30",
        )
        self.assertIs(signal.action, TradeAction.HOLD)

    def test_hold_on_bearish_signature_veto(self) -> None:
        engine = DecisionEngine(signature_layer=SignatureLayer(self._bank))
        signal = engine.generate_signal(
            ModelPrediction(source="LSTM", direction=0.2, confidence=0.8),
            ModelPrediction(source="XGB", direction=0.25, confidence=0.8),
            self._context(TrendDirection.BULLISH),
            symbol="EURUSD",
            primary_timeframe="M30",
            signature_hits=[
                SignatureHit(
                    name="bear_trap",
                    bias=-0.9,
                    strength=0.8,
                    confidence=0.9,
                )
            ],
        )
        self.assertIs(signal.action, TradeAction.HOLD)

    def test_buy_with_bullish_signature_boost(self) -> None:
        engine = DecisionEngine(signature_layer=SignatureLayer(self._bank))
        baseline = engine.generate_signal(
            ModelPrediction(source="LSTM", direction=0.3, confidence=0.7),
            ModelPrediction(source="XGB", direction=0.35, confidence=0.7),
            self._context(TrendDirection.BULLISH),
            symbol="EURUSD",
            primary_timeframe="M30",
        )
        boosted = engine.generate_signal(
            ModelPrediction(source="LSTM", direction=0.3, confidence=0.7),
            ModelPrediction(source="XGB", direction=0.35, confidence=0.7),
            self._context(TrendDirection.BULLISH),
            symbol="EURUSD",
            primary_timeframe="M30",
            signature_hits=[
                SignatureHit(
                    name="breakout",
                    bias=0.8,
                    strength=0.6,
                    confidence=0.9,
                )
            ],
        )
        self.assertIs(boosted.action, TradeAction.BUY)
        self.assertGreaterEqual(boosted.confidence, baseline.confidence)

    def test_unknown_signature_uses_default_weight(self) -> None:
        engine = DecisionEngine(signature_layer=SignatureLayer(self._bank))
        signal = engine.generate_signal(
            ModelPrediction(source="LSTM", direction=0.4, confidence=0.7),
            ModelPrediction(source="XGB", direction=0.4, confidence=0.7),
            self._context(TrendDirection.BULLISH),
            symbol="EURUSD",
            primary_timeframe="M30",
            signature_hits=[
                SignatureHit(
                    name="unknown_sig",
                    bias=0.5,
                    strength=1.0,
                    confidence=1.0,
                )
            ],
        )
        self.assertAlmostEqual(signal.metadata["sig_net_strength"], 0.5, places=2)

    def test_min_strength_filters_out_weak_hit(self) -> None:
        engine = DecisionEngine(signature_layer=SignatureLayer(self._bank))
        signal = engine.generate_signal(
            ModelPrediction(source="LSTM", direction=0.4, confidence=0.7),
            ModelPrediction(source="XGB", direction=0.4, confidence=0.7),
            self._context(TrendDirection.BULLISH),
            symbol="EURUSD",
            primary_timeframe="M30",
            signature_hits=[
                SignatureHit(
                    name="GOLDEN_FRAME",
                    bias=0.9,
                    strength=0.5,
                    confidence=0.9,
                )
            ],
        )
        self.assertAlmostEqual(signal.metadata["sig_net_strength"], 0.0, places=2)

    def test_memory_veto_on_bad_win_rate(self) -> None:
        engine = DecisionEngine()
        mem_layer = MemoryLayer(
            {
                "keys": {
                    "EURUSD:M30:trend:1:4:SIG_NET_neutral-SIG_STRENGTH_0.00": {
                        "n": 30,
                        "wins": 9,
                        "losses": 21,
                        "avg_r": -0.2,
                        "last_update": 1,
                    }
                }
            }
        )
        signal = engine.generate_signal(
            ModelPrediction(source="LSTM", direction=0.4, confidence=0.8),
            ModelPrediction(source="XGB", direction=0.4, confidence=0.8),
            self._context(TrendDirection.BULLISH),
            symbol="EURUSD",
            primary_timeframe="M30",
            memory_layer=mem_layer,
        )
        self.assertIs(signal.action, TradeAction.HOLD)
        self.assertIn("MEM_VETO", signal.reasons)

    def test_memory_boosts_confidence(self) -> None:
        engine = DecisionEngine()
        mem_layer = MemoryLayer(
            {
                "keys": {
                    "EURUSD:M30:trend:1:4:SIG_NET_neutral-SIG_STRENGTH_0.00": {
                        "n": 20,
                        "wins": 14,
                        "losses": 6,
                        "avg_r": 0.1,
                        "last_update": 1,
                    }
                }
            }
        )
        baseline = engine.generate_signal(
            ModelPrediction(source="LSTM", direction=0.4, confidence=0.8),
            ModelPrediction(source="XGB", direction=0.4, confidence=0.8),
            self._context(TrendDirection.BULLISH),
            symbol="EURUSD",
            primary_timeframe="M30",
            memory_layer=self._memory_layer,
        )
        boosted = engine.generate_signal(
            ModelPrediction(source="LSTM", direction=0.4, confidence=0.8),
            ModelPrediction(source="XGB", direction=0.4, confidence=0.8),
            self._context(TrendDirection.BULLISH),
            symbol="EURUSD",
            primary_timeframe="M30",
            memory_layer=mem_layer,
        )
        self.assertGreaterEqual(boosted.confidence, baseline.confidence)


if __name__ == "__main__":
    unittest.main()
