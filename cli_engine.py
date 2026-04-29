#!/usr/bin/env python3
"""
AURA CLI Step Execution Engine - SOC Simulator
Allows step-by-step execution of ML pipeline with detailed intermediate outputs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.helpers import setup_logging
from utils.prediction import AuraPredictor
from utils.preprocessing import (
    RISK_NAMES,
    build_feature_frame,
    load_raw_dataset,
)
from evo_integration import run_validation_and_tune
from utils.behavioral_profiles import combine_scores, get_behavior_store
from utils.orchestration import score_and_respond

logger = setup_logging("aura.cli")


class CliStepEngine:
    """Step-by-step execution engine for SOC workflow simulation"""

    def __init__(self):
        self.predictor: Optional[AuraPredictor] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None
        self.current_record: Optional[Dict[str, Any]] = None
        self.step_results: Dict[str, Any] = {}
        logger.info("CLI Step Engine initialized")

    def _ensure_dataset(self, limit: Optional[int] = None) -> None:
        if self.raw_df is None:
            self.raw_df = load_raw_dataset()
        if limit is not None:
            self.raw_df = self.raw_df.head(int(limit))

    def _ensure_features(self) -> None:
        self._ensure_dataset()
        if self.feature_df is None:
            if self.raw_df is None:
                raise RuntimeError("Dataset unavailable")
            self.feature_df = build_feature_frame(self.raw_df)

    def _ensure_models(self) -> None:
        if self.predictor is None:
            self.predictor = AuraPredictor()

    def _clamp_record_index(self, record_idx: int) -> int:
        if self.raw_df is None or len(self.raw_df) == 0:
            return 0
        try:
            idx = int(record_idx)
        except Exception:
            idx = 0
        return max(0, min(idx, len(self.raw_df) - 1))

    def step_load_dataset(self, limit: int = 5) -> Dict[str, Any]:
        """Step 1: Load dataset"""
        try:
            print("\n" + "=" * 60)
            print("STEP 1: DATASET LOADING")
            print("=" * 60)
            
            self._ensure_dataset(limit=limit)
            
            result = {
                "total_rows": len(self.raw_df),
                "columns": list(self.raw_df.columns),
                "sample": self.raw_df.iloc[0].to_dict() if len(self.raw_df) > 0 else {}
            }
            
            print(f"[OK] Loaded {result['total_rows']} records")
            print(f"[OK] Columns: {', '.join(result['columns'][:5])}...")
            
            self.step_results['load'] = result
            return result
        except Exception as e:
            error_msg = f"[FAILED] Dataset loading failed: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    def step_preprocessing(self) -> Dict[str, Any]:
        """Step 2: Preprocessing and feature engineering"""
        try:
            print("\n" + "=" * 60)
            print("STEP 2: PREPROCESSING & FEATURE ENGINEERING")
            print("=" * 60)
            
            self._ensure_features()
            
            result = {
                "processed_rows": len(self.feature_df),
                "feature_columns": list(self.feature_df.columns),
                "feature_count": len(self.feature_df.columns),
                "null_counts": self.feature_df.isnull().sum().to_dict(),
                "shape": self.feature_df.shape
            }
            
            print(f"[OK] Processed {result['processed_rows']} records")
            print(f"[OK] Engineered {result['feature_count']} features")
            print(f"[OK] Null values: {sum(result['null_counts'].values())}")
            
            self.step_results['preprocessing'] = result
            return result
        except Exception as e:
            error_msg = f"[FAILED] Preprocessing failed: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    def step_load_models(self) -> Dict[str, Any]:
        """Step 3: Load ML models"""
        try:
            print("\n" + "=" * 60)
            print("STEP 3: MODEL LOADING")
            print("=" * 60)
            
            self._ensure_models()
            
            result = {
                "risk_model": "RandomForest + IsolationForest",
                "cvss_model": "GradientBoosting",
                "lstm_model": "LSTM Sequence Model",
                "status": "Ready",
                "lstm_seq_len": self.predictor.risk.lstm_seq_len
            }
            
            print(f"[OK] Risk Model: {result['risk_model']}")
            print(f"[OK] CVSS Model: {result['cvss_model']}")
            print(f"[OK] LSTM Model: {result['lstm_model']} (seq_len={result['lstm_seq_len']})")
            
            self.step_results['models'] = result
            return result
        except Exception as e:
            error_msg = f"[FAILED] Model loading failed: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    def step_isolation_forest(self, record_idx: int = 0) -> Dict[str, Any]:
        """Step 4: Isolation Forest anomaly detection"""
        try:
            print("\n" + "=" * 60)
            print(f"STEP 4: ISOLATION FOREST ANOMALY DETECTION (Record {record_idx})")
            print("=" * 60)
            
            self._ensure_models()
            self._ensure_features()
            record_idx = self._clamp_record_index(record_idx)
            
            X = self.predictor._transform(self.feature_df)
            anomaly_scores = self.predictor.risk.isolation_forest.score_samples(X)
            anomaly_flags = self.predictor.risk.isolation_forest.predict(X)
            
            score = float(anomaly_scores[record_idx])
            flag = int(anomaly_flags[record_idx])
            
            result = {
                "anomaly_score": score,
                "anomaly_flag": flag,
                "interpretation": "ANOMALY DETECTED" if flag < 0 else "NORMAL",
                "threshold": "Isolation Forest isolation threshold",
                "all_scores_mean": float(np.mean(anomaly_scores)),
                "all_scores_std": float(np.std(anomaly_scores))
            }
            
            print(f"[OK] Anomaly Score: {score:.4f}")
            print(f"[OK] Flag: {result['interpretation']}")
            print(f"[OK] Mean across batch: {result['all_scores_mean']:.4f}")
            
            self.step_results['isolation_forest'] = result
            return result
        except Exception as e:
            error_msg = f"[FAILED] Isolation Forest failed: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    def step_random_forest_risk(self, record_idx: int = 0) -> Dict[str, Any]:
        """Step 5: Random Forest risk classification"""
        try:
            print("\n" + "=" * 60)
            print(f"STEP 5: RANDOM FOREST RISK CLASSIFICATION (Record {record_idx})")
            print("=" * 60)
            
            self._ensure_models()
            self._ensure_features()
            record_idx = self._clamp_record_index(record_idx)
            
            X = self.predictor._transform(self.feature_df)
            preds = self.predictor.risk.classifier.predict(X)
            proba = self.predictor.risk.classifier.predict_proba(X)
            
            pred_label = int(preds[record_idx])
            pred_name = RISK_NAMES[pred_label]
            proba_dict = {RISK_NAMES[i]: float(proba[record_idx][i]) for i in range(len(RISK_NAMES))}
            
            result = {
                "predicted_class": pred_name,
                "predicted_code": pred_label,
                "probabilities": proba_dict,
                "top_confidence": float(max(proba[record_idx])),
                "decision_explanation": f"Classified as {pred_name} with {max(proba[record_idx]):.2%} confidence"
            }
            
            print(f"[OK] Classification: {pred_name}")
            print(f"[OK] Probabilities:")
            for risk_name, prob in proba_dict.items():
                print(f"    - {risk_name}: {prob:.2%}")
            
            self.step_results['random_forest'] = result
            return result
        except Exception as e:
            error_msg = f"[FAILED] Random Forest failed: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    def step_gradient_boosting_cvss(self, record_idx: int = 0) -> Dict[str, Any]:
        """Step 6: Gradient Boosting CVSS prediction"""
        try:
            print("\n" + "=" * 60)
            print(f"STEP 6: GRADIENT BOOSTING CVSS PREDICTION (Record {record_idx})")
            print("=" * 60)
            
            self._ensure_models()
            self._ensure_features()
            record_idx = self._clamp_record_index(record_idx)
            
            X_cvss = self.predictor._transform_cvss(self.feature_df)
            cvss_pred = self.predictor.cvss.regressor.predict(X_cvss)
            
            cvss_value = float(cvss_pred[record_idx])
            cvss_value = max(0.0, min(10.0, cvss_value))  # Clamp to 0-10
            
            severity = "Critical" if cvss_value >= 9.0 else "High" if cvss_value >= 7.0 else "Medium" if cvss_value >= 4.0 else "Low"
            
            result = {
                "cvss_score": cvss_value,
                "severity": severity,
                "range_0_10": True,
                "mean_prediction": float(np.mean(cvss_pred)),
                "std_prediction": float(np.std(cvss_pred)),
                "severity_interpretation": f"CVSS {cvss_value:.1f} = {severity} Severity"
            }
            
            print(f"[OK] CVSS Score: {cvss_value:.2f}/10.0")
            print(f"[OK] Severity: {severity}")
            print(f"[OK] Mean across batch: {result['mean_prediction']:.2f}")
            
            self.step_results['gradient_boosting'] = result
            return result
        except Exception as e:
            error_msg = f"[FAILED] Gradient Boosting failed: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    def step_lstm_temporal(self, record_idx: int = 0) -> Dict[str, Any]:
        """Step 7: LSTM temporal analysis"""
        try:
            print("\n" + "=" * 60)
            print(f"STEP 7: LSTM TEMPORAL ANALYSIS (Record {record_idx})")
            print("=" * 60)
            
            self._ensure_models()
            self._ensure_features()
            record_idx = self._clamp_record_index(record_idx)
            
            try:
                X_lstm = self.predictor._prepare_lstm_input(self.feature_df, record_idx)
                lstm_output = self.predictor.lstm.predict(X_lstm, verbose=0)
                
                result = {
                    "sequence_length": self.predictor.risk.lstm_seq_len,
                    "lstm_output_shape": lstm_output.shape,
                    "temporal_anomaly_score": float(np.mean(lstm_output)),
                    "lstm_available": True,
                    "interpretation": "LSTM sequence analysis complete"
                }
                
                print(f"[OK] Sequence Length: {result['sequence_length']}")
                print(f"[OK] Temporal Anomaly Score: {result['temporal_anomaly_score']:.4f}")
                print(f"[OK] Output Shape: {result['lstm_output_shape']}")
                
            except Exception as lstm_err:
                result = {
                    "lstm_available": False,
                    "note": "LSTM analysis skipped (optional enhancement)",
                    "error": str(lstm_err)
                }
                print(f"[WARNING] LSTM skipped: {result['note']}")
            
            self.step_results['lstm'] = result
            return result
        except Exception as e:
            error_msg = f"[FAILED] LSTM temporal analysis failed: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    def step_risk_fusion(self, record_idx: int = 0) -> Dict[str, Any]:
        """Step 8: Risk fusion from all models"""
        try:
            print("\n" + "=" * 60)
            print(f"STEP 8: RISK FUSION & FINAL DECISION (Record {record_idx})")
            print("=" * 60)
            
            self._ensure_models()
            self._ensure_dataset()
            record_idx = self._clamp_record_index(record_idx)
            
            # Get all model outputs
            preds = self.predictor.predict_records(
                [self.raw_df.iloc[record_idx].to_dict()],
                include_explanation=True,
                use_llm=False
            )
            
            fusion_result = preds[0]
            
            result = {
                "final_risk_class": fusion_result.get("risk_class", "Unknown"),
                "risk_code": fusion_result.get("risk_code", -1),
                "final_cvss": fusion_result.get("cvss_predicted", 0.0),
                "anomaly_flag": fusion_result.get("anomaly_score_flag", 0),
                "probabilities": fusion_result.get("risk_probabilities", {}),
                "explanation": fusion_result.get("explanation", {}).get("narrative", ""),
                "recommended_action": self._get_recommended_action(fusion_result.get("risk_class", ""))
            }
            
            print(f"[OK] Final Risk Class: {result['final_risk_class']}")
            print(f"[OK] CVSS: {result['final_cvss']:.2f}")
            print(f"[OK] Anomaly Flag: {result['anomaly_flag']}")
            print(f"[OK] Recommended Action: {result['recommended_action']}")
            
            self.step_results['fusion'] = result
            return result
        except Exception as e:
            error_msg = f"[FAILED] Risk fusion failed: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    def step_behavioral_analysis(self, record_idx: int = 0) -> Dict[str, Any]:
        """Step 9: Behavioral profile analysis"""
        try:
            print("\n" + "=" * 60)
            print(f"STEP 9: BEHAVIORAL ANALYSIS (Record {record_idx})")
            print("=" * 60)
            
            behavior_store = get_behavior_store()
            
            self._ensure_dataset()
            record_idx = self._clamp_record_index(record_idx)
            if self.raw_df is not None and "ip_source" in self.raw_df.columns:
                ip = str(self.raw_df.iloc[record_idx]["ip_source"])
                profile = behavior_store.get(ip, {})
                
                result = {
                    "ip_address": ip,
                    "access_count": profile.get("access_count", 0),
                    "risk_history": profile.get("risk_history", []),
                    "avg_risk_score": profile.get("avg_risk_score", 0.0),
                    "anomaly_count": profile.get("anomaly_count", 0),
                    "profile_exists": len(profile) > 0,
                    "behavioral_risk": "High" if profile.get("anomaly_count", 0) > 2 else "Medium" if profile.get("anomaly_count", 0) > 0 else "Low"
                }
                
                print(f"[OK] IP Address: {ip}")
                print(f"[OK] Access Count: {result['access_count']}")
                print(f"[OK] Anomaly Count: {result['anomaly_count']}")
                print(f"[OK] Behavioral Risk: {result['behavioral_risk']}")
                
            else:
                result = {
                    "note": "Behavioral profile not available",
                    "profile_exists": False
                }
            
            self.step_results['behavioral'] = result
            return result
        except Exception as e:
            error_msg = f"[FAILED] Behavioral analysis failed: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    def step_firewall_decision(self, record_idx: int = 0) -> Dict[str, Any]:
        """Step 10: Firewall decision and action"""
        try:
            print("\n" + "=" * 60)
            print(f"STEP 10: FIREWALL DECISION & SYSTEM ACTION")
            print("=" * 60)
            
            self._ensure_models()
            self._ensure_dataset()
            record_idx = self._clamp_record_index(record_idx)
            
            record = self.raw_df.iloc[record_idx].to_dict()
            fusion_result = self.step_results.get('fusion', {})
            risk_class = fusion_result.get('final_risk_class', 'Unknown')
            
            result = {
                "risk_class": risk_class,
                "action": self._get_firewall_action(risk_class),
                "alert_level": "CRITICAL" if risk_class in ("Critical", "Malicious") else "WARNING" if risk_class == "Suspicious" else "INFO",
                "dry_run": True,
                "system_response": self._get_system_response(risk_class)
            }
            
            print(f"[OK] Risk Assessment: {result['risk_class']}")
            print(f"[OK] Alert Level: {result['alert_level']}")
            print(f"[OK] Firewall Action: {result['action']}")
            print(f"[OK] System Response: {result['system_response']}")
            print(f"[OK] Mode: DRY-RUN (no actual rules applied)")
            
            self.step_results['firewall'] = result
            return result
        except Exception as e:
            error_msg = f"[FAILED] Firewall decision failed: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    def step_evolutionary_tuning(self, max_rows: int = 200) -> Dict[str, Any]:
        """Step 11: Evolutionary optimization tuning (GA + PSO)."""
        try:
            print("\n" + "=" * 60)
            print("STEP 11: EVOLUTIONARY TUNING (GA + PSO)")
            print("=" * 60)

            self._ensure_models()
            self._ensure_dataset()

            report = run_validation_and_tune(self.predictor, self.raw_df, max_rows=int(max_rows), logger=logger)
            baseline = report.get("baseline", {}).get("metrics", {})
            tuned = report.get("tuned", {}).get("metrics", {})
            tuned_policy = report.get("tuned", {}).get("policy", {})
            saved_path = report.get("saved_policy_path", "unknown")

            result = {
                "baseline_metrics": baseline,
                "tuned_metrics": tuned,
                "policy_path": saved_path,
                "policy": tuned_policy,
                "fitness_improvement": float(report.get("tuned", {}).get("fitness", 0.0)) - float(report.get("baseline", {}).get("fitness", 0.0)),
            }

            print(f"[OK] Evolutionary tuning complete")
            print(f"  Baseline accuracy: {float(baseline.get('accuracy', 0.0)):.4f}")
            print(f"  Tuned accuracy: {float(tuned.get('accuracy', 0.0)):.4f}")
            print(f"  Baseline FPR: {float(baseline.get('fpr', 0.0)):.4f}")
            print(f"  Tuned FPR: {float(tuned.get('fpr', 0.0)):.4f}")
            print(f"  Baseline FNR: {float(baseline.get('fnr', 0.0)):.4f}")
            print(f"  Tuned FNR: {float(tuned.get('fnr', 0.0)):.4f}")
            print(f"  Saved policy: {saved_path}")

            self.step_results['evolutionary_tuning'] = result
            return result
        except Exception as e:
            error_msg = f"[FAILED] Evolutionary tuning failed: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    def print_summary(self) -> None:
        """Print complete pipeline summary"""
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        
        for step_name, step_data in self.step_results.items():
            if "error" not in step_data:
                print(f"[OK] {step_name.upper()}: Complete")
            else:
                print(f"[FAILED] {step_name.upper()}: Failed - {step_data['error']}")
        
        print("\n" + "=" * 60)
        if 'fusion' in self.step_results and 'error' not in self.step_results['fusion']:
            fusion = self.step_results['fusion']
            print(f"FINAL DECISION: {fusion.get('final_risk_class', 'Unknown')}")
            print(f"RECOMMENDED ACTION: {fusion.get('recommended_action', 'N/A')}")
        print("=" * 60 + "\n")

    @staticmethod
    def _get_recommended_action(risk_class: str) -> str:
        """Get recommended action for risk class"""
        if risk_class == "Critical":
            return "IMMEDIATE: Block, isolate, and escalate to SOC"
        elif risk_class == "Malicious":
            return "URGENT: Block ingress and alert SOC for investigation"
        elif risk_class == "Suspicious":
            return "MODERATE: Apply rate limits and enhanced logging"
        else:
            return "LOW: Allow with baseline monitoring"

    @staticmethod
    def _get_firewall_action(risk_class: str) -> str:
        """Get firewall action for risk class"""
        if risk_class in ("Critical", "Malicious"):
            return "DROP + BLOCK + ALERT"
        elif risk_class == "Suspicious":
            return "RATE_LIMIT + LOG"
        else:
            return "ALLOW + MONITOR"

    @staticmethod
    def _get_system_response(risk_class: str) -> str:
        """Get system response for risk class"""
        if risk_class in ("Critical", "Malicious"):
            return "Generate critical alert, notify SOC, quarantine session"
        elif risk_class == "Suspicious":
            return "Generate warning alert, require re-authentication"
        else:
            return "Log event, continue monitoring"

    def run_anomaly_check(self) -> Dict[str, Any]:
        """Real-time anomaly check on custom or sample data"""
        try:
            print("\n" + "=" * 60)
            print("REAL-TIME ANOMALY CHECK")
            print("=" * 60)
            
            self._ensure_models()
            self._ensure_dataset()
            
            print("\nOptions:")
            print("  1. Use sample data from dataset")
            print("  2. Input custom data values")
            print("  q. Back to menu")
            
            choice = input("\nSelect option (1-2): ").strip()
            if choice.lower() == "q":
                return {"status": "cancelled"}
            
            if choice == '1':
                # Use sample from dataset
                if len(self.raw_df) == 0:
                    print("[FAILED] No sample data available")
                    return {"error": "No data available"}

                try:
                    sample_idx = int(input(f"Select record index (0-{len(self.raw_df)-1}) [0]: ").strip() or "0")
                except ValueError:
                    sample_idx = 0
                sample_idx = self._clamp_record_index(sample_idx)
                print(f"\nUsing sample record {sample_idx}:")
                sample_record = self.raw_df.iloc[sample_idx].to_dict()
                
            elif choice == '2':
                # Manual input
                print("\nEnter feature values (press Enter to keep default):")
                sample_record = {}
                base_row = self.raw_df.iloc[0].to_dict() if len(self.raw_df) else {}
                for col in self.raw_df.columns:
                    default_val = base_row.get(col)
                    val = input(f"  {col} [default: {default_val}]: ").strip()
                    if val:
                        try:
                            sample_record[col] = float(val) if '.' in val else int(val)
                        except ValueError:
                            sample_record[col] = val
                    else:
                        sample_record[col] = default_val
            else:
                print("[FAILED] Invalid option")
                return {"error": "Invalid selection"}
            
            # Run prediction
            print("\n" + "=" * 60)
            print("PROCESSING INPUT DATA")
            print("=" * 60)
            
            predictions = self.predictor.predict_records(
                [sample_record],
                include_explanation=False,
                use_llm=False
            )
            
            pred = predictions[0]
            
            result = {
                "status": "Anomaly Check Complete",
                "input_record": sample_record,
                "classification": pred.get("risk_class", "Unknown"),
                "risk_code": pred.get("risk_code", -1),
                "cvss_score": pred.get("cvss_predicted", 0.0),
                "anomaly_flag": pred.get("anomaly_score_flag", 0),
                "anomaly_score": pred.get("anomaly_score", 0.0),
                "confidence": pred.get("confidence", 0.0),
                "interpretation": self._get_anomaly_interpretation(
                    pred.get("risk_class", "Unknown"),
                    pred.get("cvss_predicted", 0.0),
                    pred.get("anomaly_score_flag", 0)
                )
            }
            
            print("\n[RESULTS]")
            print(f"Classification: {result['classification']}")
            print(f"CVSS Score: {result['cvss_score']:.2f}/10.0")
            print(f"Anomaly Flag: {'ANOMALOUS' if result['anomaly_flag'] < 0 else 'NORMAL'}")
            print(f"Anomaly Score: {float(result['anomaly_score']):.4f}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"\nInterpretation:")
            print(result['interpretation'])
            
            self.step_results['anomaly_check'] = result
            return result
            
        except Exception as e:
            error_msg = f"[FAILED] Anomaly check failed: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    @staticmethod
    def _get_anomaly_interpretation(risk_class: str, cvss: float, anomaly_flag: int) -> str:
        """Generate human-readable interpretation of anomaly check result"""
        interpretation = []
        
        interpretation.append(f"Risk Classification: {risk_class}")
        
        if risk_class == "Critical":
            interpretation.append("Assessment: IMMEDIATE ACTION REQUIRED")
            interpretation.append("Recommendation: Isolate asset, escalate to SOC, preserve evidence")
        elif risk_class == "Malicious":
            interpretation.append("Assessment: CONFIRMED THREAT")
            interpretation.append("Recommendation: Block traffic, investigate source, review logs")
        elif risk_class == "Suspicious":
            interpretation.append("Assessment: REQUIRES INVESTIGATION")
            interpretation.append("Recommendation: Enable enhanced monitoring, correlate with other alerts")
        else:
            interpretation.append("Assessment: NORMAL BEHAVIOR")
            interpretation.append("Recommendation: Continue baseline monitoring")
        
        if anomaly_flag < 0:
            interpretation.append("Anomaly Detection: ANOMALOUS (unusual pattern detected)")
        else:
            interpretation.append("Anomaly Detection: NORMAL (consistent with baseline)")
        
        severity = "Critical" if cvss >= 9.0 else "High" if cvss >= 7.0 else "Medium" if cvss >= 4.0 else "Low"
        interpretation.append(f"Severity: {severity} (CVSS {cvss:.1f}/10.0)")
        
        return "\n".join(interpretation)


def main():
    """Interactive CLI menu"""
    engine = CliStepEngine()
    
    print("\n" + "=" * 60)
    print("AURA CLI STEP EXECUTION ENGINE - SOC SIMULATOR")
    print("=" * 60)
    print("\nAvailable Steps:")
    print("  1. Load Dataset")
    print("  2. Preprocessing")
    print("  3. Load Models")
    print("  4. Isolation Forest")
    print("  5. Random Forest Risk Classification")
    print("  6. Gradient Boosting CVSS")
    print("  7. LSTM Temporal Analysis")
    print("  8. Risk Fusion & Final Decision")
    print("  9. Behavioral Analysis")
    print("  10. Firewall Decision")
    print("  11. Evolutionary Tuning")
    print("  12. Real-Time Anomaly Check")
    print("  0. Run All Steps")
    print("  q. Quit")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\nEnter step number (0-12, q to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting AURA CLI...")
            break
        
        if choice == 'q':
            print("Exiting AURA CLI...")
            break
        elif choice == '0':
            print("\nRunning all steps...")
            engine.step_load_dataset(limit=1)
            engine.step_preprocessing()
            engine.step_load_models()
            engine.step_isolation_forest()
            engine.step_random_forest_risk()
            engine.step_gradient_boosting_cvss()
            engine.step_lstm_temporal()
            engine.step_risk_fusion()
            engine.step_behavioral_analysis()
            engine.step_firewall_decision()
            engine.step_evolutionary_tuning()
            engine.print_summary()
        elif choice == '1':
            engine.step_load_dataset(limit=1)
        elif choice == '2':
            engine.step_preprocessing()
        elif choice == '3':
            engine.step_load_models()
        elif choice == '4':
            engine.step_isolation_forest()
        elif choice == '5':
            engine.step_random_forest_risk()
        elif choice == '6':
            engine.step_gradient_boosting_cvss()
        elif choice == '7':
            engine.step_lstm_temporal()
        elif choice == '8':
            engine.step_risk_fusion()
        elif choice == '9':
            engine.step_behavioral_analysis()
        elif choice == '10':
            engine.step_firewall_decision()
        elif choice == '11':
            engine.step_evolutionary_tuning()
        elif choice == '12':
            engine.run_anomaly_check()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
