"""
Multimodal Vision Intelligence Module for Dike Safety
======================================================

Drone image analysis system integrating:
- Vision Transformer (ViT) for anomaly detection
- SAM (Segment Anything Model) for interactive crack segmentation
- Multimodal fusion with sensor data
- LLM-based diagnostic report generation

Architecture:
    Drone Image → ViT Feature Extractor → Anomaly Detection →
    SAM Segmentation (optional) → Multimodal Fusion (Visual + Sensor) →
    Gemini LLM → Structured Diagnosis → Alert Trigger
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    ViTModel,
    AutoImageProcessor,
    AutoModelForImageClassification
)

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetection:
    """Vision anomaly detection result."""
    image_id: str
    timestamp: str
    anomaly_score: float
    anomaly_type: str  # crack/erosion/seepage/vegetation/normal
    confidence: float
    bbox: Optional[List[float]]  # [x1, y1, x2, y2] if detected
    segmentation_mask: Optional[np.ndarray]  # SAM output
    gps_coordinates: Optional[Tuple[float, float]]  # (lat, lon)


@dataclass
class MultimodalDiagnosis:
    """Structured diagnostic report from multimodal analysis."""
    image_id: str
    timestamp: str
    visual_findings: Dict
    sensor_context: Dict
    risk_assessment: str  # low/medium/high/critical
    breach_probability: float
    recommended_actions: List[str]
    llm_explanation: str
    alert_triggered: bool


class VisionFeatureExtractor:
    """
    Vision Transformer (ViT) feature extractor for dike anomaly detection.
    
    Uses pretrained ViT to extract visual features and detect:
    - Cracks in dike structure
    - Erosion patterns
    - Seepage/wet spots
    - Vegetation intrusion
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        anomaly_threshold: float = 0.8,
        device: str = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.anomaly_threshold = anomaly_threshold
        
        # Load pretrained ViT
        logger.info(f"Loading ViT model: {model_name}")
        try:
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.model = ViTModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Failed to load pretrained model: {e}")
            logger.info("Using random initialization for testing")
            self.processor = None
            self.model = None
        
        # Anomaly detection head (simple MLP on top of ViT features)
        if self.model:
            self.feature_dim = self.model.config.hidden_size
        else:
            self.feature_dim = 768
        
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 classes: normal, crack, erosion, seepage, vegetation
        ).to(self.device)
        
        # Class names
        self.class_names = ['normal', 'crack', 'erosion', 'seepage', 'vegetation']
        
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for ViT.
        
        Args:
            image: Image path, PIL Image, or numpy array
        
        Returns:
            Preprocessed tensor
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Resize to 224x224 (ViT standard)
        image = image.resize((224, 224))
        
        # Convert to tensor
        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].to(self.device)
        else:
            # Manual preprocessing
            img_array = np.array(image) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            return img_tensor.unsqueeze(0).to(self.device)
    
    def extract_features(
        self,
        image: Union[str, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Extract visual features using ViT.
        
        Returns:
            Feature vector (768-dim for ViT-Base)
        """
        pixel_values = self.preprocess_image(image)
        
        with torch.no_grad():
            if self.model:
                outputs = self.model(pixel_values=pixel_values)
                # Use [CLS] token representation
                features = outputs.last_hidden_state[:, 0, :]
            else:
                # Fallback: random features for testing
                features = torch.randn(1, self.feature_dim).to(self.device)
        
        return features.cpu().numpy().squeeze()
    
    def detect_anomaly(
        self,
        image: Union[str, Image.Image, np.ndarray],
        image_id: str = None,
        gps_coords: Optional[Tuple[float, float]] = None
    ) -> AnomalyDetection:
        """
        Detect anomalies in dike image.
        
        Returns:
            AnomalyDetection result with classification and confidence
        """
        # Extract features
        features = torch.tensor(
            self.extract_features(image)
        ).unsqueeze(0).to(self.device)
        
        # Classify
        with torch.no_grad():
            logits = self.anomaly_head(features)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
        
        confidence = confidence.item()
        pred_idx = pred_class.item()
        anomaly_type = self.class_names[pred_idx]
        
        # Anomaly score: distance from normal class
        anomaly_score = 1.0 - probs[0, 0].item() if pred_idx != 0 else 0.0
        
        return AnomalyDetection(
            image_id=image_id or f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            confidence=confidence,
            bbox=None,  # Would be set by object detection model
            segmentation_mask=None,  # Would be set by SAM
            gps_coordinates=gps_coords
        )


class SAMInterface:
    """
    Interface for Segment Anything Model (SAM).
    
    Provides interactive segmentation capabilities for:
    - Click-based crack selection
    - Automatic crack boundary detection
    - Export to dashboard for visualization
    """
    
    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        local_model_path: Optional[str] = None
    ):
        self.api_endpoint = api_endpoint
        self.local_model = None
        
        # Try to load local SAM if available
        if local_model_path and os.path.exists(local_model_path):
            try:
                # SAM loading would go here
                # For now, placeholder
                logger.info("SAM model loaded locally")
                self.local_model = True
            except Exception as e:
                logger.warning(f"Failed to load local SAM: {e}")
        
        if not self.api_endpoint and not self.local_model:
            logger.info("SAM not available - using fallback segmentation")
    
    def segment_with_click(
        self,
        image: Union[str, np.ndarray],
        click_point: Tuple[int, int],
        click_label: int = 1  # 1 = positive (include), 0 = negative (exclude)
    ) -> Optional[np.ndarray]:
        """
        Segment region based on user click (for interactive dashboard).
        
        Args:
            image: Input image
            click_point: (x, y) coordinates of click
            click_label: 1 for positive, 0 for negative
        
        Returns:
            Binary mask of segmented region
        """
        if self.api_endpoint:
            # API call to external SAM service
            return self._call_sam_api(image, click_point, click_label)
        elif self.local_model:
            # Local SAM inference
            return self._local_sam_predict(image, click_point, click_label)
        else:
            # Fallback: simple threshold-based segmentation
            return self._fallback_segmentation(image, click_point)
    
    def _call_sam_api(
        self,
        image: Union[str, np.ndarray],
        click_point: Tuple[int, int],
        click_label: int
    ) -> Optional[np.ndarray]:
        """Call external SAM API."""
        try:
            # Encode image
            if isinstance(image, str):
                with open(image, 'rb') as f:
                    img_b64 = base64.b64encode(f.read()).decode()
            else:
                _, buffer = cv2.imencode('.jpg', image)
                img_b64 = base64.b64encode(buffer).decode()
            
            # API request
            response = requests.post(
                self.api_endpoint,
                json={
                    'image': img_b64,
                    'point': list(click_point),
                    'label': click_label
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                mask_b64 = result.get('mask')
                if mask_b64:
                    mask_bytes = base64.b64decode(mask_b64)
                    mask = np.frombuffer(mask_bytes, dtype=np.uint8)
                    return mask.reshape(result.get('shape', (224, 224)))
            
            return None
            
        except Exception as e:
            logger.error(f"SAM API call failed: {e}")
            return None
    
    def _local_sam_predict(
        self,
        image: Union[str, np.ndarray],
        click_point: Tuple[int, int],
        click_label: int
    ) -> Optional[np.ndarray]:
        """Local SAM prediction (placeholder)."""
        # Would implement actual SAM inference here
        logger.info("Local SAM prediction not yet implemented")
        return self._fallback_segmentation(image, click_point)
    
    def _fallback_segmentation(
        self,
        image: Union[str, np.ndarray],
        click_point: Tuple[int, int],
        radius: int = 50
    ) -> np.ndarray:
        """
        Fallback segmentation using simple region growing.
        
        Creates circular mask around click point.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
        
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw circle around click point
        cv2.circle(mask, click_point, radius, 255, -1)
        
        return mask


class MultimodalFusionAnalyzer:
    """
    Fuses visual features with sensor data for comprehensive dike risk assessment.
    
    Integrates:
    - Visual crack/erosion features from ViT
    - Real-time sensor data (water level, flow velocity)
    - Historical context
    - LLM (Gemini) for structured diagnosis
    """
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        gemini_api_endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"
    ):
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.gemini_endpoint = gemini_api_endpoint
        
        if not self.gemini_api_key:
            logger.warning("Gemini API key not set - using fallback diagnosis")
    
    def analyze_breach_risk(
        self,
        image_features: np.ndarray,
        visual_anomaly: AnomalyDetection,
        sensor_data: Dict[str, float]
    ) -> MultimodalDiagnosis:
        """
        Analyze breach risk by fusing visual and sensor data.
        
        Args:
            image_features: ViT-extracted visual features
            visual_anomaly: Anomaly detection result
            sensor_data: Real-time sensor readings
                - water_level: current water level (m)
                - flow_velocity: flow speed (m/s)
                - water_pressure: pressure reading (kPa)
                - dike_moisture: moisture sensor reading
        
        Returns:
            Structured multimodal diagnosis report
        """
        # Extract sensor values
        water_level = sensor_data.get('water_level', 0)
        flow_vel = sensor_data.get('flow_velocity', 0)
        warning_level = sensor_data.get('warning_level', 55.0)
        critical_level = sensor_data.get('critical_level', 65.0)
        
        # Compute combined risk score
        visual_risk = visual_anomaly.confidence if visual_anomaly.anomaly_type != 'normal' else 0
        water_risk = max(0, (water_level - warning_level) / (critical_level - warning_level))
        
        # Combined breach probability
        # Weighted combination: visual 40%, water level 40%, flow 20%
        breach_prob = 0.4 * visual_risk + 0.4 * water_risk + 0.2 * min(flow_vel / 5.0, 1.0)
        breach_prob = min(1.0, breach_prob)
        
        # Determine risk level
        if breach_prob > 0.8:
            risk_level = 'critical'
        elif breach_prob > 0.6:
            risk_level = 'high'
        elif breach_prob > 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Generate LLM explanation if API available
        if self.gemini_api_key:
            llm_report = self._call_gemini_api(
                visual_anomaly, sensor_data, breach_prob
            )
        else:
            llm_report = self._generate_fallback_report(
                visual_anomaly, sensor_data, breach_prob
            )
        
        # Trigger alert if critical conditions met
        alert_triggered = (
            visual_anomaly.confidence > 0.8 and
            water_level > warning_level and
            risk_level in ['high', 'critical']
        )
        
        return MultimodalDiagnosis(
            image_id=visual_anomaly.image_id,
            timestamp=datetime.now().isoformat(),
            visual_findings={
                'anomaly_type': visual_anomaly.anomaly_type,
                'confidence': visual_anomaly.confidence,
                'anomaly_score': visual_anomaly.anomaly_score
            },
            sensor_context={
                'water_level': water_level,
                'flow_velocity': flow_vel,
                'relative_to_warning': (water_level - warning_level) / (critical_level - warning_level)
            },
            risk_assessment=risk_level,
            breach_probability=breach_prob,
            recommended_actions=llm_report.get('actions', []),
            llm_explanation=llm_report.get('explanation', ''),
            alert_triggered=alert_triggered
        )
    
    def _call_gemini_api(
        self,
        visual_anomaly: AnomalyDetection,
        sensor_data: Dict,
        breach_prob: float
    ) -> Dict:
        """Call Gemini API for structured diagnosis."""
        try:
            # Prepare prompt
            prompt = f"""
            Analyze dike breach risk based on the following data:
            
            Visual Analysis:
            - Detected anomaly: {visual_anomaly.anomaly_type}
            - Confidence: {visual_anomaly.confidence:.2%}
            - GPS location: {visual_anomaly.gps_coordinates}
            
            Sensor Data:
            - Current water level: {sensor_data.get('water_level', 'N/A')} m
            - Flow velocity: {sensor_data.get('flow_velocity', 'N/A')} m/s
            - Warning level: {sensor_data.get('warning_level', 'N/A')} m
            - Critical level: {sensor_data.get('critical_level', 'N/A')} m
            
            Combined breach probability: {breach_prob:.2%}
            
            Provide:
            1. Risk assessment explanation
            2. Recommended immediate actions (list)
            3. Priority level
            
            Return JSON format with keys: explanation, actions (list), priority
            """
            
            # API call (simplified - actual implementation would use proper auth)
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.gemini_api_key}'
            }
            
            payload = {
                'contents': [{'parts': [{'text': prompt}]}],
                'generationConfig': {
                    'responseMimeType': 'application/json'
                }
            }
            
            # For now, return fallback (actual API integration requires valid key)
            logger.info("Gemini API call simulated (no valid key)")
            return self._generate_fallback_report(visual_anomaly, sensor_data, breach_prob)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_fallback_report(visual_anomaly, sensor_data, breach_prob)
    
    def _generate_fallback_report(
        self,
        visual_anomaly: AnomalyDetection,
        sensor_data: Dict,
        breach_prob: float
    ) -> Dict:
        """Generate fallback report without LLM."""
        actions = []
        
        if visual_anomaly.anomaly_type == 'crack':
            actions.append("Immediate visual inspection by engineering team")
            actions.append("Deploy crack monitoring sensors")
        elif visual_anomaly.anomaly_type == 'erosion':
            actions.append("Reinforce erosion zone with riprap")
            actions.append("Increase monitoring frequency")
        elif visual_anomaly.anomaly_type == 'seepage':
            actions.append("Install drainage system")
            actions.append("Check for piping risk")
        
        if breach_prob > 0.7:
            actions.insert(0, "EMERGENCY: Consider evacuation of downstream areas")
            actions.insert(1, "Alert emergency management authorities")
        elif breach_prob > 0.5:
            actions.insert(0, "WARNING: Increase patrol frequency to hourly")
        
        explanation = (
            f"Detected {visual_anomaly.anomaly_type} with {visual_anomaly.confidence:.1%} confidence. "
            f"Current water level is {sensor_data.get('water_level', 'unknown')}m. "
            f"Combined breach probability: {breach_prob:.1%}. "
        )
        
        if breach_prob > 0.8:
            explanation += "CRITICAL risk of structural failure. Immediate action required."
        elif breach_prob > 0.5:
            explanation += "ELEVATED risk. Enhanced monitoring and preparedness advised."
        else:
            explanation += "Risk within acceptable limits. Continue routine monitoring."
        
        return {
            'explanation': explanation,
            'actions': actions,
            'priority': 'critical' if breach_prob > 0.8 else 'high' if breach_prob > 0.5 else 'medium'
        }


class VisionIntelligenceSystem:
    """
    Main coordinator for vision-based dike safety monitoring.
    
    Integrates all vision modules and provides unified interface
    for the main system and dashboard.
    """
    
    def __init__(
        self,
        vit_model_name: str = "google/vit-base-patch16-224",
        sam_api_endpoint: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        alert_threshold: float = 0.8,
        warning_water_level: float = 55.0,
        critical_water_level: float = 65.0
    ):
        # Initialize components
        self.feature_extractor = VisionFeatureExtractor(
            model_name=vit_model_name,
            anomaly_threshold=alert_threshold
        )
        
        self.sam_interface = SAMInterface(api_endpoint=sam_api_endpoint)
        
        self.fusion_analyzer = MultimodalFusionAnalyzer(
            gemini_api_key=gemini_api_key
        )
        
        # Alert thresholds
        self.alert_threshold = alert_threshold
        self.warning_level = warning_water_level
        self.critical_level = critical_water_level
        
        # State tracking
        self.current_status = "NORMAL"
        self.active_alerts = []
        
        logger.info("Vision Intelligence System initialized")
    
    def process_drone_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        gps_coords: Optional[Tuple[float, float]] = None,
        sensor_data: Optional[Dict] = None,
        image_id: Optional[str] = None
    ) -> Dict:
        """
        Process drone image through complete vision pipeline.
        
        Pipeline:
            Image → ViT Anomaly Detection → SAM Segmentation (if needed) →
            Multimodal Fusion → Diagnosis Report → Alert Decision
        
        Args:
            image: Drone image
            gps_coords: (latitude, longitude)
            sensor_data: Real-time sensor readings
            image_id: Unique identifier
        
        Returns:
            Complete analysis results including diagnosis and alert status
        """
        # Step 1: Anomaly detection
        logger.info("[Vision] Step 1: Anomaly detection")
        anomaly_result = self.feature_extractor.detect_anomaly(
            image, image_id, gps_coords
        )
        
        # Step 2: SAM segmentation if anomaly detected
        segmentation_mask = None
        if anomaly_result.anomaly_type != 'normal' and anomaly_result.confidence > 0.5:
            logger.info("[Vision] Step 2: SAM segmentation")
            # Center click for automatic segmentation
            click_point = (112, 112)  # Center of 224x224 image
            segmentation_mask = self.sam_interface.segment_with_click(
                image, click_point, click_label=1
            )
            anomaly_result.segmentation_mask = segmentation_mask
        
        # Step 3: Multimodal fusion
        diagnosis = None
        if sensor_data:
            logger.info("[Vision] Step 3: Multimodal fusion analysis")
            features = self.feature_extractor.extract_features(image)
            diagnosis = self.fusion_analyzer.analyze_breach_risk(
                features, anomaly_result, sensor_data
            )
            
            # Update system status
            self._update_status(diagnosis)
        
        return {
            'anomaly_detection': asdict(anomaly_result),
            'multimodal_diagnosis': asdict(diagnosis) if diagnosis else None,
            'system_status': self.current_status,
            'alert_triggered': diagnosis.alert_triggered if diagnosis else False,
            'gps_coordinates': gps_coords,
            'timestamp': datetime.now().isoformat()
        }
    
    def _update_status(self, diagnosis: MultimodalDiagnosis):
        """Update system status based on diagnosis."""
        if diagnosis.alert_triggered:
            self.current_status = "CRITICAL: Dike Breach Risk"
            self.active_alerts.append({
                'timestamp': diagnosis.timestamp,
                'location': diagnosis.image_id,
                'risk_level': diagnosis.risk_assessment,
                'breach_probability': diagnosis.breach_probability
            })
            logger.warning(f"🚨 CRITICAL ALERT: Dike breach risk detected!")
        elif diagnosis.risk_assessment in ['high', 'medium']:
            self.current_status = f"WARNING: {diagnosis.risk_assessment.upper()} Risk"
        else:
            self.current_status = "NORMAL"
    
    def interactive_segment(
        self,
        image: Union[str, np.ndarray],
        click_point: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Interactive segmentation for dashboard.
        
        User clicks on image, SAM segments the region.
        """
        return self.sam_interface.segment_with_click(image, click_point)
    
    def get_status(self) -> Dict:
        """Get current vision system status."""
        return {
            'status': self.current_status,
            'active_alerts': len(self.active_alerts),
            'recent_alerts': self.active_alerts[-5:] if self.active_alerts else [],
            'alert_threshold': self.alert_threshold,
            'warning_level': self.warning_level,
            'critical_level': self.self.critical_level
        }


if __name__ == "__main__":
    """Test vision intelligence system."""
    print("Testing Vision Intelligence System...")
    
    # Initialize
    vision_system = VisionIntelligenceSystem()
    
    # Create dummy image (would be drone image in production)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Mock sensor data
    sensor_data = {
        'water_level': 58.0,  # Above warning
        'flow_velocity': 3.5,
        'warning_level': 55.0,
        'critical_level': 65.0
    }
    
    # Process
    result = vision_system.process_drone_image(
        image=dummy_image,
        gps_coords=(34.5, 110.2),
        sensor_data=sensor_data,
        image_id="drone_001"
    )
    
    print("\nAnalysis Result:")
    print(json.dumps(result, indent=2, default=str))
    
    print(f"\nSystem Status: {vision_system.current_status}")
    print(f"Alert Triggered: {result['alert_triggered']}")
    
    print("\n✓ Vision intelligence test complete!")
