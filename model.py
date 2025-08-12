import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from cog import BasePredictor, Input, Path
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import morphology, feature, filters, segmentation
from skimage.morphology import disk, closing, opening, erosion, dilation
from skimage.filters import gaussian, sobel, scharr, prewitt
from skimage.feature import canny
import warnings
warnings.filterwarnings('ignore')


class ProfessionalLineArtProcessor:
    """Professional-grade line art conversion matching high-quality coloring book standards"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def preprocess_image(self, image: Image.Image, target_size: int = 1024) -> np.ndarray:
        """Enhanced preprocessing optimized for line art conversion"""
        # Resize while maintaining aspect ratio
        w, h = image.size
        if max(w, h) > target_size:
            ratio = target_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            # Use LANCZOS for high quality downsampling
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance image quality before processing
        # Slight sharpening to enhance edges
        image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
        
        # Enhance contrast to make features more prominent
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Slight brightness adjustment for better edge detection
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.05)
        
        return np.array(image)
    
    def create_detail_preserving_edges(self, img: np.ndarray) -> np.ndarray:
        """Advanced edge detection that preserves facial features and fine details"""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 15, 80, 80)
        
        # Multi-scale edge detection
        edges_multi_scale = []
        
        # Scale 1: Fine details
        sigma1 = 0.5
        edges1 = canny(gaussian(filtered, sigma=sigma1), low_threshold=0.1, high_threshold=0.2)
        edges_multi_scale.append(edges1.astype(np.uint8) * 255)
        
        # Scale 2: Medium features
        sigma2 = 1.0
        edges2 = canny(gaussian(filtered, sigma=sigma2), low_threshold=0.08, high_threshold=0.15)
        edges_multi_scale.append(edges2.astype(np.uint8) * 255)
        
        # Scale 3: Major contours
        sigma3 = 2.0
        edges3 = canny(gaussian(filtered, sigma=sigma3), low_threshold=0.05, high_threshold=0.1)
        edges_multi_scale.append(edges3.astype(np.uint8) * 255)
        
        # Combine multi-scale edges with different weights
        combined_edges = np.zeros_like(edges1, dtype=np.float32)
        weights = [0.5, 0.3, 0.2]  # Favor fine details
        
        for edges, weight in zip(edges_multi_scale, weights):
            combined_edges += edges.astype(np.float32) * weight
        
        # Normalize and convert back to uint8
        combined_edges = np.clip(combined_edges, 0, 255).astype(np.uint8)
        
        return combined_edges
    
    def enhance_facial_features(self, img: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Specifically enhance facial features and important details"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        
        # Try to detect faces for enhanced processing
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            
            enhanced_edges = edges.copy()
            
            for (x, y, w, h) in faces:
                # Extract face region with padding
                padding = 20
                y1 = max(0, y - padding)
                y2 = min(gray.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(gray.shape[1], x + w + padding)
                
                face_region = gray[y1:y2, x1:x2]
                
                # Enhanced edge detection for face region
                face_filtered = cv2.bilateralFilter(face_region, 9, 75, 75)
                
                # Multiple edge detection methods for faces
                # Canny with lower thresholds for facial details
                face_canny = cv2.Canny(face_filtered, 30, 70)
                
                # Laplacian for fine facial features
                face_laplacian = cv2.Laplacian(face_filtered, cv2.CV_64F, ksize=3)
                face_laplacian = np.abs(face_laplacian)
                face_laplacian = np.clip(face_laplacian, 0, 255).astype(np.uint8)
                _, face_laplacian = cv2.threshold(face_laplacian, 20, 255, cv2.THRESH_BINARY)
                
                # Combine face edges
                face_edges = cv2.bitwise_or(face_canny, face_laplacian)
                
                # Clean up face edges
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                face_edges = cv2.morphologyEx(face_edges, cv2.MORPH_CLOSE, kernel)
                
                # Apply back to main edges
                enhanced_edges[y1:y2, x1:x2] = np.maximum(enhanced_edges[y1:y2, x1:x2], face_edges)
            
            return enhanced_edges
            
        except Exception as e:
            print(f"Face detection unavailable, using alternative feature enhancement: {e}")
            # Fallback: enhance high-contrast regions (likely to be faces/important features)
            
            # Find high-contrast regions
            contrast_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Enhance edges in high-contrast areas
            enhanced_regions = cv2.Canny(gray, 40, 80)
            enhanced_edges = np.where(contrast_mask == 0, np.maximum(edges, enhanced_regions), edges)
            
            return enhanced_edges
    
    def create_smooth_continuous_lines(self, edges: np.ndarray) -> np.ndarray:
        """Convert fragmented edges into smooth, continuous lines"""
        
        # Start with the edges
        processed = edges.copy()
        
        # Step 1: Close small gaps in lines
        # Use different kernel sizes for different gap sizes
        close_kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small gaps first
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, close_kernel_small)
        # Then medium gaps
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, close_kernel_medium)
        
        # Step 2: Remove isolated noise pixels
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, open_kernel)
        
        # Step 3: Connect nearby line segments using contour analysis
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a new image for refined contours
        refined = np.zeros_like(processed)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 10:  # Filter out very small contours
                # For larger contours, draw them with slight thickness
                if area > 100:
                    cv2.drawContours(refined, [contour], -1, 255, 2)
                else:
                    cv2.drawContours(refined, [contour], -1, 255, 1)
        
        # Step 4: Final smoothing pass
        # Slight Gaussian blur followed by threshold to smooth jagged edges
        smoothed = cv2.GaussianBlur(refined, (3, 3), 0.5)
        _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
        
        return smoothed
    
    def create_artistic_line_weights(self, img: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Create variable line weights for more artistic appearance"""
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        
        # Find contours to determine line importance
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create canvas for artistic lines
        artistic_lines = np.zeros_like(edges)
        
        # Sort contours by area (larger contours get thicker lines)
        contours_with_area = [(cv2.contourArea(cnt), cnt) for cnt in contours]
        contours_with_area.sort(reverse=True, key=lambda x: x[0])
        
        for area, contour in contours_with_area:
            if area < 5:  # Skip very small contours
                continue
            
            # Determine line thickness based on contour area and importance
            if area > 1000:  # Major outlines
                thickness = 2
            elif area > 200:  # Medium features
                thickness = 1
            else:  # Fine details
                thickness = 1
            
            # Draw contour
            cv2.drawContours(artistic_lines, [contour], -1, 255, thickness)
        
        # Add back any fine details that might have been lost
        fine_details = cv2.bitwise_and(edges, cv2.bitwise_not(artistic_lines))
        artistic_lines = cv2.bitwise_or(artistic_lines, fine_details)
        
        return artistic_lines
    
    def final_cleanup_and_enhancement(self, line_art: np.ndarray) -> np.ndarray:
        """Final processing to ensure clean, professional line art"""
        
        # Ensure we have white background, black lines
        if np.mean(line_art) < 127:  # If background is dark
            line_art = cv2.bitwise_not(line_art)
        
        # Apply final threshold to ensure pure black and white
        _, line_art = cv2.threshold(line_art, 127, 255, cv2.THRESH_BINARY)
        
        # Remove any remaining small artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        line_art = cv2.morphologyEx(line_art, cv2.MORPH_OPEN, kernel)
        
        # Final pass to ensure lines are properly connected
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        line_art = cv2.morphologyEx(line_art, cv2.MORPH_CLOSE, kernel)
        
        return line_art
    
    def process_image(self, image: Image.Image, preserve_details: bool = True) -> Image.Image:
        """Main processing pipeline for professional line art conversion"""
        
        print("🎨 Starting professional line art conversion...")
        
        # Step 1: Preprocessing
        print("📸 Preprocessing image for optimal edge detection...")
        img_array = self.preprocess_image(image)
        
        # Step 2: Create high-quality edges with detail preservation
        print("🔍 Detecting edges with multi-scale analysis...")
        edges = self.create_detail_preserving_edges(img_array)
        
        # Step 3: Enhance facial features and important details
        if preserve_details:
            print("👤 Enhancing facial features and important details...")
            edges = self.enhance_facial_features(img_array, edges)
        
        # Step 4: Create smooth, continuous lines
        print("✨ Creating smooth, continuous line art...")
        smooth_lines = self.create_smooth_continuous_lines(edges)
        
        # Step 5: Apply artistic line weights
        print("🎯 Applying artistic line weights...")
        artistic_lines = self.create_artistic_line_weights(img_array, smooth_lines)
        
        # Step 6: Final cleanup and enhancement
        print("🛠️ Final cleanup and enhancement...")
        final_result = self.final_cleanup_and_enhancement(artistic_lines)
        
        print("✅ Professional line art conversion complete!")
        
        return Image.fromarray(final_result)


class Predictor(BasePredictor):
    def setup(self):
        """Initialize the predictor"""
        print("🚀 Setting up Professional Line Art Processor...")
        self.processor = ProfessionalLineArtProcessor()
        print("✅ Professional setup complete!")
    
    def predict(
        self,
        input_image: Path = Input(description="Photo to convert into professional line art"),
        target_size: int = Input(
            description="Maximum size for the output (maintains aspect ratio)", 
            default=1024, 
            ge=512, 
            le=2048
        ),
        preserve_details: bool = Input(
            description="Apply enhanced processing for facial features and fine details", 
            default=True
        ),
        line_style: str = Input(
            description="Line art style preference",
            default="balanced",
            choices=["fine", "balanced", "bold"]
        ),
    ) -> Path:
        """Convert image to professional-quality line art"""
        
        print(f"📥 Loading image: {input_image}")
        
        # Load and validate image
        try:
            image = Image.open(input_image)
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
        except Exception as e:
            raise ValueError(f"Could not load image: {str(e)}")
        
        print(f"📐 Original image size: {image.size}")
        
        # Process the image
        result = self.processor.process_image(
            image=image, 
            preserve_details=preserve_details
        )
        
        # Apply style adjustments based on user preference
        result_array = np.array(result)
        
        if line_style == "fine":
            # Make lines thinner and more delicate
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            result_array = cv2.erode(result_array, kernel, iterations=1)
        elif line_style == "bold":
            # Make lines thicker and more prominent
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            result_array = cv2.dilate(result_array, kernel, iterations=1)
        
        result = Image.fromarray(result_array)
        
        # Ensure proper sizing
        if max(result.size) != target_size:
            w, h = result.size
            if max(w, h) > target_size:
                ratio = target_size / max(w, h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                result = result.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        print(f"📤 Final image size: {result.size}")
        
        # Save result with maximum quality
        output_path = "/tmp/professional_line_art.png"
        result.save(output_path, "PNG", optimize=False)
        
        print(f"💾 Professional line art saved to: {output_path}")
        return Path(output_path)
