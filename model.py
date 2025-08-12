import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from cog import BasePredictor, Input, Path
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage import morphology, feature, filters
from skimage.segmentation import felzenszwalb
import warnings
warnings.filterwarnings('ignore')


class AdvancedLineArtProcessor:
    """Advanced image processing pipeline for high-quality line art conversion"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def preprocess_image(self, image: Image.Image, target_size: int = 1024) -> np.ndarray:
        """Enhanced preprocessing with noise reduction and contrast optimization"""
        # Resize while maintaining aspect ratio
        w, h = image.size
        if max(w, h) > target_size:
            ratio = target_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply slight sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        return np.array(image)
    
    def advanced_edge_detection(self, img: np.ndarray) -> np.ndarray:
        """Multi-algorithm edge detection with intelligent fusion"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Multiple edge detection approaches
        edges_list = []
        
        # 1. Adaptive Canny with automatic thresholds
        sigma = 0.33
        median_val = np.median(filtered)
        lower = int(max(0, (1.0 - sigma) * median_val))
        upper = int(min(255, (1.0 + sigma) * median_val))
        canny1 = cv2.Canny(filtered, lower, upper)
        edges_list.append(canny1)
        
        # 2. Standard Canny with optimized thresholds
        canny2 = cv2.Canny(filtered, 50, 150)
        edges_list.append(canny2)
        
        # 3. Laplacian edge detection
        laplacian = cv2.Laplacian(filtered, cv2.CV_64F)
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(laplacian)
        _, laplacian = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)
        edges_list.append(laplacian)
        
        # 4. Sobel edge detection
        sobelx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel)
        _, sobel = cv2.threshold(sobel, 30, 255, cv2.THRESH_BINARY)
        edges_list.append(sobel)
        
        # Combine all edge maps with weighted fusion
        weights = [0.4, 0.3, 0.15, 0.15]  # Favor adaptive canny
        combined = np.zeros_like(edges_list[0], dtype=np.float32)
        
        for edge_map, weight in zip(edges_list, weights):
            combined += edge_map.astype(np.float32) * weight
        
        # Normalize and threshold
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        _, combined = cv2.threshold(combined, 120, 255, cv2.THRESH_BINARY)
        
        return combined
    
    def detect_faces_and_features(self, img: np.ndarray) -> np.ndarray:
        """Enhanced feature detection for better facial feature preservation"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Try to load face cascade (optional - graceful fallback)
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            
            feature_mask = np.zeros_like(gray)
            
            for (x, y, w, h) in faces:
                # Create a mask for the face region
                face_region = gray[y:y+h, x:x+w]
                
                # Enhanced edge detection in face region
                face_edges = cv2.Canny(face_region, 30, 80)
                
                # Dilate to strengthen facial features
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                face_edges = cv2.dilate(face_edges, kernel, iterations=1)
                
                feature_mask[y:y+h, x:x+w] = face_edges
            
            return feature_mask
            
        except Exception:
            # Fallback: use corner detection for important features
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
            feature_mask = np.zeros_like(gray)
            
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel().astype(int)
                    cv2.circle(feature_mask, (x, y), 2, 255, -1)
            
            return feature_mask
    
    def enhance_lines(self, edges: np.ndarray) -> np.ndarray:
        """Advanced line enhancement and cleanup"""
        # Morphological operations to clean up lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Close small gaps
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # Skeletonize to get clean single-pixel lines
        skeleton = morphology.skeletonize(opened > 0)
        skeleton = (skeleton * 255).astype(np.uint8)
        
        # Slightly dilate the skeleton to make lines more visible
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        enhanced = cv2.dilate(skeleton, dilate_kernel, iterations=1)
        
        return enhanced
    
    def create_artistic_lines(self, img: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Create artistic line weights and styles"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Create different line weights based on image content
        # Thicker lines for major contours, thinner for details
        
        # Find major contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create line weight map
        line_weights = np.zeros_like(edges)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Major contours
                cv2.drawContours(line_weights, [contour], -1, 255, 2)
            elif area > 20:  # Medium contours
                cv2.drawContours(line_weights, [contour], -1, 180, 1)
        
        # Combine with original edges
        result = np.maximum(edges, line_weights)
        
        return result
    
    def post_process(self, line_art: np.ndarray) -> np.ndarray:
        """Final post-processing for clean, printable line art"""
        # Ensure pure black lines on white background
        line_art = 255 - line_art  # Invert if needed
        
        # Apply final threshold to ensure pure black/white
        _, line_art = cv2.threshold(line_art, 127, 255, cv2.THRESH_BINARY)
        
        # Final cleanup - remove very small isolated pixels
        kernel = np.ones((2, 2), np.uint8)
        line_art = cv2.morphologyEx(line_art, cv2.MORPH_OPEN, kernel)
        
        # Ensure white background
        line_art = 255 - line_art
        
        return line_art
    
    def process_image(self, image: Image.Image, enhance_faces: bool = True) -> Image.Image:
        """Main processing pipeline"""
        print("🎨 Starting advanced line art processing...")
        
        # Step 1: Preprocess
        print("📸 Preprocessing image...")
        img_array = self.preprocess_image(image)
        
        # Step 2: Advanced edge detection
        print("🔍 Detecting edges with multi-algorithm fusion...")
        edges = self.advanced_edge_detection(img_array)
        
        # Step 3: Enhance facial features if requested
        if enhance_faces:
            print("👤 Enhancing facial features...")
            face_features = self.detect_faces_and_features(img_array)
            # Combine face features with edges
            edges = np.maximum(edges, face_features)
        
        # Step 4: Enhance lines
        print("✨ Enhancing line quality...")
        enhanced_edges = self.enhance_lines(edges)
        
        # Step 5: Create artistic line weights
        print("🎯 Creating artistic line weights...")
        artistic_lines = self.create_artistic_lines(img_array, enhanced_edges)
        
        # Step 6: Post-process
        print("🛠️ Final post-processing...")
        final_result = self.post_process(artistic_lines)
        
        print("✅ Line art conversion complete!")
        
        # Convert back to PIL Image
        return Image.fromarray(final_result)


class Predictor(BasePredictor):
    def setup(self):
        """Initialize the predictor"""
        print("🚀 Setting up Advanced Line Art Processor...")
        self.processor = AdvancedLineArtProcessor()
        print("✅ Setup complete!")
    
    def predict(
        self,
        input_image: Path = Input(description="Photo to convert into high-quality line art"),
        target_size: int = Input(
            description="Maximum size for the output (maintains aspect ratio)", 
            default=1024, 
            ge=512, 
            le=2048
        ),
        enhance_faces: bool = Input(
            description="Apply enhanced processing for facial features", 
            default=True
        ),
        line_intensity: str = Input(
            description="Line art intensity level",
            default="balanced",
            choices=["light", "balanced", "strong"]
        ),
    ) -> Path:
        """Convert image to high-quality line art"""
        
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
            enhance_faces=enhance_faces
        )
        
        # Adjust intensity based on user preference
        if line_intensity == "light":
            # Make lines slightly thinner
            result_array = np.array(result)
            kernel = np.ones((2, 2), np.uint8)
            result_array = cv2.erode(result_array, kernel, iterations=1)
            result = Image.fromarray(result_array)
        elif line_intensity == "strong":
            # Make lines slightly thicker
            result_array = np.array(result)
            kernel = np.ones((2, 2), np.uint8)
            result_array = cv2.dilate(result_array, kernel, iterations=1)
            result = Image.fromarray(result_array)
        
        # Ensure the image is the right size
        if max(result.size) != target_size:
            w, h = result.size
            if max(w, h) > target_size:
                ratio = target_size / max(w, h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                result = result.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        print(f"📤 Final image size: {result.size}")
        
        # Save result
        output_path = "/tmp/line_art_output.png"
        result.save(output_path, "PNG", optimize=False, quality=100)
        
        print(f"💾 Saved result to: {output_path}")
        return Path(output_path)
