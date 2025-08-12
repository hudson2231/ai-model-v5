import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from cog import BasePredictor, Input, Path
import warnings
warnings.filterwarnings('ignore')


class ColorbookLineArtProcessor:
    """Specialized processor for clean coloring book style line art"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def preprocess_image(self, image: Image.Image, target_size: int = 1024) -> np.ndarray:
        """Optimized preprocessing for line art conversion"""
        # Resize maintaining aspect ratio
        w, h = image.size
        if max(w, h) > target_size:
            ratio = target_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance for better edge detection
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=5))
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        
        return np.array(image)
    
    def create_base_edges(self, img: np.ndarray) -> np.ndarray:
        """Create clean base edges using optimized parameters"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Strong bilateral filter to reduce noise while keeping strong edges
        filtered = cv2.bilateralFilter(gray, 20, 80, 80)
        
        # Use adaptive thresholding to find major regions
        adaptive = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours from adaptive threshold
        contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create clean contour-based edges
        edges = np.zeros_like(gray)
        
        # Draw contours with appropriate thickness
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter small noise
                # Approximate contour to reduce noise
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(edges, [approx], -1, 255, 2)
        
        return edges
    
    def enhance_with_canny(self, img: np.ndarray, base_edges: np.ndarray) -> np.ndarray:
        """Add fine details using optimized Canny edge detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply strong noise reduction
        filtered = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        # Use moderate Canny thresholds for clean lines
        canny_edges = cv2.Canny(filtered, 100, 200)
        
        # Clean up Canny edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        canny_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)
        canny_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_OPEN, kernel)
        
        # Combine with base edges
        combined = cv2.bitwise_or(base_edges, canny_edges)
        
        return combined
    
    def create_smooth_lines(self, edges: np.ndarray) -> np.ndarray:
        """Convert edges to smooth, connected lines"""
        
        # Step 1: Dilate slightly to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Step 2: Apply closing to connect line segments
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel)
        
        # Step 3: Thin back down to single pixel lines
        # But first, let's work with contours for smoother results
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create clean line art
        line_art = np.zeros_like(edges)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 20:  # Filter very small contours
                # Smooth the contour
                epsilon = 0.01 * cv2.arcLength(contour, True)
                smoothed = cv2.approxPolyDP(contour, epsilon, True)
                
                # Draw smoothed contour
                cv2.drawContours(line_art, [smoothed], -1, 255, 1)
        
        return line_art
    
    def enhance_facial_features(self, img: np.ndarray, line_art: np.ndarray) -> np.ndarray:
        """Add enhanced detail for faces and important features"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Try face detection
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            
            enhanced = line_art.copy()
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_region = gray[y:y+h, x:x+w]
                
                # Apply gentle edge detection to face
                face_blur = cv2.GaussianBlur(face_region, (3, 3), 1)
                face_edges = cv2.Canny(face_blur, 50, 100)
                
                # Clean up face edges
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                face_edges = cv2.morphologyEx(face_edges, cv2.MORPH_CLOSE, kernel)
                
                # Add to main line art
                enhanced[y:y+h, x:x+w] = cv2.bitwise_or(enhanced[y:y+h, x:x+w], face_edges)
            
            return enhanced
            
        except Exception:
            # Fallback: enhance high-detail areas
            # Use corner detection to find important features
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            
            enhanced = line_art.copy()
            
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel().astype(int)
                    # Add small detail circles around important features
                    cv2.circle(enhanced, (x, y), 2, 255, 1)
            
            return enhanced
    
    def final_cleanup(self, line_art: np.ndarray) -> np.ndarray:
        """Final cleanup for professional coloring book appearance"""
        
        # Ensure black lines on white background
        cleaned = line_art.copy()
        
        # If most pixels are black (inverted), flip it
        if np.mean(cleaned) < 127:
            cleaned = 255 - cleaned
        
        # Apply threshold to ensure pure black/white
        _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
        
        # Remove very small isolated pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Slight closing to ensure lines are connected
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Final inversion check - we want BLACK lines on WHITE background
        # Count edge pixels - if more white pixels on edges, we need to invert
        edges_sample = cleaned[0:10, :].flatten().tolist() + cleaned[-10:, :].flatten().tolist() + \
                      cleaned[:, 0:10].flatten().tolist() + cleaned[:, -10:].flatten().tolist()
        
        white_edge_pixels = sum(1 for p in edges_sample if p > 127)
        
        if white_edge_pixels < len(edges_sample) * 0.7:  # If edges aren't mostly white, invert
            cleaned = 255 - cleaned
        
        return cleaned
    
    def process_image(self, image: Image.Image) -> Image.Image:
        """Main processing pipeline optimized for coloring book line art"""
        
        print("🎨 Starting coloring book line art conversion...")
        
        # Step 1: Preprocess
        print("📸 Preprocessing image...")
        img_array = self.preprocess_image(image)
        
        # Step 2: Create base edges using contour detection
        print("🔍 Creating base contour edges...")
        base_edges = self.create_base_edges(img_array)
        
        # Step 3: Enhance with Canny details
        print("✨ Adding fine details...")
        enhanced_edges = self.enhance_with_canny(img_array, base_edges)
        
        # Step 4: Create smooth, connected lines
        print("🎯 Creating smooth line art...")
        smooth_lines = self.create_smooth_lines(enhanced_edges)
        
        # Step 5: Enhance facial features
        print("👤 Enhancing facial features...")
        enhanced_lines = self.enhance_facial_features(img_array, smooth_lines)
        
        # Step 6: Final cleanup
        print("🛠️ Final cleanup...")
        final_result = self.final_cleanup(enhanced_lines)
        
        print("✅ Coloring book line art complete!")
        
        return Image.fromarray(final_result)


class Predictor(BasePredictor):
    def setup(self):
        """Initialize the predictor"""
        print("🚀 Setting up Coloring Book Line Art Processor...")
        self.processor = ColorbookLineArtProcessor()
        print("✅ Setup complete!")
    
    def predict(
        self,
        input_image: Path = Input(description="Photo to convert into coloring book line art"),
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
        line_thickness: str = Input(
            description="Line thickness preference",
            default="medium",
            choices=["thin", "medium", "thick"]
        ),
    ) -> Path:
        """Convert image to coloring book style line art"""
        
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
        result = self.processor.process_image(image)
        
        # Apply line thickness adjustments
        result_array = np.array(result)
        
        if line_thickness == "thin":
            # Erode to make lines thinner
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            result_array = cv2.erode(result_array, kernel, iterations=1)
            # Ensure we still have black lines on white background
            if np.mean(result_array) < 127:
                result_array = 255 - result_array
        elif line_thickness == "thick":
            # Dilate to make lines thicker
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # First make sure we have white lines on black background for dilation
            if np.mean(result_array) > 127:
                result_array = 255 - result_array
            result_array = cv2.dilate(result_array, kernel, iterations=1)
            # Then convert back to black lines on white background
            result_array = 255 - result_array
        
        result = Image.fromarray(result_array)
        
        # Resize if needed
        if max(result.size) != target_size:
            w, h = result.size
            if max(w, h) > target_size:
                ratio = target_size / max(w, h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                result = result.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        print(f"📤 Final image size: {result.size}")
        
        # Save result
        output_path = "/tmp/coloring_book_line_art.png"
        result.save(output_path, "PNG", optimize=False)
        
        print(f"💾 Coloring book line art saved to: {output_path}")
        return Path(output_path)
