import cv2
import numpy as np
import time
from datetime import datetime

class AgeGenderDetector:
    def __init__(self):
        # Use multiple face detection methods
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.frontalface_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.frontalface_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Age ranges
        self.age_ranges = {
            'child': (0, 12, '👶 Child', '(0-12)'),
            'teen': (13, 19, '🧒 Teen', '(13-19)'),
            'young': (20, 30, '👨‍🎓 Young Adult', '(20-30)'),
            'adult': (31, 45, '👨‍💼 Adult', '(31-45)'),
            'middle': (46, 60, '👨 Middle Age', '(46-60)'),
            'senior': (61, 100, '👴 Senior', '(61-100)')
        }
        
        print("✅ Enhanced Age & Gender Detection System Ready!")
    
    def detect_faces_robust(self, image):
        """Robust face detection using multiple cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance image for better detection
        gray = cv2.equalizeHist(gray)  # Improve contrast
        gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduce noise
        
        all_faces = []
        
        # Try different cascades with different parameters
        detectors = [
            (self.face_cascade, 1.05, 5),  # Default
            (self.frontalface_alt, 1.1, 3),  # More sensitive
            (self.frontalface_alt2, 1.08, 4),  # Balanced
            (self.profile_cascade, 1.1, 5)  # Profile faces
        ]
        
        for cascade, scale_factor, min_neighbors in detectors:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(30, 30),  # Smaller minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                all_faces.extend(faces)
        
        # Remove duplicates (simple overlap check)
        unique_faces = []
        for face in all_faces:
            x, y, w, h = face
            is_duplicate = False
            for ux, uy, uw, uh in unique_faces:
                if abs(x - ux) < 20 and abs(y - uy) < 20:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def analyze_age(self, face_roi):
        """Analyze age based on facial features"""
        h, w = face_roi.shape[:2]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate various age indicators
        texture = np.std(gray)
        edges = cv2.Canny(gray, 50, 150)
        wrinkle_density = np.sum(edges > 0) / edges.size
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        smoothness = np.std(laplacian)
        
        # Score based age
        age_score = (texture * 0.3) + (wrinkle_density * 100) + (smoothness * 0.2)
        
        # Determine age group
        if age_score < 25:
            age_group = self.age_ranges['child']
        elif age_score < 35:
            age_group = self.age_ranges['teen']
        elif age_score < 45:
            age_group = self.age_ranges['young']
        elif age_score < 55:
            age_group = self.age_ranges['adult']
        elif age_score < 70:
            age_group = self.age_ranges['middle']
        else:
            age_group = self.age_ranges['senior']
        
        confidence = min(85, max(50, int(100 - (age_score / 2))))
        
        return age_group, confidence
    
    def analyze_gender(self, face_roi):
        """Analyze gender based on facial features"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        edges = cv2.Canny(gray, 50, 150)
        jaw_region = edges[int(h*0.6):h, int(w*0.2):int(w*0.8)]
        jaw_strength = np.sum(jaw_region > 0) / jaw_region.size if jaw_region.size > 0 else 0
        
        left_side = gray[:, :w//2]
        right_side = cv2.flip(gray[:, w//2:], 1)
        min_w = min(left_side.shape[1], right_side.shape[1])
        left_side = left_side[:, :min_w]
        right_side = right_side[:, :min_w]
        symmetry = 1 - (np.mean(np.abs(left_side.astype(float) - right_side.astype(float))) / 255)
        
        eye_region = gray[int(h*0.25):int(h*0.45), int(w*0.2):int(w*0.8)]
        eye_intensity = np.mean(eye_region) if eye_region.size > 0 else 128
        
        male_score = ((1 - jaw_strength) * 60) + (symmetry * 30) + (eye_intensity / 255 * 10)
        female_score = (jaw_strength * 60) + ((1 - symmetry) * 30) + ((255 - eye_intensity) / 255 * 10)
        
        if male_score > female_score:
            gender = "Male 👨"
            confidence = int(male_score)
        else:
            gender = "Female 👩"
            confidence = int(female_score)
        
        confidence = min(90, max(55, confidence))
        return gender, confidence
    
    def detect_emotion(self, face_roi):
        """Detect emotion from face"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        smiles = smile_cascade.detectMultiScale(gray, 1.7, 20)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 10)
        
        if len(smiles) > 0:
            if len(eyes) >= 2:
                return "HAPPY 😊"
            else:
                return "SMILING 🙂"
        elif len(eyes) < 2:
            return "TIRED 😴"
        else:
            return "NEUTRAL 😐"
    
    def process_image(self, image_path):
        """Process single image with robust face detection"""
        # Read image with different methods
        img = cv2.imread(image_path)
        
        if img is None:
            # Try reading with different flags
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"❌ Error: Could not load {image_path}")
            print("💡 Tips:")
            print("   - Check if file exists")
            print("   - Use full path (e.g., C:/Users/.../image.jpg)")
            print("   - Try a different image format (JPG, PNG)")
            return
        
        # Make a copy for display
        display_img = img.copy()
        
        # Try multiple face detection methods
        print("\n🔍 Attempting face detection...")
        
        # Method 1: Robust detection
        faces = self.detect_faces_robust(img)
        
        if len(faces) == 0:
            # Method 2: Try with image preprocessing
            print("   Method 1 failed, trying image enhancement...")
            
            # Enhance image
            enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
            faces = self.detect_faces_robust(enhanced)
            
            if len(faces) > 0:
                display_img = enhanced
                print("   ✓ Face detected after enhancement!")
        
        if len(faces) == 0:
            # Method 3: Try different color spaces
            print("   Method 2 failed, trying different color spaces...")
            
            for color_space in ['HSV', 'YCrCb']:
                if color_space == 'HSV':
                    converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    converted = cv2.cvtColor(converted, cv2.COLOR_HSV2BGR)
                else:
                    converted = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                    converted = cv2.cvtColor(converted, cv2.COLOR_YCrCb2BGR)
                
                faces = self.detect_faces_robust(converted)
                if len(faces) > 0:
                    display_img = converted
                    print(f"   ✓ Face detected in {color_space} color space!")
                    break
        
        print(f"\n{'='*60}")
        print(f"📊 ANALYSIS RESULTS - Found {len(faces)} face(s)")
        print(f"{'='*60}")
        
        if len(faces) == 0:
            print("\n⚠️ No faces detected in the image!")
            print("\n💡 Suggestions:")
            print("   1. Ensure the image contains a clear frontal face")
            print("   2. Try an image with better lighting")
            print("   3. Make sure the face isn't too small or too large")
            print("   4. Try a different image file")
            print("\n📁 Sample images you can test:")
            print("   - image2.jpg")
            print("   - image3.jpg")
            print("   - team1.jpg")
            print("   - team.webp")
            
            # Save original image with message
            output_path = f"age_gender_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(output_path, img)
            print(f"\n💾 Original image saved: {output_path}")
            return
        
        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Add padding
            padding = int(min(w, h) * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(display_img.shape[1], x + w + padding)
            y2 = min(display_img.shape[0], y + h + padding)
            
            face_roi = display_img[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                continue
            
            # Analyze
            age_group, age_conf = self.analyze_age(face_roi)
            gender, gender_conf = self.analyze_gender(face_roi)
            emotion = self.detect_emotion(face_roi)
            
            age_display, age_emoji, age_range = age_group[2], age_group[0], age_group[3]
            
            # Color based on gender
            box_color = (255, 165, 0) if "Male" in gender else (255, 105, 180)
            
            # Draw rectangle
            cv2.rectangle(display_img, (x1, y1), (x2, y2), box_color, 3)
            
            # Background for text
            overlay = display_img.copy()
            text_height = 100
            cv2.rectangle(overlay, (x1, y1-text_height), (x2, y1), box_color, -1)
            display_img = cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0)
            
            # Display info
            y_offset = y1 - 10
            info_lines = [
                f"🎭 {gender} ({gender_conf}%)",
                f"📅 Age: {age_display} {age_emoji}",
                f"📊 Range: {age_range}",
                f"💭 Emotion: {emotion}"
            ]
            
            for idx, line in enumerate(info_lines):
                text_y = y_offset - (idx * 22)
                if text_y > 20:
                    cv2.putText(display_img, line, (x1+8, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            
            # Face number
            cv2.circle(display_img, (x1+25, y1+30), 18, box_color, -1)
            cv2.putText(display_img, str(i+1), (x1+17, y1+38), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Print results
            print(f"\n👤 Face {i+1}:")
            print(f"   {gender} (Confidence: {gender_conf}%)")
            print(f"   Age: {age_display} {age_emoji} ({age_range})")
            print(f"   Emotion: {emotion}")
        
        # Save result
        output_path = f"age_gender_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(output_path, display_img)
        print(f"\n💾 Result saved: {output_path}")
        
        # Display image
        cv2.imshow("Age + Gender Detection", display_img)
        print("\n📌 Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def process_webcam(self):
        """Real-time detection"""
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*60)
        print("🎭 REAL-TIME AGE + GENDER DETECTION")
        print("="*60)
        print("Press 'q' - Quit")
        print("Press 's' - Save Screenshot")
        print("-"*60)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces_robust(frame)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size == 0:
                    continue
                
                if frame_count % 5 == 0:
                    age_group, _ = self.analyze_age(face_roi)
                    gender, gender_conf = self.analyze_gender(face_roi)
                    emotion = self.detect_emotion(face_roi)
                    age_display, age_emoji, _ = age_group[2], age_group[0], age_group[3]
                
                box_color = (255, 165, 0) if "Male" in gender else (255, 105, 180)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                info = f"{gender.split()[0]} | {age_display} {age_emoji}"
                cv2.putText(frame, info, (x, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                cv2.putText(frame, f"{emotion}", (x, y-35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Age + Gender Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"capture_{timestamp}.jpg", frame)
                print("📸 Screenshot saved!")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = AgeGenderDetector()
    
    while True:
        print("\n" + "="*50)
        print("🎭 AGE + GENDER DETECTION SYSTEM")
        print("="*50)
        print("1. 📷 Analyze Image")
        print("2. 🎥 Real-time Webcam")
        print("3. ❌ Exit")
        print("-"*50)
        
        choice = input("\nSelect option (1-3): ")
        
        if choice == '1':
            path = input("Enter image path: ").strip()
            # Remove quotes if present
            path = path.strip('"').strip("'")
            detector.process_image(path)
        elif choice == '2':
            detector.process_webcam()
        elif choice == '3':
            print("\n👋 Thank you!")
            break
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()