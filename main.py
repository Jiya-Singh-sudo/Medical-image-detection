"""
Medical Image Segmentation Demo - Multi-Image Processing
Histological and Microscopic Elements Detection
Batch processing multiple images with comprehensive analysis

Run this script to process multiple medical images simultaneously
"""

import numpy as np
import cv2
from PIL import Image
import os
import glob
from pathlib import Path
import json
from datetime import datetime
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ Matplotlib not available, will save results as images instead")
    MATPLOTLIB_AVAILABLE = False

try:
    from cellpose import models, utils
    CELLPOSE_AVAILABLE = True
except ImportError:
    print("⚠️ Cellpose not available, will use basic segmentation")
    CELLPOSE_AVAILABLE = False

from skimage import measure, segmentation, filters
from skimage.color import rgb2gray
import warnings
warnings.filterwarnings('ignore')

class MultiImageSegmentation:
    def __init__(self, output_dir="segmentation_results"):
        """Initialize the multi-image segmentation pipeline"""
        print("🔬 Initializing Multi-Image Segmentation Pipeline...")
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {os.path.abspath(output_dir)}")
        
        # Initialize segmentation model
        if CELLPOSE_AVAILABLE:
            try:
                self.model = models.Cellpose(gpu=False, model_type='cyto')
                self.use_cellpose = True
                print("✅ Cellpose model initialized successfully!")
            except Exception as e:
                print(f"⚠️ Cellpose initialization failed: {e}")
                print("📝 Falling back to traditional segmentation methods")
                self.use_cellpose = False
        else:
            self.use_cellpose = False
            print("📝 Using traditional segmentation methods")
        
        # Results storage
        self.batch_results = []
    
    def create_synthetic_images(self, num_images=5, size=(512, 512)):
        """Create multiple synthetic histological images for demonstration"""
        print(f"🔬 Generating {num_images} synthetic histological images...")
        
        synthetic_images = []
        
        for img_idx in range(num_images):
            np.random.seed(42 + img_idx)  # Different seed for each image
            
            # Create base tissue background with variation
            tissue = np.random.rand(size[0], size[1], 3) * 0.4 + 0.3
            
            # Vary the number of cells per image
            num_cells = np.random.randint(15, 35)
            centers = []
            radii = []
            
            for i in range(num_cells):
                center_x = np.random.randint(50, size[0]-50)
                center_y = np.random.randint(50, size[1]-50)
                radius = np.random.randint(12, 40)
                
                centers.append((center_x, center_y))
                radii.append(radius)
                
                # Create circular cell structure
                y, x = np.ogrid[:size[0], :size[1]]
                mask = (x - center_y)**2 + (y - center_x)**2 <= radius**2
                
                # Add cell cytoplasm with random color variation
                intensity_factor = np.random.uniform(0.6, 0.9)
                tissue[mask, 0] *= intensity_factor
                tissue[mask, 1] *= intensity_factor + 0.1
                tissue[mask, 2] *= intensity_factor + 0.2
                
                # Add nucleus
                nucleus_radius = radius // 2
                nucleus_mask = (x - center_y)**2 + (y - center_x)**2 <= nucleus_radius**2
                tissue[nucleus_mask] *= 0.4  # Darker nucleus
            
            # Convert to uint8 and add some noise
            synthetic_image = (tissue * 255).astype(np.uint8)
            noise = np.random.normal(0, 5, synthetic_image.shape).astype(np.int16)
            synthetic_image = np.clip(synthetic_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save synthetic image
            filename = f"synthetic_image_{img_idx+1:03d}.png"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(synthetic_image, cv2.COLOR_RGB2BGR))
            
            synthetic_images.append({
                'image': synthetic_image,
                'filename': filename,
                'filepath': filepath,
                'centers': centers,
                'radii': radii,
                'true_cell_count': len(centers)
            })
        
        print(f"✅ Generated and saved {num_images} synthetic images")
        return synthetic_images
    
    def load_images_from_directory(self, directory_path):
        """Load all images from a directory"""
        print(f"📁 Loading images from directory: {directory_path}")
        
        # Supported image extensions
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))
            image_files.extend(glob.glob(os.path.join(directory_path, ext.upper())))
        
        loaded_images = []
        for filepath in sorted(image_files):
            try:
                image = cv2.imread(filepath)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    filename = os.path.basename(filepath)
                    loaded_images.append({
                        'image': image_rgb,
                        'filename': filename,
                        'filepath': filepath,
                        'centers': None,
                        'radii': None,
                        'true_cell_count': None
                    })
                    print(f"  ✅ Loaded: {filename}")
                else:
                    print(f"  ❌ Failed to load: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"  ❌ Error loading {os.path.basename(filepath)}: {e}")
        
        print(f"📊 Successfully loaded {len(loaded_images)} images")
        return loaded_images
    
    def segment_with_cellpose(self, image):
        """Perform segmentation using Cellpose"""
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        masks, flows, styles, diams = self.model.eval(
            image_rgb, 
            diameter=None,
            channels=[0, 0],
        )
        
        unique_labels = np.unique(masks)
        num_cells = len(unique_labels) - 1
        return masks, flows, num_cells
    
    def segment_traditional(self, image):
        """Traditional segmentation method"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to separate cells from background
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create masks
        masks = np.zeros(gray.shape, dtype=np.int32)
        valid_contours = 0
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 200 and area < 5000:
                temp_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(temp_mask, [contour], 255)
                masks[temp_mask == 255] = valid_contours + 1
                valid_contours += 1
        
        return masks.astype(np.uint32), None, valid_contours
    
    def analyze_single_image(self, masks, image_info):
        """Analyze segmentation results for a single image"""
        regions = measure.regionprops(masks)
        
        if len(regions) == 0:
            return {
                'filename': image_info['filename'],
                'total_objects': 0,
                'error': 'No objects detected'
            }
        
        areas = [region.area for region in regions]
        perimeters = [region.perimeter for region in regions]
        eccentricities = [region.eccentricity for region in regions]
        centroids = [(region.centroid[1], region.centroid[0]) for region in regions]  # (x, y)
        
        analysis = {
            'filename': image_info['filename'],
            'total_objects': len(regions),
            'total_area': int(np.sum(areas)),
            'mean_area': float(np.mean(areas)),
            'std_area': float(np.std(areas)),
            'min_area': int(np.min(areas)),
            'max_area': int(np.max(areas)),
            'mean_perimeter': float(np.mean(perimeters)),
            'mean_eccentricity': float(np.mean(eccentricities)),
            'area_coverage': float(np.sum(areas) / masks.size * 100),
            'centroids': centroids,
            'individual_areas': areas,
            'true_cell_count': image_info.get('true_cell_count', 'Unknown'),
            'detection_accuracy': None
        }
        
        # Calculate accuracy if ground truth is available
        if image_info.get('true_cell_count'):
            predicted = analysis['total_objects']
            true_count = image_info['true_cell_count']
            analysis['detection_accuracy'] = float(min(predicted, true_count) / max(predicted, true_count) * 100)
        
        return analysis
    
    def save_individual_results(self, image_info, masks, analysis, image_idx):
        """Save results for individual image"""
        base_name = os.path.splitext(image_info['filename'])[0]
        
        # Create subdirectory for this image
        image_dir = os.path.join(self.output_dir, f"{base_name}_results")
        os.makedirs(image_dir, exist_ok=True)
        
        # Save original image
        original_path = os.path.join(image_dir, f"{base_name}_original.png")
        cv2.imwrite(original_path, cv2.cvtColor(image_info['image'], cv2.COLOR_RGB2BGR))
        
        # Create and save colored masks
        colored_masks = np.zeros((*masks.shape, 3), dtype=np.uint8)
        unique_labels = np.unique(masks)[1:]  # Exclude background
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128), 
            (255, 165, 0), (255, 20, 147), (0, 191, 255), (154, 205, 50)
        ]
        
        for i, label in enumerate(unique_labels):
            color = colors[i % len(colors)]
            colored_masks[masks == label] = color
        
        masks_path = os.path.join(image_dir, f"{base_name}_masks.png")
        cv2.imwrite(masks_path, cv2.cvtColor(colored_masks, cv2.COLOR_RGB2BGR))
        
        # Create overlay
        overlay = image_info['image'].copy()
        mask_uint8 = (masks > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            cv2.drawContours(overlay, [contour], -1, (255, 0, 0), 2)
        
        # Add object numbers
        for i, centroid in enumerate(analysis['centroids']):
            cv2.putText(overlay, str(i+1), (int(centroid[0]), int(centroid[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        overlay_path = os.path.join(image_dir, f"{base_name}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Save analysis as JSON
        analysis_path = os.path.join(image_dir, f"{base_name}_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return {
            'original_path': original_path,
            'masks_path': masks_path,
            'overlay_path': overlay_path,
            'analysis_path': analysis_path
        }
    
    def create_batch_summary(self):
        """Create comprehensive batch processing summary"""
        if not self.batch_results:
            print("⚠️ No results to summarize")
            return
        
        print("📊 Creating batch processing summary...")
        
        # Calculate batch statistics
        total_images = len(self.batch_results)
        total_objects_detected = sum(r['analysis']['total_objects'] for r in self.batch_results)
        successful_images = len([r for r in self.batch_results if r['analysis']['total_objects'] > 0])
        
        # Average metrics
        valid_results = [r['analysis'] for r in self.batch_results if r['analysis']['total_objects'] > 0]
        if valid_results:
            avg_objects_per_image = np.mean([r['total_objects'] for r in valid_results])
            avg_object_area = np.mean([r['mean_area'] for r in valid_results])
            avg_coverage = np.mean([r['area_coverage'] for r in valid_results])
        else:
            avg_objects_per_image = avg_object_area = avg_coverage = 0
        
        # Create summary report
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'batch_statistics': {
                'total_images_processed': total_images,
                'successful_detections': successful_images,
                'total_objects_detected': total_objects_detected,
                'average_objects_per_image': float(avg_objects_per_image),
                'average_object_area': float(avg_object_area),
                'average_coverage_percentage': float(avg_coverage)
            },
            'per_image_results': [r['analysis'] for r in self.batch_results]
        }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'batch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create visual summary if matplotlib is available
        if MATPLOTLIB_AVAILABLE and valid_results:
            self.create_batch_visualization(valid_results)
        
        # Print summary
        print("\n📊 BATCH PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total images processed: {total_images}")
        print(f"Successful detections: {successful_images}")
        print(f"Total objects detected: {total_objects_detected}")
        print(f"Average objects per image: {avg_objects_per_image:.1f}")
        print(f"Average object area: {avg_object_area:.1f} pixels")
        print(f"Average coverage: {avg_coverage:.1f}%")
        print("=" * 50)
        
        return summary
    
    def create_batch_visualization(self, valid_results):
        """Create batch visualization summary"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Batch Processing Summary - Multi-Image Analysis', fontsize=16, fontweight='bold')
        
        # 1. Objects per image
        filenames = [r['filename'] for r in valid_results]
        object_counts = [r['total_objects'] for r in valid_results]
        
        axes[0, 0].bar(range(len(filenames)), object_counts, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Objects Detected Per Image')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Number of Objects')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Area distribution
        all_areas = []
        for r in valid_results:
            all_areas.extend(r['individual_areas'])
        
        axes[0, 1].hist(all_areas, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title(f'Object Area Distribution\n(Total Objects: {len(all_areas)})')
        axes[0, 1].set_xlabel('Area (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Coverage percentage
        coverage_values = [r['area_coverage'] for r in valid_results]
        
        axes[1, 0].plot(range(len(filenames)), coverage_values, 'o-', color='orange', linewidth=2, markersize=6)
        axes[1, 0].set_title('Area Coverage Per Image')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Coverage (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Accuracy (if available)
        accuracy_results = [r for r in valid_results if r['detection_accuracy'] is not None]
        if accuracy_results:
            accuracies = [r['detection_accuracy'] for r in accuracy_results]
            acc_filenames = [r['filename'] for r in accuracy_results]
            
            axes[1, 1].bar(range(len(acc_filenames)), accuracies, color='lightcoral', alpha=0.7)
            axes[1, 1].set_title('Detection Accuracy\n(For Synthetic Images)')
            axes[1, 1].set_xlabel('Image Index')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].set_ylim(0, 100)
        else:
            axes[1, 1].text(0.5, 0.5, 'No accuracy data\navailable\n(Real images)', 
                          ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Detection Accuracy')
        
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        summary_plot_path = os.path.join(self.output_dir, 'batch_summary_visualization.png')
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Batch visualization saved: {summary_plot_path}")
        plt.show()
    
    def process_multiple_images(self, image_list=None, input_directory=None, num_synthetic=5):
        """Process multiple images - main batch processing function"""
        print("🚀 Starting Multi-Image Segmentation Processing...")
        print("=" * 60)
        
        # Determine image source
        if image_list:
            images_to_process = image_list
            print(f"📝 Processing provided list of {len(images_to_process)} images")
        elif input_directory and os.path.exists(input_directory):
            images_to_process = self.load_images_from_directory(input_directory)
            print(f"📁 Processing images from directory: {input_directory}")
        else:
            images_to_process = self.create_synthetic_images(num_synthetic)
            print(f"🔬 Processing {num_synthetic} synthetic images")
        
        if not images_to_process:
            print("❌ No images to process!")
            return []
        
        # Process each image
        for idx, image_info in enumerate(images_to_process):
            print(f"\n🔍 Processing image {idx+1}/{len(images_to_process)}: {image_info['filename']}")
            
            try:
                # Perform segmentation
                if self.use_cellpose:
                    print("  🤖 Using Cellpose segmentation...")
                    masks, flows, num_cells = self.segment_with_cellpose(image_info['image'])
                else:
                    print("  🔧 Using traditional segmentation...")
                    masks, flows, num_cells = self.segment_traditional(image_info['image'])
                
                # Analyze results
                print("  📈 Analyzing results...")
                analysis = self.analyze_single_image(masks, image_info)
                
                # Save individual results
                print("  💾 Saving results...")
                file_paths = self.save_individual_results(image_info, masks, analysis, idx)
                
                # Store in batch results
                result = {
                    'image_info': image_info,
                    'masks': masks,
                    'flows': flows,
                    'analysis': analysis,
                    'file_paths': file_paths
                }
                self.batch_results.append(result)
                
                print(f"  ✅ Completed: {num_cells} objects detected")
                
            except Exception as e:
                print(f"  ❌ Error processing {image_info['filename']}: {e}")
                # Add error result
                error_result = {
                    'image_info': image_info,
                    'masks': None,
                    'flows': None,
                    'analysis': {'filename': image_info['filename'], 'error': str(e), 'total_objects': 0},
                    'file_paths': {}
                }
                self.batch_results.append(error_result)
        
        # Create batch summary
        print(f"\n📊 Processing completed for {len(images_to_process)} images")
        summary = self.create_batch_summary()
        
        print(f"\n✅ All results saved in: {os.path.abspath(self.output_dir)}")
        return self.batch_results

def main():
    """Main function for multi-image processing demo"""
    print("🎯 Multi-Image Medical Segmentation Demo")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    print(f"NumPy version: {np.__version__}")
    print(f"OpenCV available: {cv2.__version__}")
    print(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    print(f"Cellpose available: {CELLPOSE_AVAILABLE}")
    print()
    
    # Initialize processor
    processor = MultiImageSegmentation(output_dir="multi_image_results")
    
    # Example usage options:
    print("🔧 Processing Options:")
    print("1. Synthetic images (default)")
    print("2. Images from directory")
    print("3. Custom image list")
    print()
    
    # For this demo, we'll process synthetic images
    print("🚀 Running demo with synthetic images...")
    results = processor.process_multiple_images(num_synthetic=8)
    
    if results:
        print("\n🎯 Multi-Image Demo Summary:")
        print("- Multiple synthetic histological images generated")
        print("- Batch segmentation completed")
        print("- Individual results saved for each image")
        print("- Comprehensive batch analysis created")
        print("- Visualizations and reports generated")
        print(f"- Check '{processor.output_dir}' directory for all results")
    else:
        print("\n❌ Demo failed - check error messages above")

# Example usage for different scenarios
def demo_directory_processing():
    """Example: Process all images from a directory"""
    processor = MultiImageSegmentation(output_dir="directory_results")
    
    # Replace with your actual directory path
    directory_path = "path/to/your/histology/images"
    
    if os.path.exists(directory_path):
        results = processor.process_multiple_images(input_directory=directory_path)
    else:
        print(f"Directory not found: {directory_path}")
        print("Creating synthetic images instead...")
        results = processor.process_multiple_images(num_synthetic=5)

def demo_custom_image_list():
    """Example: Process a custom list of images"""
    processor = MultiImageSegmentation(output_dir="custom_results")
    
    # Example custom image list (replace with actual paths)
    custom_images = [
        "image1.png",
        "image2.jpg", 
        "image3.tiff"
    ]
    
    # Load custom images
    image_list = []
    for img_path in custom_images:
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_list.append({
                    'image': image_rgb,
                    'filename': os.path.basename(img_path),
                    'filepath': img_path,
                    'centers': None,
                    'radii': None,
                    'true_cell_count': None
                })
    
    if image_list:
        results = processor.process_multiple_images(image_list=image_list)
    else:
        print("No valid images found, using synthetic images...")
        results = processor.process_multiple_images(num_synthetic=3)

if __name__ == "__main__":
    main()