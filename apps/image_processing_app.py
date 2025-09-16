import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io

def run():
    st.title("🖼️ Image Processing App")
    st.markdown("---")
    
    # Image upload
    st.sidebar.header("📁 Image Upload")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload an image to apply various filters and transformations"
    )
    
    # Default sample image option
    use_sample = st.sidebar.checkbox("Use sample image", value=True if uploaded_file is None else False)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.sidebar.success("✅ Image uploaded successfully!")
    elif use_sample:
        # Create a sample gradient image
        image = create_sample_image()
        st.sidebar.info("📸 Using sample image")
    else:
        st.info("👆 Please upload an image or use the sample image to get started!")
        return
    
    # Display original image info
    st.subheader("📷 Original Image")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    
    with col2:
        st.markdown("### Image Information")
        st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
        st.write(f"**Mode:** {image.mode}")
        st.write(f"**Format:** {getattr(image, 'format', 'Unknown')}")
        
        # File size (if uploaded)
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue())
            st.write(f"**File Size:** {file_size / 1024:.1f} KB")
    
    st.markdown("---")
    
    # Processing options
    st.subheader("🛠️ Image Processing Options")
    
    processing_type = st.selectbox(
        "Select Processing Type:",
        ["Filters", "Enhancements", "Transformations", "Color Adjustments", "Effects"]
    )
    
    processed_image = image.copy()
    
    if processing_type == "Filters":
        processed_image = apply_filters(image)
    elif processing_type == "Enhancements":
        processed_image = apply_enhancements(image)
    elif processing_type == "Transformations":
        processed_image = apply_transformations(image)
    elif processing_type == "Color Adjustments":
        processed_image = apply_color_adjustments(image)
    elif processing_type == "Effects":
        processed_image = apply_effects(image)
    
    # Display processed image
    if processed_image is not None:
        st.markdown("---")
        st.subheader("✨ Processed Image")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original")
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("#### Processed")
            st.image(processed_image, use_column_width=True)
        
        # Download option
        st.markdown("---")
        st.subheader("💾 Download Processed Image")
        
        # Convert image to bytes for download
        img_buffer = io.BytesIO()
        processed_image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Processed Image",
            data=img_bytes,
            file_name="processed_image.png",
            mime="image/png"
        )

def create_sample_image():
    """Create a colorful sample image for demonstration"""
    width, height = 400, 300
    
    # Create a gradient image
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            # Create a colorful gradient pattern
            r = int(255 * (i / height))
            g = int(255 * (j / width))
            b = int(255 * ((i + j) / (height + width)))
            image_array[i, j] = [r, g, b]
    
    # Add some geometric shapes
    center_x, center_y = width // 2, height // 2
    for i in range(height):
        for j in range(width):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if 50 < dist < 70:
                image_array[i, j] = [255, 255, 255]  # White circle
            elif dist < 30:
                image_array[i, j] = [255, 0, 0]  # Red center
    
    return Image.fromarray(image_array)

def apply_filters(image):
    """Apply various filters to the image"""
    filter_type = st.selectbox(
        "Select Filter:",
        ["None", "Blur", "Sharpen", "Edge Enhance", "Edge Enhance More", "Emboss", "Find Edges", "Smooth", "Smooth More"]
    )
    
    if filter_type == "None":
        return image
    elif filter_type == "Blur":
        blur_radius = st.slider("Blur Radius:", 0.1, 5.0, 1.0, 0.1)
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    elif filter_type == "Sharpen":
        return image.filter(ImageFilter.SHARPEN)
    elif filter_type == "Edge Enhance":
        return image.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_type == "Edge Enhance More":
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif filter_type == "Emboss":
        return image.filter(ImageFilter.EMBOSS)
    elif filter_type == "Find Edges":
        return image.filter(ImageFilter.FIND_EDGES)
    elif filter_type == "Smooth":
        return image.filter(ImageFilter.SMOOTH)
    elif filter_type == "Smooth More":
        return image.filter(ImageFilter.SMOOTH_MORE)

def apply_enhancements(image):
    """Apply various enhancements to the image"""
    st.markdown("### Adjustment Controls")
    
    # Brightness
    brightness = st.slider("Brightness:", 0.1, 3.0, 1.0, 0.1)
    
    # Contrast
    contrast = st.slider("Contrast:", 0.1, 3.0, 1.0, 0.1)
    
    # Saturation
    saturation = st.slider("Saturation:", 0.0, 3.0, 1.0, 0.1)
    
    # Sharpness
    sharpness = st.slider("Sharpness:", 0.0, 3.0, 1.0, 0.1)
    
    # Apply enhancements
    enhanced_image = image
    
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(enhanced_image)
        enhanced_image = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(enhanced_image)
        enhanced_image = enhancer.enhance(saturation)
    
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(enhanced_image)
        enhanced_image = enhancer.enhance(sharpness)
    
    return enhanced_image

def apply_transformations(image):
    """Apply geometric transformations to the image"""
    transform_type = st.selectbox(
        "Select Transformation:",
        ["None", "Rotate", "Flip Horizontal", "Flip Vertical", "Resize", "Crop"]
    )
    
    if transform_type == "None":
        return image
    elif transform_type == "Rotate":
        angle = st.slider("Rotation Angle (degrees):", -180, 180, 0, 1)
        return image.rotate(angle, expand=True)
    elif transform_type == "Flip Horizontal":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform_type == "Flip Vertical":
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    elif transform_type == "Resize":
        scale_factor = st.slider("Scale Factor:", 0.1, 3.0, 1.0, 0.1)
        new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    elif transform_type == "Crop":
        crop_percent = st.slider("Crop Percentage:", 0, 50, 10)
        width, height = image.size
        crop_pixels = int(min(width, height) * crop_percent / 100)
        
        left = crop_pixels
        top = crop_pixels
        right = width - crop_pixels
        bottom = height - crop_pixels
        
        if right > left and bottom > top:
            return image.crop((left, top, right, bottom))
        else:
            st.warning("Crop percentage too large!")
            return image

def apply_color_adjustments(image):
    """Apply color adjustments to the image"""
    adjustment_type = st.selectbox(
        "Select Color Adjustment:",
        ["None", "Grayscale", "Sepia", "Invert", "Posterize", "Solarize"]
    )
    
    if adjustment_type == "None":
        return image
    elif adjustment_type == "Grayscale":
        return ImageOps.grayscale(image)
    elif adjustment_type == "Sepia":
        return apply_sepia_effect(image)
    elif adjustment_type == "Invert":
        return ImageOps.invert(image)
    elif adjustment_type == "Posterize":
        bits = st.slider("Posterize Bits:", 1, 8, 4)
        return ImageOps.posterize(image, bits)
    elif adjustment_type == "Solarize":
        threshold = st.slider("Solarize Threshold:", 0, 255, 128)
        return ImageOps.solarize(image, threshold)

def apply_sepia_effect(image):
    """Apply sepia effect to image"""
    # Convert to numpy array
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:  # Color image
        # Sepia transformation matrix
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        sepia_img = img_array.dot(sepia_filter.T)
        sepia_img = np.clip(sepia_img, 0, 255)
        return Image.fromarray(sepia_img.astype(np.uint8))
    else:
        return image

def apply_effects(image):
    """Apply special effects to the image"""
    effect_type = st.selectbox(
        "Select Effect:",
        ["None", "Vintage", "Cold", "Warm", "High Contrast B&W", "Soft Focus"]
    )
    
    if effect_type == "None":
        return image
    elif effect_type == "Vintage":
        return apply_vintage_effect(image)
    elif effect_type == "Cold":
        return apply_color_temperature(image, "cold")
    elif effect_type == "Warm":
        return apply_color_temperature(image, "warm")
    elif effect_type == "High Contrast B&W":
        gray = ImageOps.grayscale(image)
        return ImageOps.autocontrast(gray)
    elif effect_type == "Soft Focus":
        blur_radius = st.slider("Soft Focus Intensity:", 0.5, 5.0, 2.0, 0.5)
        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return Image.blend(image, blurred, 0.5)

def apply_vintage_effect(image):
    """Apply a vintage photo effect"""
    # Reduce saturation
    enhancer = ImageEnhance.Color(image)
    vintage = enhancer.enhance(0.7)
    
    # Add sepia tint
    vintage = apply_sepia_effect(vintage)
    
    # Slightly reduce contrast
    enhancer = ImageEnhance.Contrast(vintage)
    vintage = enhancer.enhance(0.9)
    
    # Add slight blur for softer look
    vintage = vintage.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return vintage

def apply_color_temperature(image, temperature):
    """Apply color temperature adjustment"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:  # Color image
        if temperature == "warm":
            # Increase red, decrease blue
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.1, 0, 255)  # Red
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.9, 0, 255)  # Blue
        elif temperature == "cold":
            # Decrease red, increase blue
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.9, 0, 255)  # Red
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.1, 0, 255)  # Blue
        
        return Image.fromarray(img_array.astype(np.uint8))
    else:
        return image

if __name__ == "__main__":
    run()