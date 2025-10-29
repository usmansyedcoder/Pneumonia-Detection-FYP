from PIL import Image, ImageDraw, ImageFont

def add_watermark(input_path, output_path, watermark_text="SECURED"):
    image = Image.open(input_path).convert("RGB")
    width, height = image.size

    # Draw object
    draw = ImageDraw.Draw(image)

    # Smaller font size (3% of image height)
    font_size = int(height * 0.03)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # ✅ Use textbbox (works in Pillow ≥10)
    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Position: bottom-left corner
    margin = 10
    x = margin
    y = height - text_h - margin

    # Optional shadow for readability
    shadow_offset = 2
    draw.text((x + shadow_offset, y + shadow_offset), watermark_text, font=font, fill=(0, 0, 0, 200))

    # White text
    draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 200))

    # Save
    image.save(output_path)
