from src.data.image import basic, common, voronoi


IMAGE_TYPES = {
    'basic': basic.create_image,
    'triangle': basic.create_image_shot_angle_only,
    'voronoi': voronoi.create_image_voronoi,
    'noisy_voronoi': voronoi.create_image_voronoi_noisy,
    'minimal_voronoi': voronoi.create_image_minimal_voronoi,
    # 'bubbles': basic.create_image_opponent_bubbles,
    # 'cropped_voronoi': voronoi.create_image_voronoi_cropped,
}
