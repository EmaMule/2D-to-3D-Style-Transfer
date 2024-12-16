# Helper function to blend image with background
def apply_background(rendered_image, mask, background):
    return rendered_image * mask + background * (1 - mask)

